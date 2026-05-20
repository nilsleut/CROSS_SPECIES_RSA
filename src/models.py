"""
models.py — Learning rule CNNs ported from Paper 1 (learning_rules_v8.ipynb).

Architecture: custom 3-conv CNN (C1=32, C2=64, C3=128, FC1=512) trained on CIFAR-10.
Weights loaded from Paper 1 saved checkpoints for methodological consistency.

Ported faithfully from v8; no ResNet-50.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# ============================================================
# Architecture constants (must match Paper 1 / v8 exactly)
# ============================================================
C1, C2, C3 = 32, 64, 128
FC1_DIM    = 512
N_CLS      = 10
FEAT_SIZE  = 4                          # adaptive pool target for FC input
FC1_IN     = C3 * FEAT_SIZE * FEAT_SIZE  # 2048

# PC hyperparams
T_INF = 10
LR_R  = 0.02
LR_W  = 1e-4

# STDP hyperparams
A_P   = 0.003
A_M   = 0.003
TAU_P = 20.0
TAU_M = 20.0
T_SIM = 10

# Inference transform — CIFAR-10 normalization stats (same as Paper 1 tf_things)
THINGS_TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

# Fixed layer-to-ROI mapping (same as Paper 1 LAYER_ROI_FIXED)
LAYER_ROI_FREEMANZIEMBA = {"Conv1": ["V1", "V2"]}
LAYER_ROI_MAJAJHONG     = {"Conv2": ["V4"], "FC1": ["IT"]}

# Default Paper 1 weights directory
_DEFAULT_WEIGHTS_DIR = Path(
    r"C:/Users/nilsl/Desktop/Projekte/learning-rules-rsa/outputs"
)


# ============================================================
# Shared helper
# ============================================================

def _make_conv_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


def _pool_for_fc(c3: torch.Tensor) -> torch.Tensor:
    """Adaptive pool conv3 output to fixed FEAT_SIZE×FEAT_SIZE before FC."""
    return F.adaptive_avg_pool2d(c3, FEAT_SIZE)


# ============================================================
# Random (untrained baseline)
# ============================================================

class Random_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = _make_conv_block(3,  C1)
        self.conv2 = _make_conv_block(C1, C2)
        self.conv3 = _make_conv_block(C2, C3)
        self.fc1   = nn.Linear(FC1_IN, FC1_DIM)
        self.fc2   = nn.Linear(FC1_DIM, N_CLS)

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        return self.fc2(F.relu(self.fc1(x.view(x.size(0), -1))))

    def get_features(self, x):
        with torch.no_grad():
            c1 = self.conv1(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            h1 = F.relu(self.fc1(_pool_for_fc(c3).view(c3.size(0), -1)))
        return c1.mean([2, 3]), c2.mean([2, 3]), c3.mean([2, 3]), h1


# ============================================================
# Backprop (standard cross-entropy + Adam)
# ============================================================

class BP_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = _make_conv_block(3,  C1)
        self.conv2 = _make_conv_block(C1, C2)
        self.conv3 = _make_conv_block(C2, C3)
        self.fc1   = nn.Linear(FC1_IN, FC1_DIM)
        self.fc2   = nn.Linear(FC1_DIM, N_CLS)
        self.drop  = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        return self.fc2(self.drop(F.relu(self.fc1(x.view(x.size(0), -1)))))

    def get_features(self, x):
        with torch.no_grad():
            c1 = self.conv1(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            h1 = F.relu(self.fc1(_pool_for_fc(c3).view(c3.size(0), -1)))
        return c1.mean([2, 3]), c2.mean([2, 3]), c3.mean([2, 3]), h1


# ============================================================
# Feedback Alignment (random fixed backward weights)
# ============================================================

class _FAConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, b, B_feedback, stride, padding):
        ctx.save_for_backward(x, W, b, B_feedback)
        ctx.stride  = stride
        ctx.padding = padding
        return F.conv2d(x, W, b, stride=stride, padding=padding)

    @staticmethod
    def backward(ctx, grad_output):
        x, W, b, B = ctx.saved_tensors
        stride, padding = ctx.stride, ctx.padding
        grad_W = torch.nn.grad.conv2d_weight(
            x, W.shape, grad_output, stride=stride, padding=padding
        )
        grad_b = grad_output.sum([0, 2, 3]) if b is not None else None
        grad_x = F.conv_transpose2d(grad_output, B, stride=stride, padding=padding)
        return grad_x, grad_W, grad_b, None, None, None


class FAConv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.W = nn.Parameter(
            nn.init.kaiming_normal_(
                torch.empty(out_c, in_c, kernel_size, kernel_size)
            )
        )
        self.b = nn.Parameter(torch.zeros(out_c))
        B = nn.init.xavier_normal_(
            torch.randn(out_c, in_c, kernel_size, kernel_size)
        )
        self.register_buffer("B_feedback", B)
        self.stride  = stride
        self.padding = padding

    def forward(self, x):
        return _FAConvFunction.apply(
            x, self.W, self.b, self.B_feedback, self.stride, self.padding
        )


def _make_fa_conv_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        FAConv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class FA_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = _make_fa_conv_block(3,  C1)
        self.conv2 = _make_fa_conv_block(C1, C2)
        self.conv3 = _make_fa_conv_block(C2, C3)
        self.fc1   = nn.Linear(FC1_IN, FC1_DIM)
        self.fc2   = nn.Linear(FC1_DIM, N_CLS)
        self.drop  = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        return self.fc2(self.drop(F.relu(self.fc1(x.view(x.size(0), -1)))))

    def get_features(self, x):
        with torch.no_grad():
            c1 = self.conv1(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            h1 = F.relu(self.fc1(_pool_for_fc(c3).view(c3.size(0), -1)))
        return c1.mean([2, 3]), c2.mean([2, 3]), c3.mean([2, 3]), h1


# ============================================================
# Predictive Coding (Whittington & Bogacz 2017)
# ============================================================

class PC_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1  = nn.Conv2d(3,  C1, 3, padding=1, bias=False)
        self.W2  = nn.Conv2d(C1, C2, 3, padding=1, bias=False)
        self.W3  = nn.Conv2d(C2, C3, 3, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2)
        self.bn1  = nn.BatchNorm2d(C1)
        self.bn2  = nn.BatchNorm2d(C2)
        self.bn3  = nn.BatchNorm2d(C3)
        self.P2   = nn.ConvTranspose2d(C2, C1, 3, padding=1, bias=False)
        self.P3   = nn.ConvTranspose2d(C3, C2, 3, padding=1, bias=False)
        self.fc1  = nn.Linear(FC1_IN, FC1_DIM)
        self.fc2  = nn.Linear(FC1_DIM, N_CLS)
        self.clf_opt = None  # set by _make_opt() before training

    def _make_opt(self):
        self.clf_opt = torch.optim.Adam(
            list(self.fc1.parameters()) + list(self.fc2.parameters()), lr=1e-3
        )

    def infer(self, x):
        with torch.no_grad():
            r1 = self.pool(F.relu(self.bn1(self.W1(x))))
            r2 = self.pool(F.relu(self.bn2(self.W2(r1))))
            r3 = self.pool(F.relu(self.bn3(self.W3(r2))))
            for _ in range(T_INF):
                pred1 = torch.tanh(F.interpolate(self.P2(r2), size=r1.shape[2:], mode="nearest"))
                pred2 = torch.tanh(F.interpolate(self.P3(r3), size=r2.shape[2:], mode="nearest"))
                e1 = r1 - pred1
                e2 = r2 - pred2
                r1 = F.relu(r1 + LR_R * (-e1))
                r2 = F.relu(r2 + LR_R * (-e2 + F.avg_pool2d(F.conv2d(e1, self.W2.weight, padding=1), 2)))
                r3 = F.relu(r3 + LR_R * F.avg_pool2d(F.conv2d(e2, self.W3.weight, padding=1), 2))
        return r1, r2, r3

    def weight_update(self, x, r1, r2, r3):
        with torch.no_grad():
            r1i = self.pool(F.relu(self.bn1(self.W1(x))))
            r2i = self.pool(F.relu(self.bn2(self.W2(r1i))))
            e1 = (r1 - r1i).clamp(-0.5, 0.5)
            e2 = (r2 - r2i).clamp(-0.5, 0.5)
            dW1 = (
                e1.mean([0, 2, 3]).unsqueeze(1) * x.mean([0, 2, 3]).unsqueeze(0)
            ).unsqueeze(-1).unsqueeze(-1).expand_as(self.W1.weight).clamp(-0.01, 0.01)
            self.W1.weight.data += LR_W * dW1
            dW2 = (
                e2.mean([0, 2, 3]).unsqueeze(1) * r1.mean([0, 2, 3]).unsqueeze(0)
            ).unsqueeze(-1).unsqueeze(-1).expand_as(self.W2.weight).clamp(-0.01, 0.01)
            self.W2.weight.data += LR_W * dW2

    def get_features(self, x):
        with torch.no_grad():
            r1, r2, r3 = self.infer(x)
            h1 = F.relu(self.fc1(_pool_for_fc(r3).view(r3.size(0), -1)))
        return r1.mean([2, 3]), r2.mean([2, 3]), r3.mean([2, 3]), h1

    def step(self, x, y):
        r1, r2, r3 = self.infer(x)
        self.weight_update(x, r1, r2, r3)
        self.clf_opt.zero_grad()
        logit = self.fc2(F.relu(self.fc1(r3.detach().view(r3.size(0), -1))))
        loss  = F.cross_entropy(logit, y)
        loss.backward()
        self.clf_opt.step()
        return loss.item(), (logit.argmax(1) == y).float().mean().item()

    def eval(self):
        return super().eval()  # sets BN/Dropout submodules to eval mode


# ============================================================
# STDP (Tavanaei et al. 2019 approximation)
# ============================================================

class STDP_Conv:
    """Single STDP conv layer (plain Python class — not nn.Module)."""

    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3, padding: int = 1):
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, padding=padding, bias=False)
        nn.init.kaiming_normal_(self.conv.weight)

    def poisson_spikes(self, rates: torch.Tensor) -> torch.Tensor:
        r = rates.clamp(0, 1).unsqueeze(-1).expand(*rates.shape, T_SIM)
        return (torch.rand_like(r) < r / T_SIM).float()

    def first_spike(self, spikes: torch.Tensor) -> torch.Tensor:
        has = spikes.any(-1)
        t   = torch.argmax(spikes, dim=-1).float()
        t[~has] = float(T_SIM + 1)
        return t

    def stdp_update(self, pre_act: torch.Tensor, post_act: torch.Tensor, lr: float = 5e-4):
        with torch.no_grad():
            pre_s  = self.poisson_spikes(torch.sigmoid(pre_act))
            post_s = self.poisson_spikes(torch.sigmoid(post_act))
            t_pre  = self.first_spike(pre_s)
            t_post = self.first_spike(post_s)
            dt = t_post.unsqueeze(1) - t_pre.unsqueeze(2)
            dW = (
                A_P * torch.exp(-dt.clamp(min=0) / TAU_P)
                - A_M * torch.exp( dt.clamp(max=0) / TAU_M)
            ).mean(0).clamp(-0.002, 0.002)
            dW_conv = dW.T.view(
                self.conv.weight.size(0), self.conv.weight.size(1), 1, 1
            ).expand_as(self.conv.weight)
            self.conv.weight.data += lr * dW_conv
            self.conv.weight.data.clamp_(-1.0, 1.0)

    def forward(self, x: torch.Tensor, do_stdp: bool = False,
                pre_act: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = F.relu(self.conv(x))
        if do_stdp and pre_act is not None:
            self.stdp_update(pre_act, out.mean([2, 3]))
        return out


class STDP_CNN:
    """Full STDP network (plain Python class — uses to_device() instead of .to())."""

    def __init__(self):
        self.L1   = STDP_Conv(3,  C1)
        self.L2   = STDP_Conv(C1, C2)
        self.L3   = STDP_Conv(C2, C3)
        self.pool = nn.MaxPool2d(2)
        self.bn1  = nn.BatchNorm2d(C1)
        self.bn2  = nn.BatchNorm2d(C2)
        self.bn3  = nn.BatchNorm2d(C3)
        self.fc1  = nn.Linear(FC1_IN, FC1_DIM)
        self.fc2  = nn.Linear(FC1_DIM, N_CLS)
        self.clf_opt = None

    def to_device(self, device: torch.device) -> "STDP_CNN":
        for attr in ("pool", "bn1", "bn2", "bn3", "fc1", "fc2"):
            setattr(self, attr, getattr(self, attr).to(device))
        for layer in (self.L1, self.L2, self.L3):
            layer.conv = layer.conv.to(device)
        self.clf_opt = torch.optim.Adam(
            list(self.fc1.parameters()) + list(self.fc2.parameters()) +
            [self.bn1.weight, self.bn1.bias,
             self.bn2.weight, self.bn2.bias,
             self.bn3.weight, self.bn3.bias],
            lr=1e-3,
        )
        return self

    def _forward(self, x: torch.Tensor, do_stdp: bool = False):
        c1 = self.pool(F.relu(self.bn1(self.L1.forward(x,  do_stdp, x.mean([2, 3])))))
        c2 = self.pool(F.relu(self.bn2(self.L2.forward(c1, do_stdp, c1.mean([2, 3])))))
        c3 = self.pool(F.relu(self.bn3(self.L3.forward(c2, do_stdp, c2.mean([2, 3])))))
        return c1, c2, c3

    def get_features(self, x: torch.Tensor):
        with torch.no_grad():
            c1, c2, c3 = self._forward(x, do_stdp=False)
            h1 = F.relu(self.fc1(_pool_for_fc(c3).view(c3.size(0), -1)))
        return c1.mean([2, 3]), c2.mean([2, 3]), c3.mean([2, 3]), h1

    def step(self, x: torch.Tensor, y: torch.Tensor):
        self._forward(x, do_stdp=True)
        self.clf_opt.zero_grad()
        _, _, c3b = self._forward(x, do_stdp=False)
        logit = self.fc2(F.relu(self.fc1(c3b.view(c3b.size(0), -1))))
        loss  = F.cross_entropy(logit, y)
        loss.backward()
        self.clf_opt.step()
        return loss.detach().item(), (logit.argmax(1) == y).float().mean().item()

    def eval(self):
        pass  # no-op


# ============================================================
# Weight loading
# ============================================================

_WEIGHT_FILES = {
    "backprop":           "model_weights_backprop.pt",
    "feedback_alignment": "model_weights_feedback_alignment.pt",
    "predictive_coding":  "model_weights_predictive_coding.pt",
    "stdp":               "model_weights_stdp.pt",
    "random":             "model_weights_random_weights.pt",
}

_MODEL_CLASSES = {
    "backprop":           BP_CNN,
    "feedback_alignment": FA_CNN,
    "predictive_coding":  PC_CNN,
    "random":             Random_CNN,
}


def load_model(rule: str,
               weights_dir: Optional[Path] = None) -> object:
    """
    Create the correct CNN for `rule` and load Paper 1 saved weights.
    Falls back to random init if the .pt file is missing.
    """
    if weights_dir is None:
        weights_dir = _DEFAULT_WEIGHTS_DIR
    weights_dir = Path(weights_dir)

    if rule not in _WEIGHT_FILES:
        raise ValueError(f"Unknown learning rule '{rule}'. "
                         f"Valid: {list(_WEIGHT_FILES.keys())}")

    wf = weights_dir / _WEIGHT_FILES[rule]

    if rule == "stdp":
        model = STDP_CNN().to_device(torch.device("cpu"))
        # Prefer _seed0.pt (full retrain, all layers) over original .pt (3 conv only)
        wf_full = weights_dir / _WEIGHT_FILES[rule].replace(".pt", "_seed0.pt")
        wf_load = wf_full if wf_full.exists() else wf
        if wf_load.exists():
            d = torch.load(wf_load, map_location="cpu", weights_only=True)
            model.L1.conv.weight.data = d["conv1.weight"]
            model.L2.conv.weight.data = d["conv2.weight"]
            model.L3.conv.weight.data = d["conv3.weight"]
            if "fc1.weight" in d:
                model.bn1.weight.data = d["bn1.weight"]; model.bn1.bias.data = d["bn1.bias"]
                model.bn1.running_mean.copy_(d["bn1.running_mean"])
                model.bn1.running_var.copy_(d["bn1.running_var"])
                model.bn2.weight.data = d["bn2.weight"]; model.bn2.bias.data = d["bn2.bias"]
                model.bn2.running_mean.copy_(d["bn2.running_mean"])
                model.bn2.running_var.copy_(d["bn2.running_var"])
                model.bn3.weight.data = d["bn3.weight"]; model.bn3.bias.data = d["bn3.bias"]
                model.bn3.running_mean.copy_(d["bn3.running_mean"])
                model.bn3.running_var.copy_(d["bn3.running_var"])
                model.fc1.weight.data = d["fc1.weight"]; model.fc1.bias.data = d["fc1.bias"]
                model.fc2.weight.data = d["fc2.weight"]; model.fc2.bias.data = d["fc2.bias"]
                logger.info(f"  Loaded full STDP weights from {wf_load.name}")
            else:
                logger.info(f"  Loaded STDP conv weights from {wf_load.name}")
                logger.warning(
                    "  STDP: FC1/BN weights not in checkpoint. "
                    "Run: python scripts/train_additional_seeds.py --rules stdp --seeds 0"
                )
        else:
            logger.warning(f"  STDP weights not found at {wf_load} — using random init")
        return model

    cls = _MODEL_CLASSES[rule]
    model = cls()
    if wf.exists():
        state = torch.load(wf, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        logger.info(f"  Loaded {rule} weights from {wf.name}")
    else:
        logger.warning(f"  Weights not found at {wf} — using random init for {rule}")
    model.eval()
    return model


# ============================================================
# Feature extraction
# ============================================================

def extract_features(
    model,
    image_paths: List[str],
    transform=None,
    batch_size: int = 64,
) -> Dict[str, np.ndarray]:
    """
    Run model.get_features() on a list of image paths.

    Returns dict with keys "Conv1", "Conv2", "Conv3", "FC1",
    each an ndarray of shape (n_images, n_units).
    """
    from torch.utils.data import DataLoader, Dataset

    if transform is None:
        transform = THINGS_TRANSFORM

    class _ImgDS(Dataset):
        def __init__(self, paths, t):
            self.paths, self.t = paths, t
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, i):
            return self.t(Image.open(self.paths[i]).convert("RGB")), i

    loader = DataLoader(
        _ImgDS(image_paths, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    c1s, c2s, c3s, h1s = [], [], [], []

    if hasattr(model, "eval") and callable(model.eval):
        model.eval()

    for imgs, _ in loader:
        c1, c2, c3, h1 = model.get_features(imgs)
        def _np(t):
            return t.detach().cpu().numpy() if torch.is_tensor(t) else np.array(t)
        c1s.append(_np(c1))
        c2s.append(_np(c2))
        c3s.append(_np(c3))
        h1s.append(_np(h1))

    return {
        "Conv1": np.concatenate(c1s),
        "Conv2": np.concatenate(c2s),
        "Conv3": np.concatenate(c3s),
        "FC1":   np.concatenate(h1s),
    }


# ============================================================
# Legacy shims (keep run_pipeline.py import line working)
# ============================================================

def create_model(learning_rule: str, pretrained: bool = True, **kwargs):
    """Deprecated shim — use load_model() instead."""
    return load_model(learning_rule)


class FeatureExtractor:
    """Deprecated shim — use extract_features() instead."""
    def __init__(self, model, layer_names):
        self.model = model
        self.layer_names = layer_names
    def extract_from_image_paths(self, paths, **kw):
        return extract_features(self.model, paths)
    def cleanup(self):
        pass


def get_trainer(learning_rule: str, model, **kwargs):
    """Deprecated — training done offline; weights loaded via load_model()."""
    return None


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    for rule in ["random", "backprop", "feedback_alignment", "predictive_coding", "stdp"]:
        m = load_model(rule)
        dummy = torch.randn(4, 3, 224, 224)
        c1, c2, c3, h1 = m.get_features(dummy)
        print(f"{rule:20s}  Conv1={tuple(c1.shape)}  Conv2={tuple(c2.shape)}  "
              f"Conv3={tuple(c3.shape)}  FC1={tuple(h1.shape)}")
