#!/usr/bin/env python3
"""
train_additional_seeds.py — Train seeds 1-4 for all 5 learning rules.

Paper 1 (v8) only saved weights for seed 0 (seed=42).
This script trains seeds 1-4 using the EXACT same hyperparameters as v8:
  - CIFAR-10 subset: 8000 samples (fixed subset, same as all seeds in v8)
  - N_EPOCHS = 40, BATCH = 64
  - BP:   Adam, lr=1e-3, weight_decay=1e-4, CosineAnnealingLR, grad_clip=1
  - FA:   SGD,  lr=5e-4, momentum=0.9,      CosineAnnealingLR, grad_clip=1
  - PC:   internal clf_opt (Adam fc1+fc2, lr=1e-3), conv weights via PC rule
  - STDP: STDP conv updates + Adam clf (fc1+fc2+BNs, lr=1e-3)
  - Random: no training (just re-seed initialization)

Saves to Paper 1 weights dir:
  model_weights_{rule}_seed1.pt  (seed=123)
  model_weights_{rule}_seed2.pt  (seed=456)
  model_weights_{rule}_seed3.pt  (seed=789)
  model_weights_{rule}_seed4.pt  (seed=1337)

Runtime estimate (CPU, no GPU):
  BP:     ~15 min per seed
  FA:     ~15 min per seed
  PC:     ~30-60 min per seed  (inference loop is slow on CPU)
  STDP:   ~30-60 min per seed
  Random: <1 s per seed
  Total:  ~4-8 hours — consider running on Kaggle GPU

Usage:
  python scripts/train_additional_seeds.py [--rules bp fa pc stdp random]
"""

import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from models import (
    BP_CNN, FA_CNN, PC_CNN, STDP_CNN, Random_CNN,
    _DEFAULT_WEIGHTS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("train_seeds")

# ── Constants (must match Paper 1 v8 exactly) ──────────────────────────────────
N_EPOCHS = 40
BATCH    = 64
LR       = 1e-3
N_CIFAR  = 8000
SEEDS    = [42, 123, 456, 789, 1337]   # seed 0 = 42 (already saved), 1-4 need training

WEIGHT_NAMES = {
    "backprop":           "model_weights_backprop",
    "feedback_alignment": "model_weights_feedback_alignment",
    "predictive_coding":  "model_weights_predictive_coding",
    "stdp":               "model_weights_stdp",
    "random":             "model_weights_random_weights",
}


# ── CIFAR-10 data ──────────────────────────────────────────────────────────────

def make_cifar_loader(data_dir: Path) -> DataLoader:
    """
    8000-sample CIFAR-10 subset. Subset index fixed with seed=42 (same as v8).
    Shuffle order within each epoch will vary with torch global seed.
    """
    tf = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    # Subset index is derived with seed=42, identical to v8 — FIXED for all seeds
    torch.manual_seed(42)
    full = torchvision.datasets.CIFAR10(str(data_dir), train=True, download=True, transform=tf)
    idx  = torch.randperm(len(full))[:N_CIFAR].tolist()
    loader = DataLoader(Subset(full, idx), batch_size=BATCH,
                        shuffle=True, num_workers=0, drop_last=True)
    logger.info(f"  CIFAR-10: {N_CIFAR} samples, {len(loader)} batches/epoch")
    return loader


# ── Training loops (exact v8 hyperparameters) ─────────────────────────────────

def _clip_and_step(opt, model):
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()


def train_bp(loader: DataLoader) -> BP_CNN:
    model = BP_CNN()
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EPOCHS)
    for epoch in range(N_EPOCHS):
        model.train()
        tl, tc, tn = 0.0, 0, 0
        for x, y in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            _clip_and_step(opt, model)
            tl += loss.item(); tc += (model(x).argmax(1) == y).sum().item(); tn += y.size(0)
        sched.step()
        if (epoch + 1) % 10 == 0:
            logger.info(f"    Epoch {epoch+1:3d}: loss={tl/len(loader):.4f}")
    model.eval()
    return model


def train_fa(loader: DataLoader) -> FA_CNN:
    model = FA_CNN()
    opt   = torch.optim.SGD(model.parameters(), lr=LR * 0.5,
                            momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EPOCHS)
    for epoch in range(N_EPOCHS):
        model.train()
        tl = 0.0
        for x, y in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            _clip_and_step(opt, model)
            tl += loss.item()
        sched.step()
        if (epoch + 1) % 10 == 0:
            logger.info(f"    Epoch {epoch+1:3d}: loss={tl/len(loader):.4f}")
    model.eval()
    return model


def train_pc(loader: DataLoader) -> PC_CNN:
    model = PC_CNN()
    model._make_opt()   # creates clf_opt on Adam(fc1+fc2)
    for epoch in range(N_EPOCHS):
        tl, ta, n = 0.0, 0.0, 0
        for x, y in loader:
            loss, acc = model.step(x, y)
            tl += loss; ta += acc; n += 1
        if (epoch + 1) % 10 == 0:
            logger.info(f"    Epoch {epoch+1:3d}: loss={tl/n:.4f}  acc={ta/n:.3f}")
    return model


def train_stdp(loader: DataLoader) -> STDP_CNN:
    model = STDP_CNN().to_device(torch.device("cpu"))
    for epoch in range(N_EPOCHS):
        tl, ta, n = 0.0, 0.0, 0
        for x, y in loader:
            loss, acc = model.step(x, y)
            tl += loss; ta += acc; n += 1
        if (epoch + 1) % 10 == 0:
            logger.info(f"    Epoch {epoch+1:3d}: loss={tl/n:.4f}  acc={ta/n:.3f}")
    return model


def init_random() -> Random_CNN:
    model = Random_CNN()
    model.eval()
    return model


TRAIN_FNS = {
    "backprop":           train_bp,
    "feedback_alignment": train_fa,
    "predictive_coding":  train_pc,
    "stdp":               train_stdp,
    "random":             lambda _loader: init_random(),
}


# ── Weight saving ──────────────────────────────────────────────────────────────

def save_weights(model, rule: str, seed_idx: int, weights_dir: Path):
    stem = WEIGHT_NAMES[rule]
    path = weights_dir / f"{stem}_seed{seed_idx}.pt"
    if rule == "stdp":
        # Save full STDP state: conv weights + BN params + FC layers.
        # STDP_CNN is not nn.Module so we build the dict manually.
        # Loading code detects full vs. 3-conv-only by checking for "fc1.weight".
        state = {
            "conv1.weight": model.L1.conv.weight.data.clone(),
            "conv2.weight": model.L2.conv.weight.data.clone(),
            "conv3.weight": model.L3.conv.weight.data.clone(),
            "bn1.weight":   model.bn1.weight.data.clone(),
            "bn1.bias":     model.bn1.bias.data.clone(),
            "bn1.running_mean": model.bn1.running_mean.clone(),
            "bn1.running_var":  model.bn1.running_var.clone(),
            "bn2.weight":   model.bn2.weight.data.clone(),
            "bn2.bias":     model.bn2.bias.data.clone(),
            "bn2.running_mean": model.bn2.running_mean.clone(),
            "bn2.running_var":  model.bn2.running_var.clone(),
            "bn3.weight":   model.bn3.weight.data.clone(),
            "bn3.bias":     model.bn3.bias.data.clone(),
            "bn3.running_mean": model.bn3.running_mean.clone(),
            "bn3.running_var":  model.bn3.running_var.clone(),
            "fc1.weight":   model.fc1.weight.data.clone(),
            "fc1.bias":     model.fc1.bias.data.clone(),
            "fc2.weight":   model.fc2.weight.data.clone(),
            "fc2.bias":     model.fc2.bias.data.clone(),
        }
        torch.save(state, path)
    else:
        torch.save(model.state_dict(), path)
    logger.info(f"    Saved → {path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", nargs="+",
                        choices=list(TRAIN_FNS.keys()),
                        default=list(TRAIN_FNS.keys()),
                        help="Which rules to train (default: all)")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[1, 2, 3, 4],
                        help="Which seed indices to train (default: 1 2 3 4); pass 0 to retrain seed 0 with full state_dict")
    parser.add_argument("--weights-dir", default=None,
                        help="Paper 1 weights directory (default: from models.py)")
    parser.add_argument("--data-dir", default=None,
                        help="CIFAR-10 data directory (default: Paper 1 folder)")
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir) if args.weights_dir else _DEFAULT_WEIGHTS_DIR
    data_dir = Path(args.data_dir) if args.data_dir else weights_dir.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Weights dir : {weights_dir}")
    logger.info(f"CIFAR-10 dir: {data_dir}")
    logger.info(f"Rules       : {args.rules}")
    logger.info(f"Seed indices: {args.seeds}")

    # Prepare CIFAR-10 loader (subset fixed with seed=42)
    logger.info("\nPreparing CIFAR-10 subset (8000 samples, fixed)...")
    loader = make_cifar_loader(data_dir)

    for seed_idx in args.seeds:
        seed = SEEDS[seed_idx]
        logger.info(f"\n{'='*55}")
        logger.info(f"SEED {seed_idx}  (seed={seed})")
        logger.info(f"{'='*55}")

        for rule in args.rules:
            logger.info(f"\n  Rule: {rule}")
            out_path = weights_dir / f"{WEIGHT_NAMES[rule]}_seed{seed_idx}.pt"
            if out_path.exists():
                logger.info(f"    Already exists — skipping: {out_path.name}")
                continue

            # Set global seeds before model creation and training
            torch.manual_seed(seed)
            import numpy as np, random
            np.random.seed(seed)
            random.seed(seed)

            fn = TRAIN_FNS[rule]
            if rule == "random":
                model = fn(None)
            else:
                model = fn(loader)

            save_weights(model, rule, seed_idx, weights_dir)
            del model

    logger.info("\nDone. Run scripts/run_per_seed_rsa.py to compute per-seed RSA.")


if __name__ == "__main__":
    main()
