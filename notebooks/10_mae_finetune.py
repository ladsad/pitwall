import os
import pathlib
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

try:
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = pathlib.Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.mae_model import F1MAE, F1PositionHead, MAEEncoder
from utils.tel_dataset import F1TelemetryDataset
from config import TELEMETRY_CLEAN_PATH, BASE_PATH, TEL_SEASONS, ENCODER_HPARAMS

# ── PATHS ─────────────────────────────────────────────────────────────────────

SILVER_PATH     = str(TELEMETRY_CLEAN_PATH / "silver")
PRETRAIN_CKPT   = str(BASE_PATH / "models" / "mae_checkpoint.pt")
FINETUNE_CKPT   = str(BASE_PATH / "models" / "mae_finetune_checkpoint.pt")
FINETUNED_MODEL = str(BASE_PATH / "models" / "mae_finetuned.pt")

# ── HYPERPARAMETERS ───────────────────────────────────────────────────────────

FINETUNE_CFG = dict(
    # Two-phase schedule
    linear_probe_epochs = 10,   # Phase 1: encoder frozen, train head only
    full_finetune_epochs= 40,   # Phase 2: encoder + head, lower LR
    # LR for each phase
    head_lr             = 1e-3,
    full_lr             = 1e-4,
    weight_decay        = 0.05,
    batch_size          = 128,
    val_fraction        = 0.1,
    num_workers         = 2,
    pin_memory          = True,
    n_classes           = 30,   # race positions 1-25 (supports P21, P22)
)

# ENCODER_HPARAMS imported from config.py — single source of truth shared with 09 and 11.
# These MUST match the checkpoint produced by 09_mae_pretrain.py.

# ── DEVICE ────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── DATASET — LABELLED LAPS ONLY ─────────────────────────────────────────────
# Fine-tuning needs race_position labels. We pull pre-race sessions 
# to ensure the model learns to predict the race without data leakage.

print(f"\nLoading Silver (labelled laps only) from: {SILVER_PATH}")
dataset = F1TelemetryDataset(
    silver_path    = SILVER_PATH,
    seasons        = TEL_SEASONS,
    session_types  = ["FP1", "FP2", "FP3", "Q", "SQ", "S"],   # pre-race sessions
    labelled_only  = True,
)

print(f"\nLabel distribution (position: lap count):")
for pos, count in sorted(dataset.label_distribution().items()):
    bar = "█" * (count // max(1, len(dataset) // 200))
    print(f"  P{pos:02d}: {count:>5}  {bar}")

n_total = len(dataset)
n_val   = max(1, int(n_total * FINETUNE_CFG["val_fraction"]))
n_train = n_total - n_val

train_ds, val_ds = random_split(
    dataset,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(99),
)

print(f"\nSplit: {n_train:,} train | {n_val:,} val laps")

# ── CLASS WEIGHTS — handle position imbalance ─────────────────────────────────
# More laps are recorded for mid-field drivers than backmarkers in practice.
# Inverse-frequency weighting levels the playing field.

dist = dataset.label_distribution()
# dist keys are 1-indexed positions; tensor is 0-indexed classes
class_counts = torch.zeros(FINETUNE_CFG["n_classes"])
for pos, count in dist.items():
    class_idx = pos - 1   # 0-index for CrossEntropyLoss
    if 0 <= class_idx < FINETUNE_CFG["n_classes"]:
        class_counts[class_idx] = count

# Avoid div-by-zero for positions not seen in training data
class_counts = class_counts.clamp(min=1.0)
class_weights = (1.0 / class_counts)
class_weights = class_weights / class_weights.sum() * FINETUNE_CFG["n_classes"]
class_weights = class_weights.to(device)

# ── DATA LOADERS ──────────────────────────────────────────────────────────────

train_loader = DataLoader(
    train_ds,
    batch_size  = FINETUNE_CFG["batch_size"],
    shuffle     = True,
    num_workers = FINETUNE_CFG["num_workers"],
    pin_memory  = FINETUNE_CFG["pin_memory"],
    drop_last   = True,
)
val_loader = DataLoader(
    val_ds,
    batch_size  = FINETUNE_CFG["batch_size"] * 2,
    shuffle     = False,
    num_workers = FINETUNE_CFG["num_workers"],
    pin_memory  = FINETUNE_CFG["pin_memory"],
)

# ── LOAD PRE-TRAINED ENCODER ─────────────────────────────────────────────────

if not os.path.exists(PRETRAIN_CKPT):
    raise FileNotFoundError(
        f"Pre-training checkpoint not found: {PRETRAIN_CKPT}\n"
        f"Run 09_mae_pretrain.py first."
    )

print(f"\nLoading pre-trained checkpoint: {PRETRAIN_CKPT}")
pretrain_ckpt = torch.load(PRETRAIN_CKPT, map_location=device)

mae_model = F1MAE(**ENCODER_HPARAMS)
mae_model.load_state_dict(pretrain_ckpt["model_state"])
print(f"  Pre-training resumed at epoch: {pretrain_ckpt['epoch']}")
print(f"  Pre-training val loss        : {pretrain_ckpt['val_loss']:.6f}")

# Attach the position prediction head
model = F1PositionHead(
    encoder   = mae_model.encoder,
    d_model   = ENCODER_HPARAMS["d_model"],
    n_classes = FINETUNE_CFG["n_classes"],
).to(device)

# ── LOSS ──────────────────────────────────────────────────────────────────────

criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

# ── EVALUATION HELPERS ────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(loader) -> dict[str, float]:
    """
    Returns exact accuracy, top-3 accuracy (predicted position within ±2 of actual),
    and average cross-entropy loss.
    """
    model.eval()
    total_loss   = 0.0
    n_correct    = 0
    n_top3       = 0
    n_total      = 0
    n_batches    = 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        # Filter unlabelled (shouldn't happen with labelled_only=True, but safety check)
        valid = y >= 0
        if not valid.any():
            continue
        x, y = x[valid], y[valid]

        logits = model(x)                                 # (B, 20)
        loss   = criterion(logits, y)
        preds  = logits.argmax(dim=1)                     # 0-indexed class

        n_correct += (preds == y).sum().item()
        n_top3    += (torch.abs(preds - y) <= 2).sum().item()
        n_total   += y.shape[0]
        total_loss += loss.item()
        n_batches  += 1

    return {
        "loss":         total_loss / max(n_batches, 1),
        "exact_acc":    n_correct / max(n_total, 1),
        "top3_acc":     n_top3    / max(n_total, 1),
    }


# ── CHECKPOINT HELPERS ────────────────────────────────────────────────────────

def save_checkpoint(phase: str, epoch: int, metrics: dict, optimizer):
    torch.save(
        {
            "phase":          phase,
            "epoch":          epoch,
            "model_state":    model.state_dict(),
            "optimizer_state":optimizer.state_dict(),
            "metrics":        metrics,
        },
        FINETUNE_CKPT,
    )


def resume_checkpoint():
    """Load finetune checkpoint if it exists. Returns (phase, start_epoch)."""
    if not os.path.exists(FINETUNE_CKPT):
        return "linear_probe", 0
    ckpt = torch.load(FINETUNE_CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Resumed fine-tuning from checkpoint: phase={ckpt['phase']} epoch={ckpt['epoch']}")
    return ckpt["phase"], ckpt["epoch"] + 1


# ── PHASE 1: LINEAR PROBING (encoder frozen) ──────────────────────────────────

current_phase, start_epoch = resume_checkpoint()

total_epochs = (
    FINETUNE_CFG["linear_probe_epochs"]
    + FINETUNE_CFG["full_finetune_epochs"]
)
best_top3 = 0.0

if current_phase == "linear_probe" and start_epoch < FINETUNE_CFG["linear_probe_epochs"]:
    model.freeze_encoder()

    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer   = torch.optim.AdamW(head_params, lr=FINETUNE_CFG["head_lr"],
                                    weight_decay=FINETUNE_CFG["weight_decay"])

    print(f"\n── Phase 1: Linear probing (epochs {start_epoch}–{FINETUNE_CFG['linear_probe_epochs']-1}) ──")

    for epoch in range(start_epoch, FINETUNE_CFG["linear_probe_epochs"]):
        model.train()
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()

        metrics = evaluate(val_loader)
        elapsed = time.time() - t0

        print(
            f"  Epoch {epoch:03d} | "
            f"val loss {metrics['loss']:.4f} | "
            f"exact {metrics['exact_acc']:.3f} | "
            f"top3 {metrics['top3_acc']:.3f} | "
            f"{elapsed:.0f}s"
        )

        save_checkpoint("linear_probe", epoch, metrics, optimizer)

        if metrics["top3_acc"] > best_top3:
            best_top3 = metrics["top3_acc"]

    start_epoch = 0   # reset for phase 2

# ── PHASE 2: FULL FINE-TUNING (encoder unfrozen) ──────────────────────────────

if current_phase in ("linear_probe", "full_finetune"):
    if current_phase == "linear_probe":
        start_epoch = 0
    else:
        start_epoch = start_epoch   # already set from resume

    model.unfreeze_encoder()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = FINETUNE_CFG["full_lr"],
        weight_decay = FINETUNE_CFG["weight_decay"],
        betas        = (0.9, 0.95),
    )

    # Cosine decay over full fine-tuning epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max  = FINETUNE_CFG["full_finetune_epochs"],
        eta_min= 1e-6,
    )

    print(f"\n── Phase 2: Full fine-tuning (epochs {start_epoch}–{FINETUNE_CFG['full_finetune_epochs']-1}) ──")

    for epoch in range(start_epoch, FINETUNE_CFG["full_finetune_epochs"]):
        model.train()
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        metrics = evaluate(val_loader)
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"  Epoch {epoch:03d} | "
            f"val loss {metrics['loss']:.4f} | "
            f"exact {metrics['exact_acc']:.3f} | "
            f"top3 {metrics['top3_acc']:.3f} | "
            f"lr {current_lr:.2e} | "
            f"{elapsed:.0f}s"
        )

        save_checkpoint("full_finetune", epoch, metrics, optimizer)

        if metrics["top3_acc"] > best_top3:
            best_top3 = metrics["top3_acc"]
            # Also save a "best" snapshot alongside the rolling checkpoint
            torch.save(model.state_dict(), FINETUNED_MODEL)
            print(f"    ↑ New best top-3 accuracy — saved to {FINETUNED_MODEL}")

# ── FINAL SAVE ────────────────────────────────────────────────────────────────

# Save final model even if it wasn't the best (useful for debugging)
if not os.path.exists(FINETUNED_MODEL):
    torch.save(model.state_dict(), FINETUNED_MODEL)

print(f"\nFine-tuning complete.")
print(f"  Best top-3 accuracy : {best_top3:.3f}")
print(f"  Final model saved   : {FINETUNED_MODEL}")
print(f"\nNext step: run 11_mae_predict.py")
