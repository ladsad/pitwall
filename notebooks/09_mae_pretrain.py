import os
import csv
import math
import pathlib
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

try:
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = pathlib.Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.mae_model import F1MAE
from utils.tel_dataset import F1TelemetryDataset
from config import TELEMETRY_CLEAN_PATH, BASE_PATH, TEL_SEASONS, ENCODER_HPARAMS

# ── PATHS ─────────────────────────────────────────────────────────────────────

SILVER_PATH = str(TELEMETRY_CLEAN_PATH / "silver")
CKPT_PATH   = str(BASE_PATH / "models" / "mae_checkpoint.pt")
LOG_PATH    = str(BASE_PATH / "models" / "mae_train_log.csv")

# ── HYPERPARAMETERS ───────────────────────────────────────────────────────────
# Loaded from config.py — single source of truth shared with 10 and 11.
# ViT-Small — fits on CE K80 (12 GB) with batch_size=128.
# If you change ENCODER_HPARAMS, delete mae_checkpoint.pt and retrain from scratch.

TRAIN_CFG = dict(
    total_epochs   = 200,
    warmup_epochs  = 10,
    base_lr        = 1.5e-4,    # cosine decay from this peak
    min_lr         = 1e-6,
    weight_decay   = 0.05,
    batch_size     = 128,       # reduce to 64 if OOM on K80
    val_fraction   = 0.05,
    num_workers    = 2,
    pin_memory     = True,
)

# ── DEVICE ────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── DATASET ───────────────────────────────────────────────────────────────────

print(f"\nLoading Silver telemetry from: {SILVER_PATH}")
dataset = F1TelemetryDataset(
    silver_path   = SILVER_PATH,
    seasons       = TEL_SEASONS,
    labelled_only = False,   # pre-training uses ALL laps — no label needed
)

n_total = len(dataset)
n_val   = max(1, int(n_total * TRAIN_CFG["val_fraction"]))
n_train = n_total - n_val

train_ds, val_ds = random_split(
    dataset,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(42),
)

train_loader = DataLoader(
    train_ds,
    batch_size  = TRAIN_CFG["batch_size"],
    shuffle     = True,
    num_workers = TRAIN_CFG["num_workers"],
    pin_memory  = TRAIN_CFG["pin_memory"],
    drop_last   = True,   # avoids batch-norm edge cases on the last small batch
)
val_loader = DataLoader(
    val_ds,
    batch_size  = TRAIN_CFG["batch_size"] * 2,
    shuffle     = False,
    num_workers = TRAIN_CFG["num_workers"],
    pin_memory  = TRAIN_CFG["pin_memory"],
)

print(f"Dataset: {n_train:,} train | {n_val:,} val laps")
print(f"Steps per epoch: {len(train_loader):,}")

# ── MODEL ─────────────────────────────────────────────────────────────────────

model = F1MAE(**ENCODER_HPARAMS).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters: {n_params:,}")

# ── OPTIMISER & SCHEDULE ─────────────────────────────────────────────────────

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr           = TRAIN_CFG["base_lr"],
    weight_decay = TRAIN_CFG["weight_decay"],
    betas        = (0.9, 0.95),   # MAE paper default
)


def cosine_lr(epoch: int) -> float:
    """
    Linear warmup for `warmup_epochs`, then cosine decay to `min_lr`.
    Returns the LR multiplier (applied via scheduler.step() each epoch).
    """
    total   = TRAIN_CFG["total_epochs"]
    warmup  = TRAIN_CFG["warmup_epochs"]
    base_lr = TRAIN_CFG["base_lr"]
    min_lr  = TRAIN_CFG["min_lr"]

    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup)
    cosine   = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr / base_lr + (1 - min_lr / base_lr) * cosine


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_lr)

# ── CHECKPOINT RESUME ─────────────────────────────────────────────────────────
# CE clusters terminate after 2 hours. We checkpoint after every epoch so
# restarts continue from where training left off.

start_epoch  = 0
best_val_loss = float("inf")

if os.path.exists(CKPT_PATH):
    print(f"\nResuming from checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch   = ckpt["epoch"] + 1
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    print(f"  Resumed at epoch {start_epoch} | best val loss so far: {best_val_loss:.6f}")
else:
    print("\nNo checkpoint found — starting from scratch")

# ── LOGGING ───────────────────────────────────────────────────────────────────

log_file_exists = os.path.exists(LOG_PATH)
log_file = open(LOG_PATH, "a", newline="")
log_writer = csv.writer(log_file)
if not log_file_exists:
    log_writer.writerow(["epoch", "train_loss", "val_loss", "lr", "elapsed_s"])
    log_file.flush()

# ── TRAINING FUNCTIONS ────────────────────────────────────────────────────────

def train_one_epoch(epoch: int) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for x, _ in train_loader:
        # x: (B, 6, 1024)   _: labels — unused during pre-training
        x = x.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = model.forward_loss(x)
        loss.backward()
        # Gradient clipping stabilises early training (MAE recommendation: max_norm=1.0)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches


@torch.no_grad()
def validate() -> float:
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    for x, _ in val_loader:
        x = x.to(device, non_blocking=True)
        loss = model.forward_loss(x)
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ── MAIN TRAINING LOOP ────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"Pre-training F1MAE: epochs {start_epoch}–{TRAIN_CFG['total_epochs'] - 1}")
print(f"Checkpoint path: {CKPT_PATH}")
print(f"{'='*60}\n")

t_start = time.time()

for epoch in range(start_epoch, TRAIN_CFG["total_epochs"]):
    t_epoch_start = time.time()

    train_loss = train_one_epoch(epoch)
    val_loss   = validate()
    scheduler.step()

    current_lr  = scheduler.get_last_lr()[0]
    elapsed     = time.time() - t_epoch_start
    total_elapsed = time.time() - t_start

    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss = val_loss

    print(
        f"Epoch {epoch:03d}/{TRAIN_CFG['total_epochs']-1} | "
        f"train {train_loss:.6f} | "
        f"val {val_loss:.6f}{'*' if is_best else ' '} | "
        f"lr {current_lr:.2e} | "
        f"{elapsed:.0f}s/epoch | "
        f"total {total_elapsed/3600:.2f}h"
    )

    # ── Checkpoint (every epoch — CE can terminate at any time) ───────────────
    torch.save(
        {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "train_loss":      train_loss,
            "val_loss":        val_loss,
            "best_val_loss":   best_val_loss,
            "hparams":         ENCODER_HPARAMS,
        },
        CKPT_PATH,
    )

    # ── Log to CSV ────────────────────────────────────────────────────────────
    log_writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                         f"{current_lr:.2e}", f"{elapsed:.1f}"])
    log_file.flush()

log_file.close()

print(f"\nPre-training complete.")
print(f"  Best val loss : {best_val_loss:.6f}")
print(f"  Checkpoint    : {CKPT_PATH}")
print(f"  Training log  : {LOG_PATH}")
print(f"\nNext step: run 10_mae_finetune.py")
