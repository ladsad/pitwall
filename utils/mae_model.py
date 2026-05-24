"""
utils/mae_model.py
──────────────────
1D PatchTST-style Masked Autoencoder for F1 lap telemetry.

All shapes are documented as (Batch, Channels, Length) or (B, T, D) for
transformer sequence dims so it's easy to follow without prior ViT experience.

Public API
──────────
  F1MAE                 — full pre-training model
    .forward_loss(x)    → scalar MSE loss (masked patches only)
    .forward_features(x)→ (B, n_patches, d_model) encoder output, no masking

  F1PositionHead        — fine-tuning wrapper
    .forward(x)         → (B, n_classes) logits for race position prediction
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

N_CHANNELS   = 6      # Speed, Throttle, Brake, RPM, nGear, DRS
SEQ_LEN      = 1024   # fixed after resampling in 08_tel_preprocess.py
N_PATCHES    = SEQ_LEN // 16   # = 64  (patch_stride=16)
N_CLASSES    = 20              # race positions 1–20


# ── POSITIONAL EMBEDDING ──────────────────────────────────────────────────────

class LearnedPositionalEmbedding(nn.Module):
    """Learnable positional embedding — one vector per patch position."""

    def __init__(self, n_patches: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_patches, d_model)
        return x + self.pe


# ── PATCH EMBEDDING ───────────────────────────────────────────────────────────

class PatchEmbedding1D(nn.Module):
    """
    Tokenise a (B, C, L) telemetry tensor into (B, n_patches, d_model) tokens.

    Strategy: project each channel independently with a Conv1d (stride = patch_stride),
    then sum the channel embeddings. This is simpler than concatenating and avoids
    the d_model having to grow linearly with C.

    Shape walkthrough:
      Input:  (B, 6, 1024)
      Per-channel Conv1d(1, d_model, kernel=patch_stride, stride=patch_stride):
        → (B, d_model, 64) per channel
      Stack → (B, 6, d_model, 64)
      Sum over channel dim → (B, d_model, 64)
      Transpose → (B, 64, d_model)   ← standard (B, T, D) for transformer
    """

    def __init__(
        self,
        n_channels:   int = N_CHANNELS,
        d_model:      int = 384,
        patch_stride: int = 16,
    ):
        super().__init__()
        self.patch_stride = patch_stride
        self.n_patches    = SEQ_LEN // patch_stride

        # One Conv1d per input channel — shares the same d_model output dim
        self.channel_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=1,
                out_channels=d_model,
                kernel_size=patch_stride,
                stride=patch_stride,
                bias=False,
            )
            for _ in range(n_channels)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        B, C, L = x.shape

        patch_embeds = []
        for i, conv in enumerate(self.channel_convs):
            # (B, 1, L) → (B, d_model, n_patches)
            ch = x[:, i:i+1, :]
            patch_embeds.append(conv(ch))

        # Stack → (B, C, d_model, n_patches) then sum over C
        stacked = torch.stack(patch_embeds, dim=1)       # (B, C, d, n_p)
        summed  = stacked.sum(dim=1)                     # (B, d, n_p)
        out     = summed.permute(0, 2, 1)                # (B, n_p, d)
        return self.norm(out)


# ── TRANSFORMER BLOCK ─────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block (used for both encoder and decoder)."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


# ── MAE ENCODER ──────────────────────────────────────────────────────────────

class MAEEncoder(nn.Module):
    """
    ViT encoder — operates only on the VISIBLE (unmasked) patches during pre-training.
    During fine-tuning / inference, all patches are passed (no masking).
    """

    def __init__(
        self,
        d_model:       int   = 384,
        n_heads:       int   = 6,
        n_layers:      int   = 6,
        patch_stride:  int   = 16,
        n_channels:    int   = N_CHANNELS,
        dropout:       float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding1D(n_channels, d_model, patch_stride)
        self.pos_embed   = LearnedPositionalEmbedding(N_PATCHES, d_model)
        self.blocks      = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x:            torch.Tensor,
        mask_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        x:            (B, C, L)
        mask_indices: (B, n_masked) — indices of patches to DROP for MAE pre-training.
                      Pass None at inference / fine-tuning time.

        Returns:
          tokens:      (B, n_visible, d_model) if masking; (B, n_patches, d_model) otherwise
          keep_indices:(B, n_visible) int64 indices of kept patches (None if no masking)
        """
        tokens = self.patch_embed(x)    # (B, n_patches, d_model)
        tokens = self.pos_embed(tokens) # add positional info before masking

        keep_indices = None
        if mask_indices is not None:
            B, n_patches, d = tokens.shape
            # Build keep_indices as the complement of mask_indices
            all_indices  = torch.arange(n_patches, device=x.device).unsqueeze(0).expand(B, -1)
            # mask_indices shape: (B, n_masked); keep: (B, n_visible)
            keep_indices = _complement_indices(all_indices, mask_indices)
            tokens = _gather_patches(tokens, keep_indices)

        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)
        return tokens, keep_indices


# ── MAE DECODER ──────────────────────────────────────────────────────────────

class MAEDecoder(nn.Module):
    """
    Lightweight 2-layer decoder.
    Receives visible encoder tokens + learnable mask tokens,
    restores full sequence order, and reconstructs patch values.
    """

    def __init__(
        self,
        encoder_d_model: int   = 384,
        decoder_d_model: int   = 192,
        n_heads:         int   = 4,
        n_layers:        int   = 2,
        patch_stride:    int   = 16,
        n_channels:      int   = N_CHANNELS,
    ):
        super().__init__()
        patch_dim = patch_stride * n_channels   # values to reconstruct per patch

        # Project encoder output to (smaller) decoder dimension
        self.encoder_proj = nn.Linear(encoder_d_model, decoder_d_model)
        # Learnable mask token — one shared vector added in place of every masked patch
        self.mask_token   = nn.Parameter(torch.zeros(1, 1, decoder_d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.pos_embed = LearnedPositionalEmbedding(N_PATCHES, decoder_d_model)
        self.blocks    = nn.Sequential(*[
            TransformerBlock(decoder_d_model, n_heads)
            for _ in range(n_layers)
        ])
        self.norm    = nn.LayerNorm(decoder_d_model)
        self.pred    = nn.Linear(decoder_d_model, patch_dim)   # reconstruction head

    def forward(
        self,
        visible_tokens: torch.Tensor,
        keep_indices:   torch.Tensor,
        mask_indices:   torch.Tensor,
    ) -> torch.Tensor:
        """
        visible_tokens: (B, n_visible, encoder_d_model)
        keep_indices:   (B, n_visible)
        mask_indices:   (B, n_masked)

        Returns:
          pred: (B, n_masked, patch_dim) — reconstructed patch values for masked positions only
        """
        B, n_visible, _ = visible_tokens.shape
        n_patches = N_PATCHES
        n_masked  = mask_indices.shape[1]

        # Project encoder tokens to decoder dimension
        vis = self.encoder_proj(visible_tokens)   # (B, n_visible, decoder_d)

        # Fill a full-length sequence: visible tokens at keep positions, mask tokens elsewhere
        full = self.mask_token.expand(B, n_patches, -1).clone()   # (B, n_patches, decoder_d)
        full = _scatter_patches(full, keep_indices, vis)

        # Add positional embeddings (full sequence) so decoder knows each patch's position
        full = self.pos_embed(full)
        full = self.blocks(full)
        full = self.norm(full)
        full = self.pred(full)   # (B, n_patches, patch_dim)

        # Return only the masked positions — MSE loss computed there
        return _gather_patches(full, mask_indices)   # (B, n_masked, patch_dim)


# ── FULL MAE MODEL ────────────────────────────────────────────────────────────

class F1MAE(nn.Module):
    """
    Full Masked Autoencoder model for F1 telemetry.

    Usage
    ─────
    Pre-training:
        loss = model.forward_loss(x)
        loss.backward()

    Feature extraction (fine-tuning input):
        features = model.forward_features(x)   # (B, n_patches, d_model), no masking
    """

    def __init__(
        self,
        d_model:        int   = 384,
        n_heads:        int   = 6,
        encoder_layers: int   = 6,
        decoder_d_model:int   = 192,
        decoder_n_heads:int   = 4,
        decoder_layers: int   = 2,
        patch_stride:   int   = 16,
        mask_ratio:     float = 0.75,
        n_channels:     int   = N_CHANNELS,
    ):
        super().__init__()
        self.mask_ratio   = mask_ratio
        self.patch_stride = patch_stride
        self.n_patches    = SEQ_LEN // patch_stride

        self.encoder = MAEEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=encoder_layers,
            patch_stride=patch_stride,
            n_channels=n_channels,
        )
        self.decoder = MAEDecoder(
            encoder_d_model=d_model,
            decoder_d_model=decoder_d_model,
            n_heads=decoder_n_heads,
            n_layers=decoder_layers,
            patch_stride=patch_stride,
            n_channels=n_channels,
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _random_mask(self, B: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly sample mask_ratio fraction of patches to mask.

        Returns:
          keep_indices: (B, n_visible)
          mask_indices: (B, n_masked)
        """
        n_masked  = int(self.n_patches * self.mask_ratio)
        n_visible = self.n_patches - n_masked

        noise = torch.rand(B, self.n_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)         # ascending — first n_visible = keep

        keep_indices = ids_shuffle[:, :n_visible]          # (B, n_visible)
        mask_indices = ids_shuffle[:, n_visible:]          # (B, n_masked)
        return keep_indices, mask_indices

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert raw input (B, C, L) into patch targets (B, n_patches, patch_stride*C).
        The decoder predicts these raw patch values.
        """
        B, C, L = x.shape
        p = self.patch_stride
        x = x.reshape(B, C, L // p, p)         # (B, C, n_patches, p)
        x = x.permute(0, 2, 1, 3)              # (B, n_patches, C, p)
        x = x.reshape(B, self.n_patches, C * p) # (B, n_patches, patch_dim)
        return x

    def forward_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute MAE reconstruction loss.
        Loss is MSE on the masked patches only (following MAE paper convention).
        x: (B, C, L) — normalised telemetry tensor from Silver
        """
        B = x.shape[0]
        keep_indices, mask_indices = self._random_mask(B, x.device)

        visible_tokens, _ = self.encoder(x, mask_indices=mask_indices)
        pred = self.decoder(visible_tokens, keep_indices, mask_indices)
        # (B, n_masked, patch_dim)

        target = self._patchify(x)                       # (B, n_patches, patch_dim)
        target_masked = _gather_patches(target, mask_indices)  # (B, n_masked, patch_dim)

        loss = F.mse_loss(pred, target_masked)
        return loss

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode x with NO masking. Used at fine-tuning / inference time.
        Returns (B, n_patches, d_model) — typically mean-pooled before the task head.
        """
        tokens, _ = self.encoder(x, mask_indices=None)
        return tokens


# ── FINE-TUNING HEAD ─────────────────────────────────────────────────────────

class F1PositionHead(nn.Module):
    """
    Race-position classifier built on top of a pre-trained F1MAE encoder.

    Fine-tuning protocol (two-phase, set in 10_mae_finetune.py):
      Phase 1 (epochs 0–9):  freeze encoder → linear probing only
      Phase 2 (epochs 10+):  unfreeze encoder → full fine-tuning

    Call .freeze_encoder() / .unfreeze_encoder() to switch phases.
    """

    def __init__(
        self,
        encoder:   MAEEncoder,
        d_model:   int = 384,
        n_classes: int = N_CLASSES,
        dropout:   float = 0.2,
    ):
        super().__init__()
        self.encoder = encoder
        self.head    = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )
        # Initialise head weights (encoder is already pre-trained)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        print("Encoder frozen — training head only (linear probing phase)")

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
        print("Encoder unfrozen — full fine-tuning")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, L)
        Returns logits (B, n_classes) — pass through CrossEntropyLoss for position prediction.
        """
        features = self.encoder(x, mask_indices=None)[0]  # (B, n_patches, d_model)
        pooled   = features.mean(dim=1)                   # (B, d_model) — mean over patches
        return self.head(pooled)


# ── INDEX MANIPULATION HELPERS ────────────────────────────────────────────────
# These replace the more complex scatter/gather one-liners that trip up readers
# unfamiliar with advanced PyTorch indexing.

def _gather_patches(
    tokens:  torch.Tensor,   # (B, n_patches, D)
    indices: torch.Tensor,   # (B, k) — integer indices to select
) -> torch.Tensor:           # (B, k, D)
    """Select k patch positions from a (B, n_patches, D) tensor."""
    D   = tokens.shape[-1]
    idx = indices.unsqueeze(-1).expand(-1, -1, D)   # (B, k, D)
    return torch.gather(tokens, dim=1, index=idx)


def _scatter_patches(
    full:    torch.Tensor,   # (B, n_patches, D) — destination (modified in-place clone)
    indices: torch.Tensor,   # (B, k) — positions to write into
    src:     torch.Tensor,   # (B, k, D) — values to write
) -> torch.Tensor:
    """Write src values into full at the given patch indices."""
    D   = full.shape[-1]
    idx = indices.unsqueeze(-1).expand(-1, -1, D)
    return full.scatter(dim=1, index=idx, src=src)


def _complement_indices(
    all_indices:  torch.Tensor,   # (B, n_patches)
    mask_indices: torch.Tensor,   # (B, n_masked)
) -> torch.Tensor:               # (B, n_visible)
    """Return indices not in mask_indices."""
    B, n_patches = all_indices.shape
    n_masked  = mask_indices.shape[1]
    n_visible = n_patches - n_masked

    # Build a boolean keep mask (True = keep), then select
    keep_mask = torch.ones(B, n_patches, dtype=torch.bool, device=all_indices.device)
    keep_mask.scatter_(1, mask_indices, False)
    return all_indices[keep_mask].view(B, n_visible)
