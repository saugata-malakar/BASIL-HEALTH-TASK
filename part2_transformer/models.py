"""
Baseline models for the EEG Transformer Sprint.

  EEGNet              : compact CNN baseline the field benchmarks against.
  VanillaEEGTransformer : a deliberately naive transformer with per-timestep
                          tokens and NO EEG-specific inductive bias.

Design intent
-------------
VanillaEEGTransformer is kept intentionally unoptimised to provide an honest
"attention with no help" baseline. The contrast between this model and
YourModel (in main.py) is the scientific point of Part 2 — do NOT quietly
improve this model. Put your idea in main.py so the contrast remains visible.

Both models are intentionally small to run on CPU in under one minute.
"""
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# EEGNet  (Lawhern et al., 2018)
# Compact reference implementation used as the conv-prior baseline.
# Achieves strong performance via three EEG-specific architectural choices:
#   1. Temporal convolution: learns band-specific filters (analogous to FBCSP).
#   2. Depthwise spatial convolution: learns electrode weights per spectral band
#      (a data-driven analogue of Common Spatial Patterns / virtual reference).
#   3. AvgPool after ELU: computes a phase-invariant band-power envelope BEFORE
#      combining spatial channels — critical for non-phase-locked ERD signals.
# --------------------------------------------------------------------------- #
class EEGNet(nn.Module):
    def __init__(self, n_chan, n_time, n_classes=2,
                 F1=8, D=2, F2=16, kern=64, p=0.25):
        super().__init__()
        # Block 1: temporal filter per channel.
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        # Block 2: depthwise spatial filter — mixes all electrodes for each
        # temporal feature map independently (groups=F1 keeps them separate).
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_chan, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),   # temporal pooling -> phase-invariant energy
            nn.Dropout(p),
        )
        # Block 3: separable temporal filter (depthwise + pointwise).
        self.block3 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8),
                      groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(p),
        )
        with torch.no_grad():
            d = self.block3(self.block2(self.block1(
                torch.zeros(1, 1, n_chan, n_time)))).flatten(1).shape[1]
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):       # x: (B, C, T)
        x = x.unsqueeze(1)      # (B, 1, C, T)
        return self.head(self.block3(self.block2(self.block1(x))).flatten(1))


# --------------------------------------------------------------------------- #
# VanillaEEGTransformer — per-timestep tokens, no EEG prior.
#
# Intentional weaknesses (by design — do not fix here):
#   1. Each token = linear projection of all C channels at one timestep.
#      Electrode identity is immediately discarded; the model must re-learn
#      scalp topology from the data.
#   2. Sequence length = T = 481 timesteps -> (T+1)^2 = 232,324 attention
#      entries per head. With ~360 training trials this massively overfits.
#   3. Positional embeddings are random-initialised — no spatial prior.
#   4. Raw voltage tokens are phase-sensitive: non-phase-locked ERD at 10 Hz
#      shifts by 50 ms = pi radians between trials, flipping token sign and
#      corrupting attention dot-products (The Phase-Cancellation Theorem).
#
# These weaknesses are features, not bugs: they define the honest baseline
# that YourModel's inductive biases are intended to address.
# --------------------------------------------------------------------------- #
class VanillaEEGTransformer(nn.Module):
    def __init__(self, n_chan, n_time, n_classes=2,
                 d_model=64, n_heads=4, depth=3, p=0.3):
        super().__init__()
        self.proj = nn.Linear(n_chan, d_model)   # all channels at t -> one token
        self.cls  = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos  = nn.Parameter(torch.zeros(1, n_time + 1, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2,
            dropout=p, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):        # x: (B, C, T)
        x   = x.transpose(1, 2) # (B, T, C)
        tok = self.proj(x)       # (B, T, d_model) — electrode identity lost
        cls = self.cls.expand(x.size(0), -1, -1)
        z   = torch.cat([cls, tok], dim=1) + self.pos
        z   = self.encoder(z)
        return self.head(self.norm(z[:, 0]))


def count_params(m):
    """Return number of trainable parameters."""
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
