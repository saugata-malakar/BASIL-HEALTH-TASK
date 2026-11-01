"""
Baseline models for the EEG Transformer Sprint.

  - EEGNet          : the compact CNN baseline the field benchmarks against.
  - VanillaEEGTransformer : a deliberately naive transformer with per-timestep
                      tokens and NO EEG-specific inductive bias. This is the
                      thing you are meant to BEAT (or at least understand the
                      failure of). Do not "fix" it in place silently — put your
                      idea in main.py so the contrast stays visible.

Both are intentionally small and run in well under a minute on CPU.
"""
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# EEGNet  (Lawhern et al., 2018), compact reference implementation.
# --------------------------------------------------------------------------- #
class EEGNet(nn.Module):
    def __init__(self, n_chan, n_time, n_classes=2,
                 F1=8, D=2, F2=16, kern=64, p=0.25):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        # Depthwise spatial filter across all electrodes.
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_chan, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p),
        )
        # Separable temporal filter.
        self.block3 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
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

    def forward(self, x):            # x: (B, C, T)
        x = x.unsqueeze(1)           # (B, 1, C, T)
        x = self.block3(self.block2(self.block1(x)))
        return self.head(x.flatten(1))


# --------------------------------------------------------------------------- #
# Vanilla transformer: per-timestep tokens, learned positional embedding,
# a CLS token, standard multi-head self-attention. No EEG prior whatsoever.
# This is the honest "attention with no help" baseline.
# --------------------------------------------------------------------------- #
class VanillaEEGTransformer(nn.Module):
    def __init__(self, n_chan, n_time, n_classes=2,
                 d_model=64, n_heads=4, depth=3, p=0.3):
        super().__init__()
        self.proj = nn.Linear(n_chan, d_model)       # each timestep -> a token
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, n_time + 1, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2,
            dropout=p, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):            # x: (B, C, T)
        x = x.transpose(1, 2)        # (B, T, C)
        tok = self.proj(x)           # (B, T, d)
        cls = self.cls.expand(x.size(0), -1, -1)
        z = torch.cat([cls, tok], dim=1) + self.pos
        z = self.encoder(z)
        return self.head(self.norm(z[:, 0]))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
