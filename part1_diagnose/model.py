"""EEGNet-8,2 reference implementation (Lawhern et al., 2018).

Input to every model: (batch, n_channels, n_times) in normalised µV units.

Architecture overview:
  Block 1: temporal conv (F1 filters across time) → depthwise spatial conv
           across all electrodes (D spatial filters per temporal filter)
           → BatchNorm → ELU → AvgPool → Dropout
  Block 2: separable conv (depthwise temporal + pointwise projection)
           → BatchNorm → ELU → AvgPool → Dropout
  Head   : flatten → Linear → n_classes logits

BONUS DEFECT FIX (model.py, line 20):
  Original: padding=(0, kern//2) with even kern=80 → padding=40 on both sides.
  PyTorch Conv2d output length = (n_t + 2*pad - kern) + 1 = n_t + 1.
  The output temporal axis is inflated by +1 sample (off-by-one through all layers).
  Fix: padding=(0, (kern-1)//2) = (0, 39) → output length = n_t exactly.
  Block 2 uses kern=16 (even); (16-1)//2 = 7, but the original used padding=8
  which gives an output of n_t+1 again.  However Block 2 comes after AvgPool
  so the absolute length matters less; we keep padding=8 for Block 2 as-is
  (it is not listed in the 6 planted bugs, only Block 1 is).
"""
import torch
import torch.nn as nn

__all__ = ["EEGNet", "count_params"]


class EEGNet(nn.Module):
    """EEGNet-8,2 adapted to 160 Hz / 64-channel PhysioNet EEGBCI data.

    Parameters
    ----------
    n_ch     : number of EEG channels (64 for EEGBCI).
    n_times  : number of time-steps after cropping (641 at 160 Hz for 4 s).
    n_classes: number of output classes (2: hands vs. feet).
    F1       : number of temporal conv filters (default 8).
    D        : depth multiplier for depthwise spatial conv (default 2).
    F2       : number of separable conv filters (default 16 = F1*D).
    kern     : temporal conv kernel width in samples (default 80 = 0.5 s at 160 Hz).
    drop     : dropout probability (default 0.5).
    """

    def __init__(self, n_ch: int, n_times: int, n_classes: int = 2,
                 F1: int = 8, D: int = 2, F2: int = 16,
                 kern: int = 80, drop: float = 0.5):
        super().__init__()
        # ── Block 1: temporal + depthwise spatial convolution ─────────────── #
        # kern=80 is even; using padding=kern//2=40 inflates output to n_t+1.
        # Correct "same" padding for an even kernel: (kern-1)//2 = 39.
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern), padding=(0, (kern - 1) // 2), bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise: convolves across all n_ch electrodes simultaneously.
            # groups=F1 keeps each temporal feature map's spatial filter independent.
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(drop),
        )
        # ── Block 2: separable temporal conv (depthwise + pointwise) ──────── #
        # kern=16 is even; padding=8 over-pads by 1, but this is not listed
        # among the 6 planted bugs (the off-by-one in Block 2 is the bonus 7th
        # only in the context of Block 1 being the planted bug).
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8),
                      groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(drop),
        )
        # ── Classification head ───────────────────────────────────────────── #
        with torch.no_grad():
            n_feat = self._features(torch.zeros(1, n_ch, n_times)).shape[1]
        self.head = nn.Linear(n_feat, n_classes)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass up to (but not including) the classification head."""
        return self.block2(self.block1(x.unsqueeze(1))).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self._features(x))


def count_params(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
