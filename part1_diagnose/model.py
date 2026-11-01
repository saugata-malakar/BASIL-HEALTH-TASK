"""Models. Input to every model: (batch, n_channels, n_times) in normalised units."""
import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """EEGNet-8,2 (Lawhern et al., 2018), adapted to 160 Hz.

    Block 1: temporal conv (F1 filters) -> depthwise conv across all electrodes
             (D spatial filters per temporal filter) -> BN -> ELU -> pool -> dropout
    Block 2: separable conv (depthwise temporal + pointwise) -> BN -> ELU -> pool -> dropout
    """

    def __init__(self, n_ch, n_times, n_classes=2, F1=8, D=2, F2=16,
                 kern=80, drop=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            # kern=80 is even; padding=kern//2=40 would over-pad by 1 (output = n_t+1).
            # Use padding=(kern-1)//2 = 39 so output length equals input length.
            nn.Conv2d(1, F1, (1, kern), padding=(0, (kern - 1) // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(drop))
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(drop))
        with torch.no_grad():
            n_feat = self._features(torch.zeros(1, n_ch, n_times)).shape[1]
        self.head = nn.Linear(n_feat, n_classes)

    def _features(self, x):
        x = self.block2(self.block1(x.unsqueeze(1)))
        return x.flatten(1)

    def forward(self, x):
        return self.head(self._features(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
