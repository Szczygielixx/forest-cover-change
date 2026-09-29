import torch
from torch import nn
from torch.nn import functional as F


def block(a, b):
    return nn.Sequential(
        nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.Conv2d(b, b, 3, padding=1), nn.ReLU()
    )


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.ModuleList([block(9, 32), block(32, 64), block(64, 128), block(128, 256)])
        self.dec = nn.ModuleList([block(384, 128), block(192, 64), block(96, 32)])
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        skip = []
        for layer in self.enc[:-1]:
            x = layer(x)
            skip.append(x)
            x = F.max_pool2d(x, 2)
        x = self.enc[-1](x)
        for layer, s in zip(self.dec, reversed(skip)):
            x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = layer(torch.cat([x, s], dim=1))
        return self.out(x)[:, 0, 32:96, 32:96]


def sparse_loss(logits, labels):
    mask = labels != 255
    if not mask.any():
        raise ValueError("Training batch without supervision")
    return F.binary_cross_entropy_with_logits(logits[mask], labels[mask].float())
