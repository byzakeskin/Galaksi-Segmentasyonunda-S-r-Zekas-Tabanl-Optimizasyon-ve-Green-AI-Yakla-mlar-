import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers += [
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, dropout)
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                         diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 n_classes:    int = 1,
                 base_filters:  int = 64,
                 dropout:       float = 0.1):
        super().__init__()
        f = base_filters
        self.inc   = DoubleConv(in_channels, f,        dropout)
        self.down1 = Down(f,      f * 2,  dropout)
        self.down2 = Down(f * 2,  f * 4,  dropout)
        self.down3 = Down(f * 4,  f * 8,  dropout)
        self.down4 = Down(f * 8,  f * 16, dropout)   

        self.up1   = Up(f * 16, f * 8,  dropout)
        self.up2   = Up(f * 8,  f * 4,  dropout)
        self.up3   = Up(f * 4,  f * 2,  dropout)
        self.up4   = Up(f * 2,  f,      dropout)

        self.outc  = nn.Conv2d(f, n_classes, kernel_size=1)

        self.exit_heads = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(f * 16, n_classes),
            ),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(f * 8, n_classes),
            ),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(f * 4, n_classes),
            ),
        ])

    def forward(self, x, early_exit_thresholds=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  

        if early_exit_thresholds is not None and not self.training:
            bottleneck_conf = torch.sigmoid(self.exit_heads[0](x5)).max().item()
            if bottleneck_conf >= early_exit_thresholds[0]:
                return {"logits": self.exit_heads[0](x5).unsqueeze(-1).unsqueeze(-1),
                        "exit_at": 0, "confidence": bottleneck_conf}

        x = self.up1(x5, x4)

        if early_exit_thresholds is not None and not self.training:
            conf = torch.sigmoid(self.exit_heads[1](x)).max().item()
            if conf >= early_exit_thresholds[1]:
                return {"logits": self.exit_heads[1](x).unsqueeze(-1).unsqueeze(-1),
                        "exit_at": 1, "confidence": conf}

        x = self.up2(x,  x3)

        if early_exit_thresholds is not None and not self.training:
            conf = torch.sigmoid(self.exit_heads[2](x)).max().item()
            if conf >= early_exit_thresholds[2]:
                return {"logits": self.exit_heads[2](x).unsqueeze(-1).unsqueeze(-1),
                        "exit_at": 2, "confidence": conf}

        x = self.up3(x,  x2)
        x = self.up4(x,  x1)

        return {"logits": self.outc(x), "exit_at": "full", "confidence": 1.0}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def apply_pruning(self, pruning_rates: list):
        layers  = [self.inc, self.down1, self.down2, self.down3, self.down4]
        pruned  = 0
        total   = 0

        for layer, rate in zip(layers, pruning_rates):
            for module in layer.modules():
                if isinstance(module, nn.Conv2d):
                    with torch.no_grad():
                        w    = module.weight.data
                        k    = int(w.numel() * rate)
                        flat = w.abs().view(-1)
                        thresh = flat.kthvalue(max(1, k)).values
                        mask   = (w.abs() >= thresh).float()
                        module.weight.data *= mask
                        pruned += int((mask == 0).sum())
                        total  += w.numel()

        sparsity = pruned / total * 100 if total > 0 else 0
        print(f"[Pruning] {pruned:,}/{total:,} ağırlık sıfırlandı "
              f"(Seyreklik: %{sparsity:.1f})")
        return sparsity