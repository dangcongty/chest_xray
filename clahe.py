import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalRegionTransform(nn.Module):
    """
    CLAHE-like learnable local photometric transform
    - Per-tile parameter prediction
    - Smooth spatial interpolation of parameters
    - Safe parameterization
    """

    def __init__(self, num_channels=3, tile_size=32):
        super().__init__()
        self.tile_size = tile_size
        self.num_channels = num_channels

        # Predictor outputs per-tile params: brightness, contrast, saturation
        self.param_predictor = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 1)
        )

        # Global base parameters
        self.global_brightness = nn.Parameter(torch.tensor(0.0))
        self.global_contrast   = nn.Parameter(torch.tensor(1.0))
        self.global_saturation = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        """
        x: [B, C, H, W] in range [0, 1]
        """
        B, C, H, W = x.shape
        ts = self.tile_size

        # Pad so H, W divisible by tile_size
        pad_h = (ts - H % ts) % ts
        pad_w = (ts - W % ts) % ts
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        _, _, Hp, Wp = x.shape
        nh, nw = Hp // ts, Wp // ts

        # ------------------------------------------------------------------
        # 1. Predict per-pixel parameter maps
        # ------------------------------------------------------------------
        params = self.param_predictor(x)  # [B, 3, Hp, Wp]

        # ------------------------------------------------------------------
        # 2. Average params per tile
        # ------------------------------------------------------------------
        params = params.view(B, 3, nh, ts, nw, ts)
        params = params.mean(dim=(3, 5))  # [B, 3, nh, nw]

        # ------------------------------------------------------------------
        # 3. Safe parameterization
        # ------------------------------------------------------------------
        b = 0.2 * torch.tanh(params[:, 0:1]) + self.global_brightness
        c = 1.0 + 0.3 * torch.tanh(params[:, 1:2]) + (self.global_contrast - 1.0)
        s = 1.0 + 0.3 * torch.tanh(params[:, 2:3]) + (self.global_saturation - 1.0)

        params = torch.cat([b, c, s], dim=1)  # [B, 3, nh, nw]

        # ------------------------------------------------------------------
        # 4. Smooth interpolation to full resolution (CLAHE-style)
        # ------------------------------------------------------------------
        params = F.interpolate(
            params,
            size=(Hp, Wp),
            mode="bilinear",
            align_corners=False
        )

        brightness = params[:, 0:1]
        contrast   = params[:, 1:2]
        saturation = params[:, 2:3]

        # ------------------------------------------------------------------
        # 5. Apply transformations
        # ------------------------------------------------------------------
        # Contrast (around local mean)
        local_mean = x.mean(dim=(2, 3), keepdim=True)
        x = (x - local_mean) * contrast + local_mean

        # Brightness
        x = x + brightness

        # Saturation (RGB only)
        if C == 3:
            gray = (
                0.299 * x[:, 0:1]
                + 0.587 * x[:, 1:2]
                + 0.114 * x[:, 2:3]
            )
            x = gray + saturation * (x - gray)

        x = torch.clamp(x, 0.0, 1.0)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]

        return x



lrt = LocalRegionTransform()


img = cv2.imread('datasets/process/images/fee7f5490aee6eb6626adb30d2509868.png')
img = torch.from_numpy(img/255).permute((-1, 0, 1)).unsqueeze(0).to(torch.float)

lrt(img)