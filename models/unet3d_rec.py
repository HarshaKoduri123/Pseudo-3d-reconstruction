import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class SliceConvBlock(nn.Module):

    def __init__(self, ic, oc):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(ic, oc, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.InstanceNorm3d(oc, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(oc, oc, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.InstanceNorm3d(oc, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class VolumeConvBlock(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(ic, oc, kernel_size=(2, 3, 3), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(oc, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(oc, oc, kernel_size=(2, 3, 3), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(oc, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        out = self.block(x)
        return out[:, :, :x.shape[2], :, :]


class DepthwiseSpatialRefine(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(c, c, kernel_size=(1, 3, 3), padding=(0, 1, 1), groups=c, bias=False),
            nn.Conv3d(c, c, kernel_size=1, bias=False),
            nn.InstanceNorm3d(c, affine=True),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ZMambaBlock(nn.Module):

    def __init__(self, channels, d_state=16, d_conv=4, expand=2):
        super().__init__()

        self.pre_spatial = DepthwiseSpatialRefine(channels)
        self.norm = nn.LayerNorm(channels)
        self.mamba = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.proj = nn.Linear(channels, channels)
        self.post_norm = nn.InstanceNorm3d(channels, affine=True)

    def forward(self, x):
        residual = x
        x = x + self.pre_spatial(x)

        B, C, Z, H, W = x.shape
        seq = x.permute(0, 3, 4, 2, 1).contiguous().view(B * H * W, Z, C)

        seq = self.norm(seq)
        seq = self.mamba(seq)
        seq = self.proj(seq)
        x = seq.view(B, H, W, Z, C).permute(0, 4, 3, 1, 2).contiguous()
        x = self.post_norm(x)

        return residual + x


class HybridDecoderBlock(nn.Module):

    def __init__(self, ic, oc, use_mamba=True, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.local = VolumeConvBlock(ic, oc)
        self.use_mamba = use_mamba

        if use_mamba:
            self.zmix = ZMambaBlock(
                channels=oc,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.zmix = nn.Identity()

    def forward(self, x):
        x = self.local(x)
        x = self.zmix(x)
        return x

class AsymUNet3D(nn.Module):

    def __init__(self, f=32):
        super().__init__()

        self.enc1 = SliceConvBlock(2,   f)
        self.enc2 = SliceConvBlock(f,   f * 2)
        self.enc3 = SliceConvBlock(f*2, f * 4)
        self.enc4 = SliceConvBlock(f*4, f * 8)

        self.bottleneck = VolumeConvBlock(f * 8, f * 16)

        self.dec4 = VolumeConvBlock(f * 16 + f * 8, f * 8)
        self.dec3 = VolumeConvBlock(f * 8  + f * 4, f * 4)
        self.dec2 = VolumeConvBlock(f * 4  + f * 2, f * 2)
        self.dec1 = VolumeConvBlock(f * 2  + f,     f)

        self.out_conv = nn.Sequential(
            nn.Conv3d(f, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm3d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _upsample_cat(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, sparse, obs_mask=None):
        if obs_mask is None:
            obs_mask = (sparse.mean(dim=[1, 3, 4], keepdim=True) > 0).float()
            obs_mask = obs_mask.expand_as(sparse)

        x = torch.cat([sparse, obs_mask], dim=1)

        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        x = self.bottleneck(self.pool(s4))

        x = self.dec4(self._upsample_cat(x, s4))
        x = self.dec3(self._upsample_cat(x, s3))
        x = self.dec2(self._upsample_cat(x, s2))
        x = self.dec1(self._upsample_cat(x, s1))

        return self.out_conv(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ================================================================
# Z-MAMBA MODEL
# ================================================================

class AsymZMambaUNet3D(nn.Module):

    def __init__(self, f=32, d_state=16, d_conv=4, expand=2, mamba_stages=("bottleneck", "dec4", "dec3")):
        super().__init__()

        self.enc1 = SliceConvBlock(2,   f)
        self.enc2 = SliceConvBlock(f,   f * 2)
        self.enc3 = SliceConvBlock(f*2, f * 4)
        self.enc4 = SliceConvBlock(f*4, f * 8)

        self.bottleneck = HybridDecoderBlock(
            f * 8, f * 16,
            use_mamba=("bottleneck" in mamba_stages),
            d_state=d_state, d_conv=d_conv, expand=expand
        )

        self.dec4 = HybridDecoderBlock(
            f * 16 + f * 8, f * 8,
            use_mamba=("dec4" in mamba_stages),
            d_state=d_state, d_conv=d_conv, expand=expand
        )

        self.dec3 = HybridDecoderBlock(
            f * 8 + f * 4, f * 4,
            use_mamba=("dec3" in mamba_stages),
            d_state=d_state, d_conv=d_conv, expand=expand
        )

        # keep final high-res stages local to save memory
        self.dec2 = HybridDecoderBlock(
            f * 4 + f * 2, f * 2,
            use_mamba=("dec2" in mamba_stages),
            d_state=d_state, d_conv=d_conv, expand=expand
        )

        self.dec1 = HybridDecoderBlock(
            f * 2 + f, f,
            use_mamba=("dec1" in mamba_stages),
            d_state=d_state, d_conv=d_conv, expand=expand
        )

        self.out_conv = nn.Sequential(
            nn.Conv3d(f, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm3d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _upsample_cat(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, sparse, obs_mask=None):
        if obs_mask is None:
            obs_mask = (sparse.mean(dim=[1, 3, 4], keepdim=True) > 0).float()
            obs_mask = obs_mask.expand_as(sparse)

        x = torch.cat([sparse, obs_mask], dim=1)

        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        x = self.bottleneck(self.pool(s4))
        x = self.dec4(self._upsample_cat(x, s4))
        x = self.dec3(self._upsample_cat(x, s3))
        x = self.dec2(self._upsample_cat(x, s2))
        x = self.dec1(self._upsample_cat(x, s1))

        return self.out_conv(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)




class MAELoss(nn.Module):
    """
    Masked reconstruction loss:
    compute loss only on missing positions.
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target, obs_mask):
        loss_mask = 1.0 - obs_mask

        if loss_mask.sum() < 1:
            return F.mse_loss(pred, target)

        sq_err = (pred - target) ** 2
        loss = (sq_err * loss_mask).sum() / (loss_mask.sum() + 1e-8)
        return loss


def random_mask(volume, mask_ratio_range=(0.5, 0.9)):
    B, C, Z, H, W = volume.shape
    device = volume.device

    sparse = volume.clone()
    obs_mask = torch.ones(B, 1, Z, 1, 1, device=device)

    for b in range(B):
        mask_ratio = random.uniform(*mask_ratio_range)
        n_mask = int(Z * mask_ratio)
        n_mask = max(1, min(n_mask, Z - 1))

        masked_indices = random.sample(range(Z), n_mask)
        sparse[b, :, masked_indices, :, :] = 0.0
        obs_mask[b, :, masked_indices, :, :] = 0.0

    obs_mask = obs_mask.expand(B, 1, Z, H, W)
    return sparse, obs_mask


def asym_mae(input_channels, num_classes, **kwargs):
    return AsymUNet3D(f=32)


def asym_zmamba(input_channels, num_classes, **kwargs):
    return AsymZMambaUNet3D(
        f=32,
        d_state=16,
        d_conv=4,
        expand=2,
        mamba_stages=("bottleneck", "dec4", "dec3"),
    )

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print("\n--- baseline ---")
    base = AsymUNet3D(f=16).to(device)
    x = torch.rand(1, 1, 32, 64, 64).to(device)
    sparse, mask = random_mask(x, (0.5, 0.7))
    with torch.no_grad():
        y = base(sparse, mask)
    print("baseline out:", y.shape)


    print("\n--- z-mamba ---")
    model = AsymZMambaUNet3D(f=16).to(device)
    with torch.no_grad():
        y = model(sparse, mask)
    print("zmamba out:", y.shape)
