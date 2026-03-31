"""
data/sandstone.py
Drop-in dataset for seannz/svr repo — loads Neumann sandstone .raw CT volumes.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

# ── Sample definitions ─────────────────────────────────────────────

SAMPLE_PREFIXES = {
    "Bandera Brown":     "BanderaBrown",
    "Bandera Gray":      "BanderaGray",
    "Bentheimer":        "Bentheimer",
    "Berea":             "Berea",
    "Berea Sister Gray": "BSG",
    "Berea Upper Gray":  "BUG",
    "Buff Berea":        "BB",
    "CastleGate":        "CastleGate",
    "Kirby":             "Kirby",
    "Leopard":           "Leopard",
    "Parker":            "Parker",
}

TRAIN_SAMPLES = [
    "Berea", "Bandera Brown", "Bandera Gray", "Bentheimer",
    "Berea Sister Gray", "Berea Upper Gray", "Buff Berea", "Kirby",
]
VAL_SAMPLES  = ["Leopard", "CastleGate"]
TEST_SAMPLES = ["Parker"]

VOL_SHAPE         = (1000, 1000, 1000)
SUBSAMPLE_FACTORS = [5, 10, 15]


# ── Path resolver ──────────────────────────────────────────────────

def get_raw_path(dataroot: str, sample_name: str, file_type: str = "filtered") -> str:
    """
    Build full path to a .raw file.
    Structure: dataroot/SampleName/Prefix_suffix.raw/Prefix_suffix.raw
    Uses os.path.normpath to handle both forward and back slashes on Windows.
    """
    prefix = SAMPLE_PREFIXES[sample_name]
    suffix_map = {
        "grayscale": f"{prefix}_2d25um_grayscale.raw",
        "filtered":  f"{prefix}_2d25um_grayscale_filtered.raw",
        "binary":    f"{prefix}_2d25um_binary.raw",
    }
    filename = suffix_map[file_type]
    path = os.path.normpath(
        os.path.join(dataroot, sample_name, filename, filename)
    )
    return path


# ── Patch reader ───────────────────────────────────────────────────

def read_patch(filepath: str, origin: tuple, size: int,
               vol_shape: tuple = VOL_SHAPE) -> np.ndarray:
    """Read one cubic patch via memmap — only loads patch into RAM."""
    z0, h0, w0 = origin
    mmap  = np.memmap(filepath, dtype="uint8", mode="r", shape=vol_shape)
    patch = mmap[z0:z0+size, h0:h0+size, w0:w0+size].copy()
    del mmap
    return patch.astype(np.float32) / 255.0


# ── Dataset class ──────────────────────────────────────────────────

class SandstoneDataset(Dataset):
    """
    Sandstone CT serial section reconstruction dataset.
    Matches the interface expected by seannz/svr train.py.
    """

    def __init__(
        self,
        dataroot:            str,
        sample_names:        list,
        patch_size:          int   = 64,
        stride:              int   = 32,
        subsample_factors:   list  = None,
        input_type:          str   = "filtered",
        augment:             bool  = True,
        max_patches_per_vol: int   = None,
        vol_shape:           tuple = VOL_SHAPE,
    ):
        self.patch_size        = patch_size
        self.subsample_factors = subsample_factors or SUBSAMPLE_FACTORS
        self.input_type        = input_type
        self.augment           = augment
        self.vol_shape         = vol_shape

        Z, H, W = vol_shape
        self.index = []

        print(f"\nBuilding dataset index ({len(sample_names)} samples)...")
        print(f"  dataroot: {dataroot}")

        for name in sample_names:
            path = get_raw_path(dataroot, name, input_type)
            print(f"  Checking: {path}")

            if not os.path.exists(path):
                print(f"  WARNING: {name} not found — skipping")
                continue

            origins = [
                (z, h, w)
                for z in range(0, Z - patch_size + 1, stride)
                for h in range(0, H - patch_size + 1, stride)
                for w in range(0, W - patch_size + 1, stride)
            ]

            if max_patches_per_vol and len(origins) > max_patches_per_vol:
                origins = random.sample(origins, max_patches_per_vol)

            for origin in origins:
                for factor in self.subsample_factors:
                    self.index.append((path, origin, factor, name))

            print(f"  ✓ {name:22s}  {len(origins):6,} origins × {len(self.subsample_factors)} factors")

        print(f"  Total: {len(self.index):,} samples\n")

    def __numinput__(self):  return 1
    def __numclass__(self):  return 1
    def __len__(self):       return len(self.index)

    def __getitem__(self, idx):
        path, origin, factor, name = self.index[idx]

        dense  = read_patch(path, origin, self.patch_size, self.vol_shape)
        sparse = np.zeros_like(dense)
        observed = list(range(0, self.patch_size, factor))
        sparse[observed] = dense[observed]

        if self.augment:
            for axis in range(3):
                if random.random() > 0.5:
                    dense  = np.flip(dense,  axis=axis).copy()
                    sparse = np.flip(sparse, axis=axis).copy()

        return {
            "vol":    torch.FloatTensor(sparse).unsqueeze(0),
            "tgt":    torch.FloatTensor(dense).unsqueeze(0),
            "factor": factor,
            "name":   name,
        }


# ── Factory function ───────────────────────────────────────────────

def sandstone(
    dataroot:            str   = r"C:\Users\PRASANTH\3D-Reconstruction\dataset\DRP-317",
    seed:                int   = 0,
    fraction:            float = 1.0,
    augment:             bool  = True,
    patch_size:          int   = 64,
    stride:              int   = 32,
    subsample_factors:   list  = None,
    input_type:          str   = "filtered",
    max_patches_per_vol: int   = None,
):
    random.seed(seed)
    np.random.seed(seed)

    # Normalise path — handles both forward and back slashes
    dataroot = os.path.normpath(dataroot)

    factors = subsample_factors or SUBSAMPLE_FACTORS

    train_data = SandstoneDataset(dataroot=dataroot, sample_names=TRAIN_SAMPLES,
        patch_size=patch_size, stride=stride, subsample_factors=factors,
        input_type=input_type, augment=augment, max_patches_per_vol=max_patches_per_vol)

    valid_data = SandstoneDataset(dataroot=dataroot, sample_names=VAL_SAMPLES,
        patch_size=patch_size, stride=stride, subsample_factors=[10],
        input_type=input_type, augment=False, max_patches_per_vol=max_patches_per_vol)

    tests_data = SandstoneDataset(dataroot=dataroot, sample_names=TEST_SAMPLES,
        patch_size=patch_size, stride=stride, subsample_factors=[5, 10, 15],
        input_type=input_type, augment=False, max_patches_per_vol=max_patches_per_vol)

    if fraction < 1.0:
        n = int(len(train_data) * fraction)
        train_data.index = random.sample(train_data.index, n)

    return train_data, valid_data, tests_data


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_data, valid_data, tests_data = sandstone(max_patches_per_vol=3)
    loader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=0)
    batch  = next(iter(loader))
    print("vol  :", batch['vol'].shape,  f"[{batch['vol'].min():.3f}, {batch['vol'].max():.3f}]")
    print("tgt  :", batch['tgt'].shape,  f"[{batch['tgt'].min():.3f}, {batch['tgt'].max():.3f}]")
    print("factor:", batch['factor'])
    print("name  :", batch['name'])
    print("✓ passed")