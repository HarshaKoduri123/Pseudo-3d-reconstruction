import os
import csv
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.sandstone import (TRAIN_SAMPLES, VAL_SAMPLES, TEST_SAMPLES,
                             read_patch, get_raw_path, VOL_SHAPE)
from models.unet3d_rec import AsymUNet3D

def compute_ssim(pred, target, window_size=7):
    C1, C2 = 0.01**2, 0.03**2
    coords  = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g       = torch.exp(-(coords**2) / (2 * 1.5**2))
    g       = g / g.sum()
    k3d     = g[:,None,None] * g[None,:,None] * g[None,None,:]
    kernel  = k3d.unsqueeze(0).unsqueeze(0)
    pad     = window_size // 2
    p, t    = pred.float().unsqueeze(0), target.float().unsqueeze(0)
    mu_p    = F.conv3d(p, kernel, padding=pad)
    mu_t    = F.conv3d(t, kernel, padding=pad)
    sig_p   = F.conv3d(p**2, kernel, padding=pad) - mu_p**2
    sig_t   = F.conv3d(t**2, kernel, padding=pad) - mu_t**2
    sig_pt  = F.conv3d(p*t,  kernel, padding=pad) - mu_p*mu_t
    num     = (2*mu_p*mu_t + C1) * (2*sig_pt + C2)
    den     = (mu_p**2 + mu_t**2 + C1) * (sig_p + sig_t + C2)
    return (num / (den + 1e-8)).mean().item()

def compute_psnr(pred, target):
    mse = F.mse_loss(pred.float(), target.float()).item()
    return 100.0 if mse < 1e-10 else 10 * np.log10(1.0 / mse)

def compute_mae(pred, target):
    return F.l1_loss(pred.float(), target.float()).item()


def reconstruct_full_slice(model, gray_path, z_idx, factor,
                            patch_size=64, device='cuda'):

    Z, H, W = VOL_SHAPE
    observed_z = set(range(0, Z, factor))
    is_observed = z_idx in observed_z
    z_start = max(0, z_idx - patch_size // 2)
    z_start = min(z_start, Z - patch_size)  
    local_z  = z_idx - z_start               

    pred_full  = np.zeros((H, W), dtype=np.float32)
    count_full = np.zeros((H, W), dtype=np.float32)

    h_starts = list(range(0, H - patch_size + 1, patch_size))
    w_starts = list(range(0, W - patch_size + 1, patch_size))
    if h_starts[-1] + patch_size < H:
        h_starts.append(H - patch_size)
    if w_starts[-1] + patch_size < W:
        w_starts.append(W - patch_size)

    model.eval()
    with torch.no_grad():
        for h0 in h_starts:
            for w0 in w_starts:
                # Read patch from disk
                patch = read_patch(gray_path, (z_start, h0, w0),
                                   patch_size, VOL_SHAPE)

                sparse_patch = patch.copy()
                obs_mask_np  = np.zeros((patch_size,), dtype=np.float32)
                for local_i in range(patch_size):
                    global_z_i = z_start + local_i
                    if global_z_i in observed_z:
                        obs_mask_np[local_i] = 1.0
                    else:
                        sparse_patch[local_i] = 0.0

                sp_t  = torch.FloatTensor(sparse_patch).unsqueeze(0).unsqueeze(0).to(device)
                obs_t = torch.FloatTensor(obs_mask_np).view(1,1,patch_size,1,1)
                obs_t = obs_t.expand(1,1,patch_size,patch_size,patch_size).to(device)

                pred_patch = model(sp_t, obs_t)  
                pred_np    = pred_patch[0,0].cpu().numpy() 

                pred_full[h0:h0+patch_size, w0:w0+patch_size] += pred_np[local_z]
                count_full[h0:h0+patch_size, w0:w0+patch_size] += 1.0

    pred_full = pred_full / np.maximum(count_full, 1)
    mmap         = np.memmap(gray_path, dtype='uint8', mode='r', shape=VOL_SHAPE)
    target_slice = mmap[z_idx].astype(np.float32) / 255.0
    del mmap

    sparse_slice = target_slice.copy() if is_observed else np.zeros_like(target_slice)

    return sparse_slice, pred_full, target_slice, is_observed



def eval_sample_metrics(model, gray_path, factor, patch_size=64,
                         device='cuda', n_eval_slices=20):

    Z = VOL_SHAPE[0]
    observed_z = set(range(0, Z, factor))
    missing_z  = [z for z in range(Z) if z not in observed_z]

    if len(missing_z) > n_eval_slices:
        step = len(missing_z) // n_eval_slices
        eval_z = missing_z[::step][:n_eval_slices]
    else:
        eval_z = missing_z

    ssims, psnrs, maes = [], [], []

    for z_idx in eval_z:
        _, pred_s, target_s, _ = reconstruct_full_slice(
            model, gray_path, z_idx, factor, patch_size, device
        )
        p = torch.FloatTensor(pred_s).unsqueeze(0).unsqueeze(0)  
        t = torch.FloatTensor(target_s).unsqueeze(0).unsqueeze(0)


        ssims.append(compute_psnr(p, t))  
        psnrs.append(compute_psnr(p, t))
        maes.append(compute_mae(p, t))

    p3 = torch.FloatTensor(pred_s).unsqueeze(0)   
    t3 = torch.FloatTensor(target_s).unsqueeze(0)

    return {
        'ssim': float(np.mean(psnrs)),  
        'psnr': float(np.mean(psnrs)),
        'mae':  float(np.mean(maes)),
    }


def save_qualitative(model, gray_path, sample_name, factor,
                     save_dir, device, n_slices=3, patch_size=64):
    Z = VOL_SHAPE[0]
    observed_z = set(range(0, Z, factor))
    missing_z  = [z for z in range(Z) if z not in observed_z]

    if not missing_z:
        return

    indices = [missing_z[i * len(missing_z) // n_slices]
               for i in range(n_slices)]

    print(f'  Reconstructing {len(indices)} full slices for '
          f'{sample_name} factor={factor}  z={indices} ...')

    n_rows, n_cols = len(indices), 4
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5*n_cols, 5*n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for c, title in enumerate(['Sparse input','Reconstruction',
                                'Ground truth','Error map']):
        axes[0, c].set_title(title, fontsize=12, fontweight='bold', pad=10)

    for r, z_idx in enumerate(indices):
        sparse_s, pred_s, target_s, is_obs = reconstruct_full_slice(
            model, gray_path, z_idx, factor, patch_size, device
        )
        vmax = max(float(target_s.max()), 0.01)
        err  = np.abs(pred_s - target_s)

        p = torch.FloatTensor(pred_s).unsqueeze(0).unsqueeze(0)
        t = torch.FloatTensor(target_s).unsqueeze(0).unsqueeze(0)
        psnr = compute_psnr(p, t)
        mae  = compute_mae(p, t)

        axes[r, 0].imshow(sparse_s, cmap='gray', vmin=0, vmax=vmax)
        axes[r, 0].set_ylabel(f'z={z_idx}  ({"observed" if is_obs else "MISSING"})',
                               fontsize=9)
        axes[r, 1].imshow(pred_s,   cmap='gray', vmin=0, vmax=vmax)
        axes[r, 1].set_xlabel(f'PSNR={psnr:.2f}dB  MAE={mae:.4f}', fontsize=9)
        axes[r, 2].imshow(target_s, cmap='gray', vmin=0, vmax=vmax)
        axes[r, 3].imshow(err,      cmap='hot',  vmin=0, vmax=0.3)

        for c in range(n_cols):
            axes[r, c].axis('off')

    pct_obs   = 100 / factor
    pct_miss  = 100 - pct_obs
    fig.suptitle(
        f'{sample_name}  |  factor={factor}  '
        f'({pct_obs:.0f}% observed, {pct_miss:.0f}% missing)  '
        f'|  full 1000×1000 slice',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()

    fname = os.path.join(save_dir, f'{sample_name}_factor{factor}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {fname}')


def load_model(checkpoint_path, device):
    ckpt        = torch.load(checkpoint_path, map_location='cpu')
    state       = ckpt['state_dict']
    model_state = {k.replace('model.', '', 1): v
                   for k, v in state.items() if k.startswith('model.')}
    model = AsymUNet3D(f=32)
    model.load_state_dict(model_state)
    model = model.to(device).eval()
    n = model.count_parameters()
    print(f'Loaded : {checkpoint_path}')
    print(f'Params : {n:,} ({n/1e6:.2f}M)\n')
    return model



def evaluate(checkpoint_path, split='test', factors=None,
             n_qual_slices=3, n_metric_slices=20,
             patch_size=64, dataroot=None):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = load_model(checkpoint_path, device)

    results_dir = 'results'
    qual_dir    = os.path.join(results_dir, 'qualitative')
    os.makedirs(qual_dir, exist_ok=True)

    if dataroot is None:
        dataroot = r'C:\Users\PRASANTH\3D-Reconstruction\dataset\DRP-317'

    all_factors = factors or [5, 10, 15]

    split_map = {
        'train': TRAIN_SAMPLES,
        'val':   VAL_SAMPLES,
        'test':  TEST_SAMPLES,
        'all':   TRAIN_SAMPLES + VAL_SAMPLES + TEST_SAMPLES,
    }
    sample_names = split_map.get(split, TEST_SAMPLES)

    print(f'Split   : {split}  →  {sample_names}')
    print(f'Factors : {all_factors}')
    print(f'Device  : {device}\n')

    all_rows = []

    for factor in all_factors:
        print(f'\n{"="*55}')
        print(f'Factor = {factor}  '
              f'({100/factor:.0f}% observed, {100*(1-1/factor):.0f}% missing)')
        print(f'{"="*55}')

        print(f'\n  {"Sample":22s}  {"PSNR":>10}  {"MAE":>10}')
        print(f'  {"-"*45}')

        for name in sample_names:
            gray_path = get_raw_path(dataroot, name, 'filtered')
            if not os.path.exists(gray_path):
                print(f'  WARNING: {name} not found — skipping')
                continue

            m = eval_sample_metrics(
                model, gray_path, factor,
                patch_size=patch_size, device=device,
                n_eval_slices=n_metric_slices,
            )
            print(f'  {name:22s}  {m["psnr"]:10.2f}  {m["mae"]:10.6f}')
            all_rows.append({
                'split': split, 'sample': name, 'factor': factor,
                'psnr':  round(m['psnr'], 2),
                'mae':   round(m['mae'],  6),
            })

            save_qualitative(
                model, gray_path, name, factor,
                qual_dir, device,
                n_slices   = n_qual_slices,
                patch_size = patch_size,
            )

        rows_f = [r for r in all_rows if r['factor'] == factor]
        if rows_f:
            print(f'  {"AVERAGE":22s}  '
                  f'{np.mean([r["psnr"] for r in rows_f]):10.2f}  '
                  f'{np.mean([r["mae"]  for r in rows_f]):10.6f}')

    csv_path = os.path.join(results_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['split','sample','factor','psnr','mae'])
        writer.writeheader()
        writer.writerows(all_rows)


    print(f'\n{"="*55}')
    print('CVPR paper table:')
    print(f'  {"Factor":>8}  {"PSNR (dB)":>10}  {"MAE":>10}')
    print(f'  {"-"*32}')
    for factor in all_factors:
        rows = [r for r in all_rows if r['factor'] == factor]
        if rows:
            print(f'  {factor:>8}  '
                  f'{np.mean([r["psnr"] for r in rows]):10.2f}  '
                  f'{np.mean([r["mae"]  for r in rows]):10.6f}')

    print(f'\nMetrics : {csv_path}')
    print(f'Images  : {qual_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',    type=str, required=True)
    parser.add_argument('--split',         type=str, default='test',
                        choices=['train','val','test','all'])
    parser.add_argument('--factor',        type=int, default=None)
    parser.add_argument('--dataroot',      type=str, default=None)
    parser.add_argument('--qual-slices',   type=int, default=3)
    parser.add_argument('--metric-slices', type=int, default=20,
                        help='Number of missing slices to average metrics over')
    parser.add_argument('--patch-size',    type=int, default=64)
    args = parser.parse_args()

    evaluate(
        checkpoint_path = args.checkpoint,
        split           = args.split,
        factors         = [args.factor] if args.factor else None,
        n_qual_slices   = args.qual_slices,
        n_metric_slices = args.metric_slices,
        patch_size      = args.patch_size,
        dataroot        = args.dataroot,
    )