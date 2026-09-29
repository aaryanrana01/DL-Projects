#!/usr/bin/env python3

# Cell 1
get_ipython().system('pip install -q rasterio scikit-image torchvision kaggle')

# Cell 2
import os, zipfile

from google.colab import files
print("Select your kaggle.json:")
uploaded = files.upload()
os.makedirs("/root/.kaggle", exist_ok=True)
with open("/root/.kaggle/kaggle.json", "wb") as f:
    f.write(list(uploaded.values())[0])
os.chmod("/root/.kaggle/kaggle.json", 0o600)

print("Downloading dataset...")
get_ipython().system('kaggle datasets download -d shubhank001/rice-remote-sensing-images-for-cloud-removal -p /content/')
print("Unzipping...")
with zipfile.ZipFile("/content/rice-remote-sensing-images-for-cloud-removal.zip", "r") as z:
    z.extractall("/content/")
print("Done!")

# Cell 3
import os
from pathlib import Path

def find_rice1(root="/content"):
    """Walk /content and find the RICE1 folder containing cloud/ and label/."""
    for r, dirs, _ in os.walk(root):
        for d in dirs:
            if d.upper() == "RICE1":
                c = Path(r) / d
                if (c / "cloud").exists() and (c / "label").exists():
                    return c
    # fallback: any folder with both cloud/ and label/
    for r, dirs, _ in os.walk(root):
        if "cloud" in dirs and "label" in dirs:
            return Path(r)
    return None

rice1 = find_rice1()

if rice1 is None:
    print("Could not find RICE1. Listing all dirs under /content:")
    for p in sorted(Path("/content").rglob("*")):
        if p.is_dir(): print(" ", p)
    raise RuntimeError("Set RICE1_PATH manually below and re-run.")

RICE1_PATH = str(rice1)
print(f"✓ RICE1 found at: {RICE1_PATH}")
for folder in ["cloud", "label", "Test/cloud", "Test/label"]:
    p = rice1 / folder
    marker = "✓" if p.exists() else "✗"
    count  = len(list(p.glob("*.*"))) if p.exists() else 0
    print(f"  {marker}  {folder:20s}  {count} images")

# Cell 4 · Model & training code (do not edit)
import os
import csv
import json
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image as PILImage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import rasterio
    RASTERIO_OK = True
except ImportError:
    RASTERIO_OK = False

from skimage.metrics import structural_similarity as calc_ssim
from skimage.metrics import peak_signal_noise_ratio as calc_psnr

def load_image(path):
    """Load a .tif or image file as (4, H, W) float32 array."""
    path = Path(path)
    if RASTERIO_OK and path.suffix.lower() in (".tif", ".tiff"):
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)
    else:
        arr = np.array(PILImage.open(path)).astype(np.float32)
        img = arr[np.newaxis] if arr.ndim == 2 else arr.transpose(2, 0, 1)

    # make sure we always have 4 bands
    if img.shape[0] < 4:
        pad = np.zeros((4 - img.shape[0], *img.shape[1:]), dtype=np.float32)
        img = np.concatenate([img, pad], axis=0)
    return img[:4]


def crop_or_pad(img, size=256):
    """Resize image to exactly (C, size, size) by cropping or padding."""
    C, H, W = img.shape
    if H > size:
        start = (H - size) // 2
        img = img[:, start:start + size, :]
    if W > size:
        start = (W - size) // 2
        img = img[:, :, start:start + size]
    C, H, W = img.shape
    if H < size or W < size:
        img = np.pad(img, ((0, 0), (0, size - H), (0, size - W)))
    return img


class CloudDataset(Dataset):
    IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    def __init__(self, pairs, augment=False, band_min=None, band_max=None):
        super().__init__()
        self.pairs   = pairs
        self.augment = augment
        self.band_min = band_min if band_min is not None else np.zeros(4, np.float32)
        self.band_max = band_max if band_max is not None else np.ones(4, np.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        cloudy_path, clear_path = self.pairs[idx]
        cloudy = crop_or_pad(load_image(cloudy_path))
        clear  = crop_or_pad(load_image(clear_path))
        cloudy = self.normalize(cloudy)
        clear  = self.normalize(clear)
        if self.augment:
            cloudy, clear = self.apply_augment(cloudy, clear)
        return torch.from_numpy(cloudy.copy()), torch.from_numpy(clear.copy())

    def normalize(self, img):
        out = img.copy()
        for b in range(4):
            r = float(self.band_max[b] - self.band_min[b])
            if r > 0:
                out[b] = (img[b] - self.band_min[b]) / r
        return np.clip(out, 0.0, 1.0)

    @staticmethod
    def apply_augment(cloudy, clear):
        if np.random.rand() > 0.5:
            cloudy = np.flip(cloudy, axis=2)
            clear  = np.flip(clear,  axis=2)
        if np.random.rand() > 0.5:
            cloudy = np.flip(cloudy, axis=1)
            clear  = np.flip(clear,  axis=1)
        k = np.random.randint(0, 4)
        if k:
            cloudy = np.rot90(cloudy, k, axes=(1, 2))
            clear  = np.rot90(clear,  k, axes=(1, 2))
        for b in range(4):
            factor = 1.0 + np.random.uniform(-0.1, 0.1)
            cloudy[b] = np.clip(cloudy[b] * factor, 0.0, 1.0)
        return cloudy, clear


def get_band_stats(pairs, n_samples=500):
    n   = min(n_samples, len(pairs))
    idx = np.random.choice(len(pairs), n, replace=False)
    print(f"Computing normalisation stats from {n} image pairs...")
    mins = np.full(4,  np.inf,  np.float64)
    maxs = np.full(4, -np.inf, np.float64)
    for i in idx:
        for path in [pairs[i][0], pairs[i][1]]:
            img = load_image(path)
            for b in range(4):
                mins[b] = min(mins[b], float(img[b].min()))
                maxs[b] = max(maxs[b], float(img[b].max()))
    print(f"  min: {mins.round(2)}")
    print(f"  max: {maxs.round(2)}")
    return mins.astype(np.float32), maxs.astype(np.float32)


def _collect_pairs(cloudy_dir, clear_dir):
    cloudy_dir = Path(cloudy_dir)
    clear_dir  = Path(clear_dir)
    if not cloudy_dir.exists():
        raise FileNotFoundError(f"Could not find cloud/ folder at: {cloudy_dir}")
    if not clear_dir.exists():
        raise FileNotFoundError(f"Could not find label/ folder at: {clear_dir}")
    pairs = []
    for f in sorted(cloudy_dir.iterdir()):
        if f.suffix.lower() in CloudDataset.IMG_EXTS:
            match = clear_dir / f.name
            if match.exists():
                pairs.append((f, match))
    if not pairs:
        raise FileNotFoundError(
            f"No matching pairs found.\n  cloud/ -> {cloudy_dir}\n  label/ -> {clear_dir}"
        )
    return pairs


def split_dataset(data_root, seed=42):
    root = Path(data_root)
    train_val_pairs = _collect_pairs(root / "cloud", root / "label")
    test_dir_cloud = root / "Test" / "cloud"
    test_dir_label = root / "Test" / "label"
    if test_dir_cloud.exists() and test_dir_label.exists():
        test_pairs        = _collect_pairs(test_dir_cloud, test_dir_label)
        use_prebuilt_test = True
    else:
        test_pairs        = []
        use_prebuilt_test = False
    n   = len(train_val_pairs)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    if use_prebuilt_test:
        n_train   = int(0.82 * n)
        train_idx = idx[:n_train].tolist()
        val_idx   = idx[n_train:].tolist()
    else:
        n_train   = int(0.70 * n)
        n_val     = int(0.15 * n)
        train_idx = idx[:n_train].tolist()
        val_idx   = idx[n_train:n_train + n_val].tolist()
        test_idx  = idx[n_train + n_val:].tolist()
        test_pairs = [train_val_pairs[i] for i in test_idx]
    train_pairs = [train_val_pairs[i] for i in train_idx]
    val_pairs   = [train_val_pairs[i] for i in val_idx]
    print(
        f"RICE dataset | train_val pairs: {n}"
        + (f" | test pairs (Test/ folder): {len(test_pairs)}" if use_prebuilt_test else "")
    )
    print(f"  -> train: {len(train_pairs)}, val: {len(val_pairs)}, test: {len(test_pairs)}")
    return train_pairs, val_pairs, test_pairs

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        self.window_size = window_size
        self.num_heads   = num_heads
        self.scale       = (dim // num_heads) ** -0.5

        self.rel_pos_bias = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords   = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_f = torch.flatten(coords, 1)
        rel      = coords_f[:, :, None] - coords_f[:, None, :]
        rel      = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size[0] - 1
        rel[:, :, 1] += window_size[1] - 1
        rel[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer("rel_pos_index", rel.sum(-1))

        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        ws   = self.window_size[0] * self.window_size[1]
        bias = self.rel_pos_bias[self.rel_pos_index.view(-1)].view(ws, ws, -1)
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)
        if mask is not None:
            nW   = mask.shape[0]
            attn = (attn.view(B_ // nW, nW, self.num_heads, N, N)
                    + mask.unsqueeze(1).unsqueeze(0)).view(-1, self.num_heads, N, N)
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        x    = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class SwinLayer(nn.Module):

    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=4.):
        super().__init__()
        self.window_size = window_size
        self.shift_size  = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, dim),
        )
        self.attn_mask = None

    def set_input_size(self, H, W):
        if self.shift_size > 0:
            mask = torch.zeros(1, H, W, 1)
            ws, ss = self.window_size, self.shift_size
            cnt = 0
            for h_slice in [slice(0, -ws), slice(-ws, -ss), slice(-ss, None)]:
                for w_slice in [slice(0, -ws), slice(-ws, -ss), slice(-ss, None)]:
                    mask[:, h_slice, w_slice, :] = cnt
                    cnt += 1
            mw = window_partition(mask, self.window_size).view(-1, self.window_size ** 2)
            diff = mw.unsqueeze(1) - mw.unsqueeze(2)
            self.attn_mask = diff.masked_fill(diff != 0, -100.0).masked_fill(diff == 0, 0.0)
        else:
            self.attn_mask = None

    def forward(self, x, H, W):
        B, _, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift_size > 0:
            x = torch.roll(x, (-self.shift_size, -self.shift_size), dims=(1, 2))
        xw   = window_partition(x, self.window_size).view(-1, self.window_size ** 2, C)
        mask = self.attn_mask.to(x.device) if self.attn_mask is not None else None
        aw   = self.attn(xw, mask=mask).view(-1, self.window_size, self.window_size, C)
        x    = window_reverse(aw, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(x, (self.shift_size, self.shift_size), dims=(1, 2))
        x = shortcut + x.view(B, H * W, C)
        return x + self.mlp(self.norm2(x))


class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, depth, window_size=8, mlp_ratio=4.):
        super().__init__()
        self.layers = nn.ModuleList([
            SwinLayer(
                dim, num_heads, window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio,
            )
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def set_input_size(self, H, W):
        for layer in self.layers:
            layer.set_input_size(H, W)

    def forward(self, x, H, W):
        residual = x
        B, C, _, _ = x.shape
        seq = x.flatten(2).transpose(1, 2)
        for layer in self.layers:
            if self.training and seq.requires_grad:
                seq = torch.utils.checkpoint.checkpoint(
                    layer, seq, H, W, use_reentrant=False)
            else:
                seq = layer(seq, H, W)
        x = seq.transpose(1, 2).view(B, C, H, W)
        return residual + self.conv(x)


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm      = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.view(B, H, W, C)
        x = torch.cat([
            x[:, 0::2, 0::2],
            x[:, 1::2, 0::2],
            x[:, 0::2, 1::2],
            x[:, 1::2, 1::2],
        ], dim=-1).view(B, -1, 4 * C)
        return self.reduction(self.norm(x)), H // 2, W // 2


class PatchExpand(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, f"PatchExpand requires even dim, got {dim}"
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm   = nn.LayerNorm(dim // 2)

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = self.expand(x)
        x = x.view(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, 2 * H, 2 * W, C // 2)
        return self.norm(x.view(B, -1, C // 2)), 2 * H, 2 * W

class SwinIRCloudRemoval(nn.Module):

    def __init__(self, in_ch=4, out_ch=4, embed_dim=96,
                 depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=8, mlp_ratio=4.0, img_size=256):
        super().__init__()
        depths    = list(depths)
        num_heads = list(num_heads)
        n_stages  = len(depths)
        self.shallow_conv = nn.Conv2d(in_ch, embed_dim, 3, 1, 1)
        self.enc_blocks  = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        enc_dims = []
        dim = embed_dim
        for i in range(n_stages):
            enc_dims.append(dim)
            self.enc_blocks.append(SwinBlock(dim, num_heads[i], depths[i], window_size, mlp_ratio))
            self.downsamples.append(PatchMerging(dim))
            dim = dim * 2
        self.bottleneck = SwinBlock(dim, num_heads[-1], 2, window_size, mlp_ratio)
        self.upsamples  = nn.ModuleList()
        self.dec_projs  = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(n_stages - 1, -1, -1):
            skip_dim = enc_dims[i]
            up_out   = dim // 2
            self.upsamples.append(PatchExpand(dim))
            self.dec_projs.append(nn.Linear(up_out + skip_dim, skip_dim, bias=False))
            self.dec_blocks.append(SwinBlock(skip_dim, num_heads[i], depths[i], window_size, mlp_ratio))
            dim = skip_dim
        self.out_norm    = nn.LayerNorm(dim)
        self.out_conv    = nn.Conv2d(dim, out_ch, 3, 1, 1)
        self.global_proj = (
            nn.Conv2d(embed_dim, dim, 1, bias=False)
            if dim != embed_dim else nn.Identity()
        )
        self._n_stages = n_stages
        self._enc_dims = enc_dims
        self._setup_masks(img_size, img_size, n_stages)
        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"SwinIR model created - {n_params:,} trainable parameters")

    def _setup_masks(self, H, W, n_stages):
        h, w = H, W
        for i, blk in enumerate(self.enc_blocks):
            blk.set_input_size(h, w)
            h, w = h // 2, w // 2
        self.bottleneck.set_input_size(h, w)
        for blk in self.dec_blocks:
            h, w = h * 2, w * 2
            blk.set_input_size(h, w)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B, _, H, W = x.shape
        feat        = self.shallow_conv(x)
        global_skip = feat
        enc_feats = []
        h, w = H, W
        seq  = feat.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.enc_blocks):
            f = seq.transpose(1, 2).view(B, -1, h, w)
            f = blk(f, h, w)
            enc_feats.append(f)
            seq, h, w = self.downsamples[i](f.flatten(2).transpose(1, 2), h, w)
        f   = seq.transpose(1, 2).view(B, -1, h, w)
        f   = self.bottleneck(f, h, w)
        seq = f.flatten(2).transpose(1, 2)
        for i, (up, proj, blk) in enumerate(zip(self.upsamples, self.dec_projs, self.dec_blocks)):
            seq, h, w = up(seq, h, w)
            skip = enc_feats[self._n_stages - 1 - i].flatten(2).transpose(1, 2)
            seq  = proj(torch.cat([seq, skip], dim=-1))
            f    = seq.transpose(1, 2).view(B, -1, h, w)
            f    = blk(f, h, w)
            seq  = f.flatten(2).transpose(1, 2)
        seq = self.out_norm(seq)
        f   = seq.transpose(1, 2).view(B, -1, H, W)
        f   = f + self.global_proj(global_skip)
        return torch.sigmoid(self.out_conv(f))

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vgg.features.children())[:9])
        for p in self.features.parameters():
            p.requires_grad = False
        self.features.eval()

    def forward(self, pred, target):
        with torch.no_grad():
            feat_pred   = self.features(pred[:, :3].float())
            feat_target = self.features(target[:, :3].float())
        return F.l1_loss(feat_pred, feat_target)


def ssim_loss(pred, target):
    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    score = torch.tensor(0.0, device=pred.device)
    p, t  = pred, target
    for i, w in enumerate(weights):
        mu_p = F.avg_pool2d(p, 3, 1, 1)
        mu_t = F.avg_pool2d(t, 3, 1, 1)
        sig_p  = F.avg_pool2d(p ** 2, 3, 1, 1) - mu_p ** 2
        sig_t  = F.avg_pool2d(t ** 2, 3, 1, 1) - mu_t ** 2
        sig_pt = F.avg_pool2d(p * t,  3, 1, 1) - mu_p * mu_t
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu_p * mu_t + C1) * (2 * sig_pt + C2)) / \
                   ((mu_p ** 2 + mu_t ** 2 + C1) * (sig_p + sig_t + C2))
        score = score + w * ssim_map.mean()
        if i < len(weights) - 1:
            p = F.avg_pool2d(p, 2)
            t = F.avg_pool2d(t, 2)
    return 1.0 - score


class TotalLoss(nn.Module):
    def __init__(self, w_l1=1.0, w_perc=0.05, w_ssim=0.5):
        super().__init__()
        self.w_l1   = w_l1
        self.w_perc = w_perc
        self.w_ssim = w_ssim
        self.perceptual = PerceptualLoss()

    def train(self, mode=True):
        super().train(mode)
        self.perceptual.features.eval()
        return self

    def forward(self, pred, target):
        l1   = F.l1_loss(pred, target)
        perc = self.perceptual(pred, target)
        s    = ssim_loss(pred, target)
        total = self.w_l1 * l1 + self.w_perc * perc + self.w_ssim * s
        return total, l1.item(), perc.item(), s.item()

def compute_metrics(pred, gt):
    psnr_list, ssim_list, cc_list = [], [], []
    for b in range(4):
        p = pred[..., b]
        g = gt[..., b]
        psnr_list.append(calc_psnr(g, p, data_range=1.0))
        ssim_list.append(calc_ssim(g, p, data_range=1.0))
        cc = np.corrcoef(p.ravel(), g.ravel())[0, 1]
        cc_list.append(0.0 if np.isnan(cc) else float(cc))
    return (
        float(np.mean(psnr_list)),
        float(np.mean(ssim_list)),
        float(np.mean(cc_list)),
        psnr_list[3],
        ssim_list[3],
    )


def to_uint8(arr):
    return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)


def save_samples(inputs, preds, targets, out_dir, prefix="sample", n=8):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(min(n, inputs.shape[0])):
        cloudy_rgb = np.stack([to_uint8(inputs[i, b]) for b in range(3)], axis=-1)
        pred_rgb   = np.stack([to_uint8(preds[i,  b]) for b in range(3)], axis=-1)
        gt_rgb     = np.stack([to_uint8(targets[i, b]) for b in range(3)], axis=-1)
        H, W, _ = cloudy_rgb.shape
        gap      = np.full((H, 4, 3), 200, dtype=np.uint8)
        canvas   = np.concatenate([cloudy_rgb, gap, pred_rgb, gap, gt_rgb], axis=1)
        PILImage.fromarray(canvas).save(os.path.join(out_dir, f"{prefix}_{i:03d}.png"))
    print(f"  Saved {min(n, inputs.shape[0])} samples -> {out_dir}/")


@torch.no_grad()
def evaluate(model, loader, device, vis_dir=None, epoch=None):
    model.eval()
    all_metrics = []
    saved = False
    for batch_idx, (inputs, targets) in enumerate(loader):
        preds  = model(inputs.to(device)).cpu().float().numpy()
        inp_np = inputs.float().numpy()
        tgt_np = targets.float().numpy()
        if vis_dir and not saved:
            tag = f"epoch_{epoch:03d}" if epoch is not None else "final"
            save_samples(inp_np, preds, tgt_np, os.path.join(vis_dir, tag))
            saved = True
        for i in range(preds.shape[0]):
            all_metrics.append(compute_metrics(
                preds[i].transpose(1, 2, 0),
                tgt_np[i].transpose(1, 2, 0),
            ))
    return tuple(np.array(all_metrics).mean(axis=0).tolist())

CSV_FIELDS = ["epoch", "train_loss", "l1", "perc", "ssim_loss",
              "val_psnr", "val_ssim", "val_cc", "nir_psnr", "nir_ssim", "lr"]


def init_csv(path):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_FIELDS)


def write_csv_row(path, row):
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


def plot_training_curves(csv_path, out_path):
    data = {k: [] for k in CSV_FIELDS}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            for k in CSV_FIELDS:
                if k != "lr":
                    data[k].append(float(row[k]))
    epochs = data["epoch"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("SwinIR Cloud Removal - Training Progress", fontsize=13)
    plots = [
        (axes[0, 0], data["train_loss"], "Training Loss",     "Loss",  "coral"),
        (axes[0, 1], data["val_psnr"],   "Validation PSNR",   "dB",    "steelblue"),
        (axes[0, 2], data["val_ssim"],   "Validation SSIM",   "SSIM",  "steelblue"),
        (axes[1, 0], data["val_cc"],     "Correlation Coeff", "CC",    "steelblue"),
        (axes[1, 1], data["nir_psnr"],   "NIR PSNR",          "dB",    "teal"),
        (axes[1, 2], data["nir_ssim"],   "NIR SSIM",          "SSIM",  "teal"),
    ]
    for ax, y, title, ylabel, color in plots:
        ax.plot(epochs, y, color=color, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved -> {out_path}")


# =============================================================================
#  FIXED train() — robust checkpoint resume
#  Changes vs. original:
#   1. `best_ssim` and `no_improve` are initialised BEFORE the resume block
#      so they are never clobbered by the "if not exists" guard that followed.
#   2. Scheduler states are saved/restored so LR curve continues correctly.
#   3. `start_epoch` defaults to 1 so the variable is always defined.
#   4. A clear "Resuming …" vs "Starting fresh …" log line is printed.
# =============================================================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    vis_dir     = os.path.join(args.checkpoint_dir, "visuals")
    metrics_csv = os.path.join(args.checkpoint_dir, "metrics.csv")
    init_csv(metrics_csv)

    train_pairs, val_pairs, test_pairs = split_dataset(args.data_root)

    # --- band stats (compute once, then load from Drive) ---
    stats_path = os.path.join(args.checkpoint_dir, "band_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            s = json.load(f)
        band_min = np.array(s["band_min"], np.float32)
        band_max = np.array(s["band_max"], np.float32)
        print(f"Loaded band stats from {stats_path}")
    else:
        band_min, band_max = get_band_stats(train_pairs)
        with open(stats_path, "w") as f:
            json.dump({"band_min": band_min.tolist(), "band_max": band_max.tolist()}, f)

    train_ds = CloudDataset(train_pairs, augment=True,  band_min=band_min, band_max=band_max)
    val_ds   = CloudDataset(val_pairs,   augment=False, band_min=band_min, band_max=band_max)
    test_ds  = CloudDataset(test_pairs,  augment=False, band_min=band_min, band_max=band_max)

    nw = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False,
                              num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
    test_loader  = DataLoader(test_ds,  args.batch_size, shuffle=False,
                              num_workers=nw, pin_memory=True, persistent_workers=nw > 0)

    model = SwinIRCloudRemoval(
        embed_dim   = args.embed_dim,
        depths      = args.depths,
        num_heads   = args.num_heads,
        window_size = args.window_size,
        mlp_ratio   = args.mlp_ratio,
    ).to(device)

    criterion = TotalLoss(args.w_l1, args.w_perc, args.w_ssim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.999), weight_decay=0.01)

    # --- schedulers (built before resume so we can restore their states) ---
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=5)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - 5), eta_min=1e-6)

    # -------------------------------------------------------------------------
    #  FIX: initialise tracking variables HERE so they are always defined
    # -------------------------------------------------------------------------
    start_epoch = 1
    best_ssim   = 0.0
    no_improve  = 0

    checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint.pth")

    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path} — resuming training …")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        # restore scheduler states if present (saved by this script)
        if "warmup_sched" in checkpoint:
            warmup_sched.load_state_dict(checkpoint["warmup_sched"])
        if "cosine_sched" in checkpoint:
            cosine_sched.load_state_dict(checkpoint["cosine_sched"])

        start_epoch = checkpoint["epoch"] + 1          # resume from next epoch
        best_ssim   = checkpoint.get("best_ssim", 0.0) # FIX: read from ckpt
        no_improve  = checkpoint.get("no_improve", 0)  # FIX: read from ckpt

        print(f"  Resumed from epoch {start_epoch - 1}  "
              f"| best_ssim so far: {best_ssim:.4f}  "
              f"| no_improve count: {no_improve}")
    else:
        print("No checkpoint found — starting fresh training.")

    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    # restore scaler state if saved
    if os.path.exists(checkpoint_path):
        ckpt_tmp = torch.load(checkpoint_path, map_location=device)
        if "scaler" in ckpt_tmp:
            scaler.load_state_dict(ckpt_tmp["scaler"])

    if device.type == "cuda":
        torch.cuda.empty_cache()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = l1_sum = perc_sum = ssim_sum = 0.0

        for step, (inputs, targets) in enumerate(train_loader):
            inputs  = inputs.to(device)
            targets = targets.to(device)
            inputs.requires_grad_(True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                preds = model(inputs)
                loss, l1_v, perc_v, ssim_v = criterion(preds, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            l1_sum     += l1_v
            perc_sum   += perc_v
            ssim_sum   += ssim_v

            if step % 50 == 0:
                print(f"  Epoch {epoch}/{args.epochs} | step {step:04d} | "
                      f"loss={loss.item():.4f} l1={l1_v:.4f} ssim={ssim_v:.4f}")

        # update learning rate
        if epoch <= 5:
            warmup_sched.step()
        else:
            cosine_sched.step()

        lr = optimizer.param_groups[0]["lr"]
        n  = len(train_loader)

        # validation
        psnr, ssim, cc, nir_psnr, nir_ssim = evaluate(
            model, val_loader, device, vis_dir=vis_dir, epoch=epoch)

        print(f"Epoch {epoch:03d} | loss={total_loss/n:.4f} | "
              f"PSNR={psnr:.2f} SSIM={ssim:.4f} CC={cc:.4f} | "
              f"NIR PSNR={nir_psnr:.2f} NIR SSIM={nir_ssim:.4f}")

        write_csv_row(metrics_csv, {
            "epoch":      epoch,
            "train_loss": round(total_loss / n, 6),
            "l1":         round(l1_sum   / n, 6),
            "perc":       round(perc_sum  / n, 6),
            "ssim_loss":  round(ssim_sum  / n, 6),
            "val_psnr":   round(psnr,     4),
            "val_ssim":   round(ssim,     6),
            "val_cc":     round(cc,       6),
            "nir_psnr":   round(nir_psnr, 4),
            "nir_ssim":   round(nir_ssim, 6),
            "lr":         f"{lr:.2e}",
        })

        # --- save best model ---
        if ssim > best_ssim:
            best_ssim  = ssim
            no_improve = 0
            ckpt = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ssim":      ssim,
                "psnr":      psnr,
                "band_min":  band_min.tolist(),
                "band_max":  band_max.tolist(),
                "cfg": {
                    "embed_dim":   args.embed_dim,
                    "depths":      args.depths,
                    "num_heads":   args.num_heads,
                    "window_size": args.window_size,
                },
            }, ckpt)
            print(f"  ✓ Best model saved (SSIM={best_ssim:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

        # -------------------------------------------------------------------------
        #  FIX: save checkpoint with ALL state needed to resume perfectly
        # -------------------------------------------------------------------------
        torch.save({
            "epoch":        epoch,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "warmup_sched": warmup_sched.state_dict(),   # NEW
            "cosine_sched": cosine_sched.state_dict(),   # NEW
            "scaler":       scaler.state_dict(),          # NEW
            "best_ssim":    best_ssim,
            "no_improve":   no_improve,                   # NEW
        }, checkpoint_path)
        print(f"  Checkpoint saved → {checkpoint_path}  (epoch {epoch})")

        # periodic snapshot every 10 epochs
        if epoch % 10 == 0:
            snap = os.path.join(args.checkpoint_dir, f"epoch_{epoch:03d}.pth")
            torch.save(model.state_dict(), snap)
            print(f"  Snapshot saved → {snap}")

    # --- final curves ---
    try:
        plot_training_curves(metrics_csv,
                             os.path.join(args.checkpoint_dir, "training_curves.png"))
    except Exception as e:
        print(f"Could not plot curves: {e}")

    print("\nEvaluating best model on test set …")
    best_ckpt = os.path.join(args.checkpoint_dir, "best_model.pth")
    ckpt_data = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt_data["model"])
    psnr, ssim, cc, nir_psnr, nir_ssim = evaluate(
        model, test_loader, device, vis_dir=vis_dir)

    print(f"\n{'='*50}")
    print(f"  Test PSNR     : {psnr:.2f} dB")
    print(f"  Test SSIM     : {ssim:.4f}")
    print(f"  Test CC       : {cc:.4f}")
    print(f"  NIR Test PSNR : {nir_psnr:.2f} dB")
    print(f"  NIR Test SSIM : {nir_ssim:.4f}")
    print(f"{'='*50}")


def predict(model_path, cloudy_path, out_path="prediction.tif"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device)
    cfg  = ckpt.get("cfg", {})
    model = SwinIRCloudRemoval(
        embed_dim   = cfg.get("embed_dim",   96),
        depths      = cfg.get("depths",      [2, 2, 6, 2]),
        num_heads   = cfg.get("num_heads",   [3, 6, 12, 24]),
        window_size = cfg.get("window_size", 8),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    band_min = np.array(ckpt.get("band_min", [0.0]*4), np.float32)
    band_max = np.array(ckpt.get("band_max", [1.0]*4), np.float32)
    img = crop_or_pad(load_image(cloudy_path)).astype(np.float32)
    for b in range(4):
        r = float(band_max[b] - band_min[b])
        if r > 0:
            img[b] = (img[b] - band_min[b]) / r
    img = np.clip(img, 0.0, 1.0)
    with torch.no_grad():
        out = model(torch.from_numpy(img[np.newaxis]).to(device)).cpu().numpy()[0]
    if not out_path.lower().endswith((".tif", ".tiff")):
        out_path = out_path.rsplit(".", 1)[0] + ".tif"
    if RASTERIO_OK:
        cloudy_p = Path(cloudy_path)
        if cloudy_p.suffix.lower() in (".tif", ".tiff"):
            with rasterio.open(cloudy_path) as _src:
                profile = _src.profile.copy()
        else:
            profile = {}
        h, w = out.shape[1], out.shape[2]
        for bad_key in ("BLOCKXSIZE", "BLOCKYSIZE", "TILED", "INTERLEAVE"):
            profile.pop(bad_key, None)
        profile.update(driver="GTiff", count=4, dtype="float32", width=w, height=h)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out)
        print(f"Saved GeoTIFF: {out_path}")
    else:
        np.save(out_path.replace(".tif", ".npy"), out)
        print(f"Saved numpy: {out_path.replace('.tif', '.npy')}")
    preview = np.stack([to_uint8(out[b]) for b in range(3)], axis=-1)
    preview_path = out_path.replace(".tif", "_preview.png").replace(".tiff", "_preview.png")
    PILImage.fromarray(preview).save(preview_path)
    print(f"Saved preview: {preview_path}")

# Cell 5
from google.colab import drive
drive.mount('/content/drive')

# =============================================================================
#  Cell 5 · Configure & run training
# =============================================================================
import os
from types import SimpleNamespace

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ── NOTE ──────────────────────────────────────────────────────────────────────
# Make sure Cell 3 (dataset download) has already run so RICE1_PATH is defined.
# The checkpoint_dir points to Google Drive so checkpoints survive disconnects.
# ─────────────────────────────────────────────────────────────────────────────
args = SimpleNamespace(
    # --- paths ---
    data_root      = RICE1_PATH,          # set automatically by Cell 3
    checkpoint_dir = "/content/drive/MyDrive/checkpoints",

    # --- training ---
    epochs     = 50,
    batch_size = 2,
    lr         = 2e-4,
    patience   = 15,

    # --- model ---
    embed_dim   = 64,
    depths      = [2, 2, 4, 2],
    num_heads   = [2, 4, 8, 16],
    window_size = 8,
    mlp_ratio   = 4.0,

    # --- loss weights ---
    w_l1  = 1.0,
    w_perc= 0.01,
    w_ssim= 0.5,

    # --- inference: leave None to train ---
    predict        = None,
    predict_output = "/content/prediction.tif",
    model_path     = None,
)

os.makedirs(args.checkpoint_dir, exist_ok=True)

if args.predict:
    if args.model_path is None:
        args.model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    print(f"Running inference on: {args.predict}")
    predict(args.model_path, args.predict, args.predict_output)
else:
    train(args)
