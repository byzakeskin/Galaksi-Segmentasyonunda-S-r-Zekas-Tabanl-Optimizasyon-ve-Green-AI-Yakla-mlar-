"""
Sentetik Galaksi Veri Seti Üretici
====================================
Gerçek JWST/SDSS verisi olmadan da çalışabilmek için
fiziksel olarak gerçekçi sentetik galaksi görüntüleri üretir.

Galaksi profili: Sérsic profili (n=1 disk, n=4 bulge)
Gürültü: Poisson + Gaussian (SNR parametresi ile kontrol edilir)
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def sersic_profile(r: np.ndarray, I_e: float, r_e: float, n: float) -> np.ndarray:
    """Sérsic yoğunluk profili."""
    b_n = 2 * n - 1 / 3 + 4 / (405 * n)   # Ciotti & Bertin (1999) yaklaşımı
    return I_e * np.exp(-b_n * ((r / r_e) ** (1.0 / n) - 1.0))


def make_galaxy_image(
    size:       int   = 128,
    snr:        float = 10.0,    # sinyal-gürültü oranı (yüksek = temiz)
    galaxy_type: str  = "spiral",# "spiral", "elliptical", "lenticular"
    rng:        np.random.Generator = None
) -> tuple:
    """
    Tek bir galaksi görüntüsü + ikili maske üretir.
    
    Döner: (image [H,W] float32 [0,1]), (mask [H,W] uint8 {0,1})
    """
    if rng is None:
        rng = np.random.default_rng()

    H = W = size
    cx, cy = rng.uniform(size * 0.3, size * 0.7, 2)

    Y, X = np.ogrid[:H, :W]
    dx, dy = X - cx, Y - cy

    # Eliptik galaksi şekli (eksen oranı ve döndürme açısı)
    axial_ratio = rng.uniform(0.4, 1.0)
    angle       = rng.uniform(0, np.pi)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    Xr = cos_a * dx + sin_a * dy
    Yr = -sin_a * dx + cos_a * dy
    r  = np.sqrt(Xr ** 2 + (Yr / axial_ratio) ** 2)

    # Sérsic profili parametreleri
    r_e = rng.uniform(size * 0.06, size * 0.18)
    if galaxy_type == "elliptical":
        n_sersic = rng.uniform(3.0, 6.0)
    elif galaxy_type == "lenticular":
        n_sersic = rng.uniform(1.5, 3.0)
    else:   # spiral → disk bileşeni + küçük bulge
        n_sersic = 1.0

    image = sersic_profile(r, I_e=1.0, r_e=r_e, n=n_sersic)

    # Spiral kollar (yalnızca spiral galaksiler)
    if galaxy_type == "spiral":
        n_arms = rng.integers(2, 5)
        arm_strength = rng.uniform(0.2, 0.6)
        theta = np.arctan2(Yr, Xr)
        pitch_angle = rng.uniform(0.2, 0.5)
        for arm in range(n_arms):
            arm_theta = theta - np.log(r / (r_e * 0.5) + 1e-6) / pitch_angle
            arm_phase  = 2 * np.pi * arm / n_arms
            arm_mask   = np.exp(-((arm_theta - arm_phase) % np.pi) ** 2 / 0.05)
            radial_mask = np.exp(-(r - r_e * 1.5) ** 2 / (r_e * 0.8) ** 2)
            image += arm_strength * arm_mask * radial_mask

    # Gürültü ekleme
    signal = image.max()
    noise_sigma = signal / snr
    noise = rng.normal(0, noise_sigma, (H, W))
    image = np.clip(image + noise, 0, None)

    # Normalize
    if image.max() > 0:
        image = image / image.max()
    image = image.astype(np.float32)

    # İkili maske (galaksi pikselleri > arka plan eşiği)
    threshold = sersic_profile(3 * r_e, I_e=1.0, r_e=r_e, n=n_sersic)
    mask = (sersic_profile(r, I_e=1.0, r_e=r_e, n=n_sersic) > threshold + 1e-6).astype(np.uint8)

    return image, mask


class SyntheticGalaxyDataset(Dataset):
    """
    PyTorch Dataset: sentetik galaksi görüntüleri.
    
    Parametreler
    ------------
    n_samples   : kaç görüntü üretileceği
    image_size  : kare görüntü boyutu (piksel)
    snr_range   : (min_snr, max_snr) — düşük SNR: zor, yüksek SNR: kolay
    galaxy_types: üretilecek galaksi tipleri
    augment     : eğitim artırımı (çevirme, döndürme)
    seed        : tekrarlanabilirlik
    """

    GALAXY_TYPES = ["spiral", "elliptical", "lenticular"]

    def __init__(self,
                 n_samples:    int   = 500,
                 image_size:   int   = 128,
                 snr_range:    tuple = (5.0, 20.0),
                 galaxy_types: list  = None,
                 augment:      bool  = False,
                 seed:         int   = 42):
        self.n_samples    = n_samples
        self.image_size   = image_size
        self.snr_range    = snr_range
        self.galaxy_types = galaxy_types or self.GALAXY_TYPES
        self.augment      = augment
        self.rng          = np.random.default_rng(seed)

        print(f"[Dataset] {n_samples} galaksi görüntüsü üretiliyor "
              f"(SNR: {snr_range[0]:.1f}–{snr_range[1]:.1f})...")
        self.images, self.masks = self._generate()
        print(f"[Dataset] Üretim tamamlandı. Boyut: {self.images.shape}")

    def _generate(self):
        imgs, msks = [], []
        for _ in range(self.n_samples):
            snr  = self.rng.uniform(*self.snr_range)
            gtype = self.rng.choice(self.galaxy_types)
            img, msk = make_galaxy_image(self.image_size, snr, gtype, self.rng)
            imgs.append(img)
            msks.append(msk)
        return np.stack(imgs), np.stack(msks)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        img  = self.images[idx]   # [H, W]
        mask = self.masks[idx]    # [H, W]

        # Veri artırımı (augmentation)
        if self.augment:
            if self.rng.random() > 0.5:
                img  = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
            if self.rng.random() > 0.5:
                img  = np.flipud(img).copy()
                mask = np.flipud(mask).copy()
            k = self.rng.integers(4)
            if k > 0:
                img  = np.rot90(img,  k).copy()
                mask = np.rot90(mask, k).copy()

        # [H, W] → [1, H, W] tensor
        img_t  = torch.from_numpy(img).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        return img_t, mask_t