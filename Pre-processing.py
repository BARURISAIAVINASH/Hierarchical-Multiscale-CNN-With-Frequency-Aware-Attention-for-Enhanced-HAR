# transforms_ts2img.py
# ============================================================
# Time-series -> Image-like transforms for HAR preprocessing
# Implements: FT, STFT, CWT (Mexican Hat), HHT (EMD+Hilbert), GAF (GASF/GADF)
# - No smoothing is applied (signal integrity preserved)
# - Designed to be dropped into a GitHub repo
# ============================================================

from __future__ import annotations
import os
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np

# Optional SciPy (recommended for STFT/CWT/Hilbert)
try:
    from scipy.signal import stft as scipy_stft
    from scipy.signal import cwt as scipy_cwt
    from scipy.signal import ricker  # close to Mexican hat
    from scipy.signal import hilbert as scipy_hilbert
except Exception:
    scipy_stft = None
    scipy_cwt = None
    ricker = None
    scipy_hilbert = None

# Optional EMD for HHT
try:
    from PyEMD import EMD  # pip install EMD-signal
except Exception:
    EMD = None


# -----------------------------
# Small helpers
# -----------------------------
def _to_1d_float(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {x.shape}")
    x = x.astype(np.float32, copy=False)
    if not np.isfinite(x).all():
        raise ValueError("Input contains NaN/Inf.")
    return x


def _normalize_minmax(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn, mx = float(x.min()), float(x.max())
    if (mx - mn) < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def _safe_log1p(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.log1p(np.maximum(x, 0.0))


def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = float(x.mean())
    sd = float(x.std())
    if sd < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mu) / (sd + eps)


def _resize_nearest(img: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """Simple nearest resize without extra deps; good enough for preprocessing."""
    h, w = img.shape
    oh, ow = out_hw
    if (h, w) == (oh, ow):
        return img
    ys = (np.linspace(0, h - 1, oh)).astype(np.int32)
    xs = (np.linspace(0, w - 1, ow)).astype(np.int32)
    return img[ys[:, None], xs[None, :]]


# ============================================================
# 1) FT: magnitude spectrum image (1D -> 1D; you can stack channels later)
# ============================================================
def ft_spectrum(
    x: np.ndarray,
    fs: float,
    n_fft: Optional[int] = None,
    one_sided: bool = True,
    log_scale: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (freqs, mag) where mag is the magnitude spectrum.
    Typical image: stack per-axis spectra as rows, or convert mag to a 2D map later.
    """
    x = _to_1d_float(x)
    if fs <= 0:
        raise ValueError("fs must be > 0")

    n = int(n_fft) if n_fft is not None else int(2 ** math.ceil(math.log2(len(x))))
    n = max(n, 8)

    X = np.fft.fft(x, n=n)
    mag = np.abs(X).astype(np.float32)

    if one_sided:
        mag = mag[: n // 2 + 1]
        freqs = np.fft.rfftfreq(n, d=1.0 / fs).astype(np.float32)
    else:
        freqs = np.fft.fftfreq(n, d=1.0 / fs).astype(np.float32)

    if log_scale:
        mag = _safe_log1p(mag)

    if normalize:
        mag = _normalize_minmax(mag)

    return freqs, mag


# ============================================================
# 2) STFT: time-frequency spectrogram image
# ============================================================
def stft_spectrogram(
    x: np.ndarray,
    fs: float,
    nperseg: int = 128,
    noverlap: Optional[int] = None,
    nfft: Optional[int] = None,
    window: str = "hann",
    log_scale: bool = True,
    normalize: bool = True,
    out_hw: Optional[Tuple[int, int]] = None,
) -> Dict[str, np.ndarray]:
    """
    Returns dict with:
      - "f": frequency bins
      - "t": time bins
      - "S": spectrogram magnitude (freq x time), ready as image
    """
    x = _to_1d_float(x)
    if fs <= 0:
        raise ValueError("fs must be > 0")
    if scipy_stft is None:
        raise ImportError("scipy is required for STFT. Install: pip install scipy")

    if noverlap is None:
        noverlap = nperseg // 2

    f, t, Zxx = scipy_stft(
        x,
        fs=fs,
        window=window,
        nperseg=int(nperseg),
        noverlap=int(noverlap),
        nfft=int(nfft) if nfft is not None else None,
        boundary=None,
        padded=False,
    )
    S = np.abs(Zxx).astype(np.float32)

    if log_scale:
        S = _safe_log1p(S)

    if normalize:
        S = _normalize_minmax(S)

    if out_hw is not None:
        S = _resize_nearest(S, out_hw)

    return {"f": f.astype(np.float32), "t": t.astype(np.float32), "S": S}


# ============================================================
# 3) CWT (Mexican Hat): scalogram image
# ============================================================
def cwt_mexican_hat(
    x: np.ndarray,
    widths: np.ndarray,
    fs: float,
    log_scale: bool = True,
    normalize: bool = True,
    out_hw: Optional[Tuple[int, int]] = None,
) -> Dict[str, np.ndarray]:
    """
    Mexican hat wavelet scalogram using SciPy's ricker (Mexican hat-like).
    Returns dict with:
      - "widths": scales
      - "t": time axis
      - "W": abs(cwt) (scale x time)
    """
    x = _to_1d_float(x)
    if fs <= 0:
        raise ValueError("fs must be > 0")
    if scipy_cwt is None or ricker is None:
        raise ImportError("scipy is required for CWT. Install: pip install scipy")

    widths = np.asarray(widths, dtype=np.float32)
    if widths.ndim != 1 or len(widths) < 2:
        raise ValueError("widths must be a 1D array with len >= 2")

    c = scipy_cwt(x, ricker, widths)  # (num_scales, T)
    W = np.abs(c).astype(np.float32)

    if log_scale:
        W = _safe_log1p(W)

    if normalize:
        W = _normalize_minmax(W)

    t = (np.arange(len(x), dtype=np.float32) / float(fs)).astype(np.float32)

    if out_hw is not None:
        W = _resize_nearest(W, out_hw)

    return {"widths": widths, "t": t, "W": W}


# ============================================================
# 4) HHT: EMD + Hilbert spectral features
# ============================================================
@dataclass
class HHTConfig:
    max_imfs: int = 6
    # Hilbert spectrum grid (time x freq image)
    n_f_bins: int = 128
    f_min: float = 0.0
    f_max: Optional[float] = None  # default: Nyquist
    # energy mapping
    log_scale: bool = True
    normalize: bool = True
    out_hw: Optional[Tuple[int, int]] = None


def hht_hilbert_spectrum(
    x: np.ndarray,
    fs: float,
    cfg: HHTConfig = HHTConfig(),
) -> Dict[str, np.ndarray]:
    """
    HHT pipeline:
      1) EMD -> IMFs
      2) Hilbert transform per IMF -> instantaneous amplitude & frequency
      3) Accumulate amplitude (or energy) into a time-frequency image

    Returns dict:
      - "imfs": (K, T) (float32)
      - "hs": (F, T) Hilbert spectrum image (freq x time)
      - "f_bins": frequency bin centers (F,)
      - "t": time axis (T,)
    """
    x = _to_1d_float(x)
    if fs <= 0:
        raise ValueError("fs must be > 0")
    if EMD is None:
        raise ImportError("PyEMD is required for HHT/EMD. Install: pip install EMD-signal")
    if scipy_hilbert is None:
        raise ImportError("scipy is required for Hilbert transform. Install: pip install scipy")

    T = len(x)
    t = (np.arange(T, dtype=np.float32) / float(fs)).astype(np.float32)

    # 1) EMD
    emd = EMD()
    imfs = emd(x)  # (K, T)
    if imfs.ndim != 2:
        raise RuntimeError("Unexpected EMD output shape.")
    imfs = imfs[: cfg.max_imfs].astype(np.float32, copy=False)
    K = imfs.shape[0]

    # Frequency bins
    nyq = 0.5 * fs
    f_max = float(cfg.f_max) if cfg.f_max is not None else nyq
    f_min = float(cfg.f_min)
    f_max = min(f_max, nyq)
    if f_max <= f_min:
        raise ValueError("Invalid frequency bounds for Hilbert spectrum.")
    F = int(cfg.n_f_bins)
    f_bins = np.linspace(f_min, f_max, F).astype(np.float32)

    # 2-3) Hilbert spectrum accumulation
    hs = np.zeros((F, T), dtype=np.float32)

    # instantaneous frequency: ω(t) = dφ/dt ; f(t)=ω/(2π)
    for k in range(K):
        analytic = scipy_hilbert(imfs[k].astype(np.float64))
        amp = np.abs(analytic).astype(np.float32)
        phase = np.unwrap(np.angle(analytic)).astype(np.float32)

        # dφ/dt with finite differences
        dphi = np.diff(phase, prepend=phase[0])
        inst_f = (fs * dphi) / (2.0 * np.pi)  # Hz
        inst_f = np.clip(inst_f, f_min, f_max)

        # Map each time sample to nearest frequency bin and add amplitude
        idx = np.searchsorted(f_bins, inst_f, side="left")
        idx = np.clip(idx, 0, F - 1)
        hs[idx, np.arange(T)] += amp

    if cfg.log_scale:
        hs = _safe_log1p(hs)
    if cfg.normalize:
        hs = _normalize_minmax(hs)

    if cfg.out_hw is not None:
        hs = _resize_nearest(hs, cfg.out_hw)

    return {"imfs": imfs, "hs": hs, "f_bins": f_bins, "t": t}


# ============================================================
# 5) GAF: Gramian Angular Fields (GASF/GADF)
# ============================================================
def gaf(
    x: np.ndarray,
    method: str = "summation",  # "summation" (GASF) or "difference" (GADF)
    scale: str = "minmax_01",   # "minmax_01" or "minmax_-11"
    out_hw: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    GAF encoding (1D -> 2D NxN):
      x~ in [0,1] or [-1,1]
      φ = arccos(x~)
      GASF = cos(φ_i + φ_j)
      GADF = sin(φ_i - φ_j)
    """
    x = _to_1d_float(x)
    N = len(x)

    if scale == "minmax_01":
        x_tilde = _normalize_minmax(x)  # [0,1]
        x_cos = np.clip(x_tilde, 0.0, 1.0)
        phi = np.arccos(x_cos).astype(np.float32)  # [0, π]
    elif scale == "minmax_-11":
        x_tilde = _normalize_minmax(x) * 2.0 - 1.0  # [-1,1]
        x_cos = np.clip(x_tilde, -1.0, 1.0)
        phi = np.arccos(x_cos).astype(np.float32)
    else:
        raise ValueError("scale must be 'minmax_01' or 'minmax_-11'")

    # Build Gramian
    if method.lower() in ["summation", "gasf", "sum"]:
        # cos(φ_i + φ_j)
        gaf_img = np.cos(phi[:, None] + phi[None, :]).astype(np.float32)
    elif method.lower() in ["difference", "gadf", "diff"]:
        # sin(φ_i - φ_j)
        gaf_img = np.sin(phi[:, None] - phi[None, :]).astype(np.float32)
    else:
        raise ValueError("method must be 'summation' (GASF) or 'difference' (GADF)")

    # Map to [0,1] for image usage
    gaf_img = _normalize_minmax(gaf_img)

    if out_hw is not None:
        gaf_img = _resize_nearest(gaf_img, out_hw)

    return gaf_img


# ============================================================
# Convenience: multi-channel (e.g., 3-axis accel -> RGB-like)
# ============================================================
def stack_channels_as_image(ch_imgs: Tuple[np.ndarray, ...]) -> np.ndarray:
    """
    Stack 2D images as channels -> (H, W, C).
    All inputs must have same H, W.
    """
    imgs = [np.asarray(im, dtype=np.float32) for im in ch_imgs]
    hws = {(im.shape[0], im.shape[1]) for im in imgs}
    if len(hws) != 1:
        raise ValueError(f"All channel images must have same H,W. Got: {hws}")
    return np.stack(imgs, axis=-1)


# ============================================================
# Minimal example (safe to delete in your repo)
# ============================================================
if __name__ == "__main__":
    # Dummy signal: mix of sinusoids (no smoothing)
    fs = 50.0
    t = np.arange(0, 5.0, 1.0 / fs)
    x = (0.8 * np.sin(2 * np.pi * 2.0 * t) + 0.4 * np.sin(2 * np.pi * 8.0 * t)).astype(np.float32)

    # FT
    freqs, mag = ft_spectrum(x, fs=fs, n_fft=512)

    # STFT (spectrogram)
    st = stft_spectrogram(x, fs=fs, nperseg=128, noverlap=64, out_hw=(128, 128))

    # CWT (scalogram)
    widths = np.linspace(1, 64, 64)
    cw = cwt_mexican_hat(x, widths=widths, fs=fs, out_hw=(128, 128))

    # GAF (NxN)
    gaf_img = gaf(x, method="summation", scale="minmax_-11", out_hw=(128, 128))

    # HHT (if deps available)
    if EMD is not None and scipy_hilbert is not None:
        hs = hht_hilbert_spectrum(x, fs=fs, cfg=HHTConfig(n_f_bins=128, out_hw=(128, 128)))
        print("HHT hs shape:", hs["hs"].shape)

    print("FT mag:", mag.shape, "| STFT:", st["S"].shape, "| CWT:", cw["W"].shape, "| GAF:", gaf_img.shape)