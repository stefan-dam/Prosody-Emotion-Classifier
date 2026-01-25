import numpy as np
import librosa
from scipy.signal import savgol_filter

EPS = 1e-6

def _pre_emphasis(x, coef):
    if coef <= 0.0:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - coef * x[:-1]
    return y

def _robust_normalize(x, mask=None):
    if mask is not None:
        x_sel = x[mask]
    else:
        x_sel = x
    if x_sel.size < 2:
        return np.zeros_like(x)
    median = np.median(x_sel)
    iqr = np.percentile(x_sel, 75) - np.percentile(x_sel, 25)
    scale = iqr if iqr > EPS else (np.std(x_sel) + EPS)
    return (x - median) / scale

def extract_prosody(
    x,
    sr,
    hop_ms=10.0,
    frame_ms=25.0,
    f0_frame_ms=40.0,
    fmin=50.0,
    fmax=500.0,
    pre_emphasis=0.97,
    smooth_energy=True,
    sg_window=11,
    sg_poly=2,
):
    hop_length = max(1, int(sr * hop_ms / 1000.0))
    frame_length = max(1, int(sr * frame_ms / 1000.0))
    f0_frame_length = max(1, int(sr * f0_frame_ms / 1000.0))

    x = np.asarray(x, dtype=np.float32)
    x = _pre_emphasis(x, pre_emphasis)

    f0, voiced_flag, _ = librosa.pyin(
        x,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=f0_frame_length,
        hop_length=hop_length,
    )
    if voiced_flag is None:
        voiced_flag = np.zeros_like(f0, dtype=bool)
    vuv = voiced_flag.astype(np.float32)
    f0 = np.where(np.isnan(f0), 0.0, f0)

    rms = librosa.feature.rms(
        y=x,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]
    energy = rms.astype(np.float32, copy=False)
    if smooth_energy and energy.size >= sg_window:
        if sg_window % 2 == 0:
            sg_window += 1
        energy = savgol_filter(energy, sg_window, sg_poly, mode="interp")

    T = max(len(f0), len(energy))
    f0 = librosa.util.fix_length(f0, size=T)
    vuv = librosa.util.fix_length(vuv, size=T)
    energy = librosa.util.fix_length(energy, size=T)

    log_f0 = np.log(np.clip(f0, EPS, None))
    log_energy = np.log(np.clip(energy, EPS, None))

    f0_norm = _robust_normalize(log_f0, mask=vuv > 0.5)
    f0_norm = np.where(vuv > 0.5, f0_norm, 0.0)
    energy_norm = _robust_normalize(log_energy, mask=None)

    d_f0 = librosa.feature.delta(f0_norm.reshape(1, -1), width=9, order=1, axis=1)[0]
    d_energy = librosa.feature.delta(energy_norm.reshape(1, -1), width=9, order=1, axis=1)[0]

    feats = np.stack([f0_norm, vuv, energy_norm, d_f0, d_energy], axis=1)
    return feats
