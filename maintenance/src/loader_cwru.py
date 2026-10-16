"""Read small CWRU .mat recordings into channel-wise 32D vibration features."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat


FEATURE_DIM = 32
SAMPLE_RATE = 12_000
SEGMENT_SAMPLES = 2_048
CHANNELS = ("DE", "FE")


def _mat_channels(path: str | Path) -> tuple[dict[str, np.ndarray], float]:
    path = Path(path)
    mat = loadmat(path)
    channels = {}
    rpm = None
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        key_upper = key.upper()
        if key_upper.endswith("_DE_TIME"):
            channels["DE"] = np.asarray(value, dtype=np.float64).reshape(-1)
        elif key_upper.endswith("_FE_TIME"):
            channels["FE"] = np.asarray(value, dtype=np.float64).reshape(-1)
        elif key_upper.endswith("RPM"):
            rpm = float(np.asarray(value).reshape(-1)[0])
    missing = [channel for channel in CHANNELS if channel not in channels]
    if missing:
        raise ValueError(f"{path} missing required vibration channels: {missing}")
    if rpm is None or not np.isfinite(rpm) or rpm <= 0:
        raise ValueError(f"{path} has no valid RPM field")
    if any(not np.isfinite(signal).all() for signal in channels.values()):
        raise ValueError(f"{path} contains non-finite vibration values")
    return channels, rpm


def _frequency_ratios(rpm: float) -> np.ndarray:
    shaft = rpm / 60.0
    # CWRU 6205 bearing geometry: 9 rolling elements, 0.3125 in ball,
    # 1.537 in pitch diameter, zero nominal contact angle.
    ratio = 0.3125 / 1.537
    ftf = 0.5 * shaft * (1.0 - ratio)
    bpfo = 0.5 * 9 * shaft * (1.0 - ratio)
    bsf = 0.5 * (1.537 / 0.3125) * shaft * (1.0 - ratio**2)
    bpfi = 0.5 * 9 * shaft * (1.0 + ratio)
    return np.asarray([shaft, 2 * shaft, 3 * shaft, ftf, bpfo, 2 * bpfo,
                       3 * bpfo, bsf, 2 * bsf, bpfi, 2 * bpfi, 3 * bpfi,
                       0.5 * bpfo], dtype=np.float64)


def _channel_features(signal: np.ndarray, rpm: float) -> np.ndarray:
    centered = signal - float(np.mean(signal))
    rms = float(np.sqrt(np.mean(np.square(centered))))
    crest = float(np.max(np.abs(centered)) / max(rms, 1e-12))
    tapered = centered * np.hanning(len(centered))
    spectrum = np.abs(np.fft.rfft(tapered)) ** 2
    frequencies = np.fft.rfftfreq(len(tapered), d=1.0 / SAMPLE_RATE)
    total = float(spectrum[1:].sum()) + 1e-12
    resolution = SAMPLE_RATE / len(tapered)
    bearing_frequencies = _frequency_ratios(rpm)
    band_energy = []
    for center in bearing_frequencies:
        half_width = max(2.0 * resolution, 0.05 * center)
        selected = np.abs(frequencies - center) <= half_width
        band_energy.append(float(spectrum[selected].sum() / total))
    return np.asarray([rms, crest, *band_energy], dtype=np.float32)


def extract_recording(path: str | Path, label: int, load_hp: float = 0.0,
                      segment_samples: int = SEGMENT_SAMPLES) -> dict:
    """Convert a raw DE/FE recording into non-overlapping 32D feature points."""
    channels, rpm = _mat_channels(path)
    length = min(len(channels[channel]) for channel in CHANNELS)
    n_segments = length // segment_samples
    if n_segments < 1:
        raise ValueError(f"{path} is shorter than one {segment_samples}-sample segment")
    features = np.empty((n_segments, FEATURE_DIM), dtype=np.float32)
    context = np.asarray([rpm / 1800.0, load_hp / 3.0], dtype=np.float32)
    for row in range(n_segments):
        start = row * segment_samples
        stop = start + segment_samples
        de = _channel_features(channels["DE"][start:stop], rpm)
        fe = _channel_features(channels["FE"][start:stop], rpm)
        features[row] = np.concatenate([de, fe, context])
    return {
        "features": features,
        "labels": np.full(n_segments, int(label), dtype=np.int64),
        "rpm": rpm,
        "load_hp": float(load_hp),
        "channels": list(CHANNELS),
        "source": str(Path(path).resolve()),
        "source_samples": int(length),
        "segments": int(n_segments),
    }


def load_cwru(data_dir: str | Path) -> dict[str, dict]:
    """Load the verified tiny pair: 97.mat normal and 118.mat ball-faulted."""
    root = Path(data_dir).expanduser().resolve()
    normal_path = root / "97.mat"
    fault_path = root / "118.mat"
    if not normal_path.is_file() or not fault_path.is_file():
        raise FileNotFoundError(f"expected 97.mat and 118.mat below {root}")
    return {
        "normal": extract_recording(normal_path, label=0, load_hp=0.0),
        "ball_fault_007": extract_recording(fault_path, label=1, load_hp=0.0),
    }
