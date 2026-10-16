"""Catalog, download, and feature extraction for the expanded public CWRU set."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from scipy.io import loadmat


SAMPLE_RATE = 12_000
SEGMENT_SAMPLES = 2_048
FEATURE_DIM = 32
MAX_SEGMENTS_PER_RECORDING = 24
DOWNLOAD_BUDGET_BYTES = int(1.5 * 1024**3)
DOWNLOAD_BUDGET_SECONDS = 14 * 60
NOMINAL_RPM = (1797.0, 1772.0, 1750.0, 1730.0)
OFFICIAL_BASE = "https://engineering.case.edu/sites/default/files/{}.mat"
MIRROR_BASE = "https://raw.githubusercontent.com/AiChiXiaoXiongBingGan/CWRU-dataset/main/op_{}/{}/{}.mat"


@dataclass(frozen=True)
class RecordingSpec:
    file_id: int
    position: str
    fault_type: str
    severity_in: float
    load_hp: int
    rpm_nominal: float
    fault_location: str
    official_page: str

    @property
    def filename(self) -> str:
        size = "normal" if self.severity_in == 0 else f"{int(self.severity_in * 1000):03d}"
        return f"{self.position.lower()}_12k_{self.fault_type}_{size}_load{self.load_hp}_{self.file_id}.mat"

    @property
    def path(self) -> str:
        return self.filename

    @property
    def official_url(self) -> str:
        return OFFICIAL_BASE.format(self.file_id)

    def mirror_url(self) -> str | None:
        if self.position != "DE" or self.severity_in not in (0, .007, .014, .021):
            return None
        mirror_type = "normal" if self.fault_type == "normal" else f"{self.fault_type}_{int(self.severity_in * 1000):02d}"
        return MIRROR_BASE.format(self.load_hp, mirror_type, self.file_id)


def _make_catalog() -> list[RecordingSpec]:
    """Use official 12 kHz DE and FE tables; unavailable severities are omitted."""
    rows: list[RecordingSpec] = []
    pages = {
        "DE": "https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data",
        "FE": "https://engineering.case.edu/bearingdatacenter/12k-fan-end-bearing-fault-data",
    }

    def add(position: str, fault: str, severity: float, load: int, file_id: int,
            fault_location: str = "bearing") -> None:
        rows.append(RecordingSpec(file_id, position, fault, severity, load,
                                  NOMINAL_RPM[load], fault_location, pages[position]))

    # Normal files include the common DE/FE channels and are shared across positions.
    for load, file_id in enumerate((97, 98, 99, 100)):
        add("DE", "normal", 0.0, load, file_id, "healthy")

    de_ids = {
        "ball": {0.007: (118, 119, 120, 121), 0.014: (185, 186, 187, 188),
                 0.021: (222, 223, 224, 225), 0.028: (3005, 3006, 3007, 3008)},
        "inner": {0.007: (105, 106, 107, 108), 0.014: (169, 170, 171, 172),
                  0.021: (209, 210, 211, 212)},
        "outer": {0.007: (130, 131, 132, 133), 0.014: (197, 198, 199, 200),
                  0.021: (234, 235, 236, 237)},
    }
    for fault, sizes in de_ids.items():
        for severity, ids in sizes.items():
            for load, file_id in enumerate(ids):
                add("DE", fault, severity, load, file_id)

    fe_ids = {
        "ball": {0.007: (282, 283, 284, 285), 0.014: (286, 287, 288, 289),
                 0.021: (290, 291, 292, 293)},
        "inner": {0.007: (278, 279, 280, 281), 0.014: (274, 275, 276, 277),
                  0.021: (270, 271, 272, 273)},
        "outer": {0.007: (294, 295, 296, 297), 0.014: (310, 309, 311, 312),
                  0.021: (315, 316, 317, 318)},
    }
    for fault, sizes in fe_ids.items():
        for severity, ids in sizes.items():
            for load, file_id in enumerate(ids):
                location = "centered" if fault != "outer" else ("centered" if severity != .021 or load == 0 else "orthogonal")
                add("FE", fault, severity, load, file_id, location)
    return rows


CATALOG = _make_catalog()


def _log(log, message: str) -> None:
    if log is not None:
        log(message)


def download_recordings(data_dir: str | Path, log=None,
                        byte_budget: int = DOWNLOAD_BUDGET_BYTES,
                        time_budget: int = DOWNLOAD_BUDGET_SECONDS) -> dict:
    """Download missing catalog files, official first then the verified GitHub mirror."""
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    downloaded, skipped, failed = [], [], []
    bytes_used = 0
    _log(log, f"download budget: {byte_budget} bytes / {time_budget}s; {len(CATALOG)} catalog recordings")
    for spec in CATALOG:
        destination = root / spec.filename
        if destination.is_file() and destination.stat().st_size > 0:
            skipped.append({"file": spec.filename, "reason": "already present", "bytes": destination.stat().st_size})
            continue
        if time.monotonic() - start >= time_budget:
            failed.append({"file": spec.filename, "reason": "15-minute download budget reached"})
            _log(log, f"DOWNLOAD SKIP {spec.filename}: time budget reached")
            continue
        candidates = [("official", spec.official_url)]
        if spec.mirror_url():
            candidates.append(("github-mirror", spec.mirror_url()))
        errors = []
        done = False
        for source, url in candidates:
            if bytes_used >= byte_budget:
                errors.append("aggregate download byte budget reached")
                break
            remaining_seconds = time_budget - (time.monotonic() - start)
            if remaining_seconds <= 0:
                errors.append("download time budget reached")
                break
            part = destination.with_suffix(destination.suffix + ".part")
            size = 0
            try:
                request = Request(url, headers={"User-Agent": "quann-3d-maintenance/1.0"})
                with urlopen(request, timeout=max(1, min(25, remaining_seconds))) as response, part.open("wb") as stream:
                    declared = response.headers.get("Content-Length")
                    if declared and int(declared) > byte_budget - bytes_used:
                        raise RuntimeError(f"declared {declared} bytes exceeds remaining budget")
                    while True:
                        if time.monotonic() - start >= time_budget:
                            raise TimeoutError("download time budget reached")
                        remaining_bytes = byte_budget - bytes_used
                        if remaining_bytes <= 0:
                            raise RuntimeError("aggregate download byte budget reached")
                        chunk = response.read(min(128 * 1024, remaining_bytes))
                        if not chunk:
                            break
                        size += len(chunk)
                        bytes_used += len(chunk)
                        stream.write(chunk)
                if size < 1024:
                    raise ValueError(f"response too small ({size} bytes), likely an HTML error page")
                part.replace(destination)
                downloaded.append({"file": spec.filename, "source": source, "url": url, "bytes": size})
                _log(log, f"DOWNLOADED {spec.filename} via {source}: {size} bytes")
                done = True
                break
            except (HTTPError, URLError, TimeoutError, RuntimeError, ValueError, OSError) as exc:
                try:
                    part.unlink(missing_ok=True)
                except OSError:
                    pass
                errors.append(f"{source}: {type(exc).__name__}: {exc}")
                _log(log, f"DOWNLOAD FAILURE {spec.filename} {source}: {type(exc).__name__}: {exc}")
        if not done:
            entry = {"file": spec.filename, "reason": "; ".join(errors)}
            failed.append(entry)
            _log(log, f"DOWNLOAD FAILED {spec.filename}: {entry['reason']}")
    summary = {"requested": len(CATALOG), "downloaded": downloaded, "already_present": skipped,
               "failed": failed, "bytes_downloaded_this_run": bytes_used,
               "elapsed_seconds": round(time.monotonic() - start, 2),
               "budget_bytes": byte_budget, "budget_seconds": time_budget}
    _log(log, f"download summary: downloaded={len(downloaded)} existing={len(skipped)} failed={len(failed)} bytes={bytes_used} elapsed={summary['elapsed_seconds']}s")
    return summary


def _mat_record(path: Path, spec: RecordingSpec, max_segments: int) -> dict:
    mat = loadmat(path)
    channels: dict[str, np.ndarray] = {}
    rpm = None
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        upper = key.upper()
        for channel in ("DE", "FE", "BA", "BE"):
            if upper.endswith(f"_{channel}_TIME") or upper.endswith(f"_{channel}_TIME".replace("_", "")):
                channels[channel] = np.asarray(value, dtype=np.float64).reshape(-1)
        if upper.endswith("RPM"):
            raw_rpm = np.asarray(value).reshape(-1)
            if raw_rpm.size and np.isfinite(raw_rpm[0]) and float(raw_rpm[0]) > 0:
                rpm = float(raw_rpm[0])
    if not channels:
        raise ValueError(f"no DE/FE/BA/BE vibration channels in {path.name}")
    rpm_source = "mat_rpm" if rpm is not None else "catalog_nominal"
    if rpm is None:
        rpm = spec.rpm_nominal
    available = [name for name in ("DE", "FE", "BA", "BE") if name in channels]
    length = min(len(channels[name]) for name in available)
    if length < SEGMENT_SAMPLES:
        raise ValueError(f"{path.name} shorter than one {SEGMENT_SAMPLES}-sample segment")
    n_segments = min(max_segments, max(1, length // SEGMENT_SAMPLES))
    starts = np.linspace(0, length - SEGMENT_SAMPLES, n_segments, dtype=np.int64)
    features = np.zeros((n_segments, FEATURE_DIM), dtype=np.float32)
    for row, start in enumerate(starts):
        for channel_index, name in enumerate(("DE", "FE")):
            if name not in channels:
                continue
            signal = channels[name][start:start + SEGMENT_SAMPLES]
            centered = signal - float(np.mean(signal))
            rms = float(np.sqrt(np.mean(np.square(centered))))
            crest = float(np.max(np.abs(centered)) / max(rms, 1e-12))
            tapered = centered * np.hanning(len(centered))
            spectrum = np.abs(np.fft.rfft(tapered)) ** 2
            frequencies = np.fft.rfftfreq(len(tapered), d=1.0 / SAMPLE_RATE)
            total = float(spectrum[1:].sum()) + 1e-12
            resolution = SAMPLE_RATE / len(tapered)
            shaft = rpm / 60.0
            ratio = .3125 / 1.537
            ftf = .5 * shaft * (1 - ratio)
            bpfo = .5 * 9 * shaft * (1 - ratio)
            bsf = .5 * (1.537 / .3125) * shaft * (1 - ratio**2)
            bpfi = .5 * 9 * shaft * (1 + ratio)
            centers = (shaft, 2*shaft, 3*shaft, ftf, bpfo, 2*bpfo, 3*bpfo,
                       bsf, 2*bsf, bpfi, 2*bpfi, 3*bpfi, .5*bpfo)
            ratios = [float(spectrum[np.abs(frequencies - center) <= max(2*resolution, .05*center)].sum() / total)
                      for center in centers]
            offset = channel_index * 15
            features[row, offset:offset + 15] = np.asarray([rms, crest, *ratios], dtype=np.float32)
        features[row, 30:] = (rpm / 1800.0, spec.load_hp / 3.0)
    return {
        "features": features,
        "labels": np.full(n_segments, int(spec.fault_type != "normal"), dtype=np.int64),
        "recording_id": spec.filename,
        "source": str(path.resolve()),
        "fault_type": spec.fault_type,
        "severity_in": spec.severity_in,
        "position": spec.position,
        "fault_location": spec.fault_location,
        "load_hp": spec.load_hp,
        "rpm": rpm,
        "rpm_source": rpm_source,
        "channels_available": available,
        "segments": n_segments,
        "source_samples": int(length),
    }


def load_expanded_records(data_dir: str | Path, log=None,
                          max_segments: int = MAX_SEGMENTS_PER_RECORDING) -> list[dict]:
    """Load catalog recordings; failures are logged with their file and reason."""
    root = Path(data_dir)
    records = []
    for spec in CATALOG:
        path = root / spec.filename
        if not path.is_file():
            continue
        try:
            record = _mat_record(path, spec, max_segments)
            records.append(record)
            if record["rpm_source"] != "mat_rpm":
                _log(log, f"RPM FALLBACK {spec.filename}: .mat has no RPM field; catalog nominal={record['rpm']}")
            _log(log, f"LOADED {spec.filename}: {record['segments']} segments, channels={record['channels_available']}, rpm={record['rpm']} ({record['rpm_source']})")
        except Exception as exc:
            _log(log, f"RECORD FAILURE {spec.filename}: {type(exc).__name__}: {exc}")
    if not records:
        raise FileNotFoundError(f"no readable CWRU expansion recordings under {root}")
    return records


def catalog_summary() -> dict:
    """Compact inventory used in result JSON and tests."""
    counts = {}
    for spec in CATALOG:
        key = f"{spec.position}/{spec.fault_type}/{spec.severity_in:.3f}in"
        counts[key] = counts.get(key, 0) + 1
    return {"n_catalog_recordings": len(CATALOG), "available_severity_in": sorted({s.severity_in for s in CATALOG if s.severity_in}),
            "official_sources": sorted({s.official_page for s in CATALOG}), "recordings_by_position_type_size": counts,
            "not_available_in_official_12k_tables": ["ball 0.035 inch", "0.028-inch inner/outer race"]}
