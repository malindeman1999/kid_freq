from __future__ import annotations

import json
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from . import analysis_models as _analysis_models
from .analysis_models import Dataset, VNAScan, _complex_from_polar, _make_event


PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_STATE_FILE = Path(__file__).resolve().parent / "analysis_gui_state.json"
DATASETS_DIR = PROJECT_ROOT / "data_sets"
DEFAULT_DATASET_FILE = DATASETS_DIR / "analysis_dataset.pkl"


def _read_app_state() -> Path:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    if APP_STATE_FILE.exists():
        try:
            payload = json.loads(APP_STATE_FILE.read_text(encoding="utf-8"))
            dataset_path = payload.get("active_dataset_path", "")
            if dataset_path:
                return Path(dataset_path)
        except Exception:
            pass
    return DEFAULT_DATASET_FILE


def _write_app_state(dataset_path: Path) -> None:
    payload = {"active_dataset_path": str(dataset_path.resolve())}
    APP_STATE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_dataset(dataset_path: Path) -> Dataset:
    if dataset_path.exists():
        sys.modules.setdefault("analysis_models", _analysis_models)
        with dataset_path.open("rb") as f:
            data = pickle.load(f)
        if not isinstance(data, Dataset):
            raise TypeError(f"File does not contain a Dataset object: {dataset_path}")
        return _normalize_dataset(data, dataset_path)
    return Dataset(source_file=str(dataset_path.resolve()))


def _normalize_dataset(dataset: Dataset, dataset_path: Path) -> Dataset:
    # Development mode: assume current schema and avoid compatibility shims.
    dataset.source_file = str(dataset_path.resolve())
    if not hasattr(dataset, "res_neighbor_initial_date"):
        dataset.res_neighbor_initial_date = ""
    if not hasattr(dataset, "saved_scan_selections") or not isinstance(dataset.saved_scan_selections, dict):
        dataset.saved_scan_selections = {}
    for scan in dataset.vna_scans:
        if not hasattr(scan, "plot_group"):
            scan.plot_group = None
        if not hasattr(scan, "file_timestamp"):
            scan.file_timestamp = ""
    _backfill_missing_vna_file_timestamps(dataset)

    # Backfill metadata from filename if possible: YYYYMMDD_HHMMSS_name.pkl
    stem = dataset_path.stem
    match = re.match(r"^(?P<prefix>\d{8}_\d{6})_(?P<name>.+)$", stem)
    if match:
        if not dataset.created_at:
            try:
                dt = datetime.strptime(match.group("prefix"), "%Y%m%d_%H%M%S")
                dataset.created_at = dt.isoformat(timespec="seconds")
            except Exception:
                pass
        if not dataset.dataset_name:
            # Preserve the exact prefix used in the dataset path.
            dataset.dataset_name = match.group("name")

    return dataset


def _backfill_missing_vna_file_timestamps(dataset: Dataset) -> int:
    updated = 0
    for scan in dataset.vna_scans:
        if getattr(scan, "file_timestamp", ""):
            continue
        try:
            path = Path(str(scan.filename)).resolve()
        except Exception:
            continue
        if not path.exists():
            continue
        try:
            scan.file_timestamp = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
        except Exception:
            continue
        scan.processing_history.append(
            _make_event(
                "backfill_vna_file_timestamp",
                {
                    "filename": scan.filename,
                    "file_timestamp": scan.file_timestamp,
                },
            )
        )
        updated += 1
    return updated


def _safe_name(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._ -]+", "", name).strip().replace(" ", "_")
    slug = re.sub(r"_+", "_", slug)
    return slug


def _format_created_prefix(created_at: str) -> str:
    dt = datetime.fromisoformat(created_at)
    return dt.strftime("%Y%m%d_%H%M%S")


def _dataset_dir_name(dataset: Dataset) -> str:
    if not dataset.dataset_name or not dataset.created_at:
        return ""
    return f"{_format_created_prefix(dataset.created_at)}_{dataset.dataset_name}"


def _dataset_dir(dataset: Dataset) -> Path:
    dir_name = _dataset_dir_name(dataset)
    if not dir_name:
        return DATASETS_DIR
    return DATASETS_DIR / dir_name


def _dataset_pickle_path(dataset: Dataset) -> Path:
    dir_name = _dataset_dir_name(dataset)
    if not dir_name:
        return DEFAULT_DATASET_FILE
    return _dataset_dir(dataset) / f"{dir_name}.pkl"


def _save_dataset(dataset: Dataset, dataset_path: Path) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.last_saved_at = datetime.now().isoformat(timespec="seconds")
    dataset.source_file = str(dataset_path.resolve())
    with dataset_path.open("wb") as f:
        pickle.dump(dataset, f)


def _load_vna_npy(path: Path) -> VNAScan:
    arr = np.load(path)
    arr = np.asarray(arr)
    if arr.ndim not in {2, 3}:
        raise ValueError(
            "Expected 2D/3D array with freq/real/imag data, "
            f"got shape {arr.shape}"
        )

    # Supported formats:
    # - row format:    (3, N) rows = [freq, real, imag]
    # - column format: (N, 3) cols = [freq, real, imag]
    # - complex row:   (2, N) rows = [freq, complex_s21]
    # - complex col:   (N, 2) cols = [freq, complex_s21]
    # - paged rows:    (M, 3, N) pages with rows [freq, real, imag], flattened to one scan
    if arr.ndim == 2 and arr.shape[1] == 3:
        freq = arr[:, 0]
        s21_real = arr[:, 1]
        s21_imag = arr[:, 2]
    elif arr.ndim == 2 and arr.shape[0] == 3:
        freq = arr[0, :]
        s21_real = arr[1, :]
        s21_imag = arr[2, :]
    elif arr.ndim == 2 and arr.shape[1] == 2:
        freq = np.real(arr[:, 0])
        s21_complex = arr[:, 1]
        s21_real = np.real(s21_complex)
        s21_imag = np.imag(s21_complex)
    elif arr.ndim == 2 and arr.shape[0] == 2:
        freq = np.real(arr[0, :])
        s21_complex = arr[1, :]
        s21_real = np.real(s21_complex)
        s21_imag = np.imag(s21_complex)
    elif arr.ndim == 3 and arr.shape[1] == 3:
        # Flatten M pages of length N into one trace of length M*N.
        freq = np.reshape(np.asarray(arr[:, 0, :], dtype=float), -1)
        s21_real = np.reshape(np.asarray(arr[:, 1, :], dtype=float), -1)
        s21_imag = np.reshape(np.asarray(arr[:, 2, :], dtype=float), -1)
    else:
        raise ValueError(
            "Expected shape (3, N) rows [freq, real, imag], (N, 3) columns [freq, real, imag], "
            "(2, N) rows [freq, complex_s21], (N, 2) columns [freq, complex_s21], or "
            "(M, 3, N) paged rows [freq, real, imag], "
            f"got {arr.shape}"
        )

    freq = np.asarray(freq, dtype=float)
    s21_real = np.asarray(s21_real, dtype=float)
    s21_imag = np.asarray(s21_imag, dtype=float)
    if not (freq.size == s21_real.size == s21_imag.size):
        raise ValueError("Frequency, real, and imaginary arrays must have the same length.")
    if freq.size < 3:
        raise ValueError("VNA data must contain at least 3 points.")

    s21_complex = s21_real + 1j * s21_imag
    loaded_at = datetime.now().isoformat(timespec="seconds")
    file_timestamp = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    scan = VNAScan(
        filename=str(path.resolve()),
        source_dir=str(path.resolve().parent),
        loaded_at=loaded_at,
        file_timestamp=file_timestamp,
        freq=freq,
        s21_complex_raw=s21_complex,
        s21_phase_deg_unwrapped=None,
    )
    scan.processing_history.append(
        _make_event(
            "load_vna_npy",
            {
                "filename": scan.filename,
                "source_dir": scan.source_dir,
                "shape": list(arr.shape),
                "flattened_paged_input": bool(arr.ndim == 3 and arr.shape[1] == 3),
                "points_stored": int(freq.size),
                "file_timestamp": scan.file_timestamp,
            },
        )
    )
    return scan


def _load_vna_npy_mhz_db_deg(path: Path) -> VNAScan:
    arr = np.load(path)
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2D array with [frequency_MHz, amplitude_dB, phase_deg], got shape {arr.shape}"
        )

    # Explicit loader for legacy/problem format:
    # - row format:    (3, N) rows = [freq_MHz, amp_dB, phase_deg]
    # - column format: (N, 3) cols = [freq_MHz, amp_dB, phase_deg]
    if arr.shape[1] == 3:
        freq_mhz = arr[:, 0]
        amp_db = arr[:, 1]
        phase_deg = arr[:, 2]
    elif arr.shape[0] == 3:
        freq_mhz = arr[0, :]
        amp_db = arr[1, :]
        phase_deg = arr[2, :]
    else:
        raise ValueError(
            "Expected shape (3, N) rows [frequency_MHz, amplitude_dB, phase_deg] or "
            f"(N, 3) columns [frequency_MHz, amplitude_dB, phase_deg], got {arr.shape}"
        )

    freq_hz = np.asarray(freq_mhz, dtype=float) * 1.0e6
    amp_db = np.asarray(amp_db, dtype=float)
    phase_deg = np.asarray(phase_deg, dtype=float)
    if not (freq_hz.size == amp_db.size == phase_deg.size):
        raise ValueError("Frequency, amplitude, and phase arrays must have the same length.")
    if freq_hz.size < 3:
        raise ValueError("VNA data must contain at least 3 points.")

    amp_linear = np.power(10.0, amp_db / 20.0)
    s21_complex = _complex_from_polar(amp_linear, phase_deg)

    loaded_at = datetime.now().isoformat(timespec="seconds")
    file_timestamp = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    scan = VNAScan(
        filename=str(path.resolve()),
        source_dir=str(path.resolve().parent),
        loaded_at=loaded_at,
        file_timestamp=file_timestamp,
        freq=freq_hz,
        s21_complex_raw=s21_complex,
        s21_phase_deg_unwrapped=None,
    )
    scan.processing_history.append(
        _make_event(
            "load_vna_npy_mhz_db_deg",
            {
                "filename": scan.filename,
                "source_dir": scan.source_dir,
                "shape": list(arr.shape),
                "frequency_units_in": "MHz",
                "frequency_units_stored": "Hz",
                "amplitude_units_in": "dB",
                "phase_units_in": "deg",
                "file_timestamp": scan.file_timestamp,
            },
        )
    )
    return scan


def _load_vna_text_db_phase0(path: Path) -> VNAScan:
    try:
        arr = np.loadtxt(path, dtype=float)
    except Exception as exc:
        raise ValueError(f"Could not parse text VNA data: {exc}") from exc

    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            "Expected 2-column text data [frequency_MHz, amplitude_dB], "
            f"got shape {arr.shape}"
        )

    freq_hz = np.asarray(arr[:, 0], dtype=float) * 1.0e6
    amp_db = np.asarray(arr[:, 1], dtype=float)
    if freq_hz.size != amp_db.size:
        raise ValueError("Frequency and amplitude arrays must have the same length.")
    if freq_hz.size < 3:
        raise ValueError("VNA data must contain at least 3 points.")

    amp_linear = np.power(10.0, amp_db / 20.0)
    phase_deg = np.zeros_like(amp_linear)
    s21_complex = _complex_from_polar(amp_linear, phase_deg)

    loaded_at = datetime.now().isoformat(timespec="seconds")
    file_timestamp = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    scan = VNAScan(
        filename=str(path.resolve()),
        source_dir=str(path.resolve().parent),
        loaded_at=loaded_at,
        file_timestamp=file_timestamp,
        freq=freq_hz,
        s21_complex_raw=s21_complex,
        s21_phase_deg_unwrapped=None,
    )
    scan.processing_history.append(
        _make_event(
            "load_vna_text_db_phase0_assumed",
            {
                "filename": scan.filename,
                "source_dir": scan.source_dir,
                "shape": list(arr.shape),
                "frequency_units_in": "MHz",
                "frequency_units_stored": "Hz",
                "amplitude_units_in": "dB",
                "phase_assumed_deg": 0.0,
                "assumed_file_type": "2-column text [frequency_MHz, amplitude_dB]",
                "file_timestamp": scan.file_timestamp,
            },
        )
    )
    return scan


def _load_vna_text_complex_hz(path: Path) -> VNAScan:
    try:
        arr = np.loadtxt(path, dtype=float)
    except Exception as exc:
        raise ValueError(f"Could not parse text VNA data: {exc}") from exc

    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            "Expected 3-column text data [frequency_Hz, real_S21, imag_S21], "
            f"got shape {arr.shape}"
        )

    freq_hz = np.asarray(arr[:, 0], dtype=float)
    s21_real = np.asarray(arr[:, 1], dtype=float)
    s21_imag = np.asarray(arr[:, 2], dtype=float)
    if not (freq_hz.size == s21_real.size == s21_imag.size):
        raise ValueError("Frequency, real, and imaginary arrays must have the same length.")
    if freq_hz.size < 3:
        raise ValueError("VNA data must contain at least 3 points.")

    s21_complex = s21_real + 1j * s21_imag

    loaded_at = datetime.now().isoformat(timespec="seconds")
    file_timestamp = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    scan = VNAScan(
        filename=str(path.resolve()),
        source_dir=str(path.resolve().parent),
        loaded_at=loaded_at,
        file_timestamp=file_timestamp,
        freq=freq_hz,
        s21_complex_raw=s21_complex,
        s21_phase_deg_unwrapped=None,
    )
    scan.processing_history.append(
        _make_event(
            "load_vna_text_complex_hz",
            {
                "filename": scan.filename,
                "source_dir": scan.source_dir,
                "shape": list(arr.shape),
                "file_timestamp": scan.file_timestamp,
            },
        )
    )
    return scan


def _touchstone_sparam_to_complex(value_a: float, value_b: float, data_format: str) -> complex:
    fmt = str(data_format).strip().upper()
    if fmt == "RI":
        return complex(float(value_a), float(value_b))
    if fmt == "MA":
        mag = float(value_a)
        phase_rad = np.radians(float(value_b))
        return complex(mag * np.cos(phase_rad), mag * np.sin(phase_rad))
    if fmt == "DB":
        mag = float(np.power(10.0, float(value_a) / 20.0))
        phase_rad = np.radians(float(value_b))
        return complex(mag * np.cos(phase_rad), mag * np.sin(phase_rad))
    raise ValueError(f"Unsupported Touchstone data format: {data_format}")


def _load_vna_touchstone_s2p(path: Path) -> VNAScan:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception as exc:
        raise ValueError(f"Could not read Touchstone file: {exc}") from exc

    freq_unit = "HZ"
    data_format = "MA"
    option_line = None
    numeric_tokens: list[float] = []
    for raw_line in lines:
        line = str(raw_line).strip()
        if not line:
            continue
        if line.startswith("!"):
            continue
        if line.startswith("#"):
            option_line = line
            opts = [tok.strip().upper() for tok in line[1:].split() if tok.strip()]
            if opts:
                for tok in opts:
                    if tok in {"HZ", "KHZ", "MHZ", "GHZ"}:
                        freq_unit = tok
                        break
                for tok in opts:
                    if tok in {"RI", "MA", "DB"}:
                        data_format = tok
                        break
            continue
        if "!" in line:
            line = line.split("!", 1)[0].strip()
        if not line:
            continue
        try:
            numeric_tokens.extend(float(tok) for tok in line.split())
        except Exception as exc:
            raise ValueError(f"Could not parse numeric Touchstone data line: {raw_line}") from exc

    if not numeric_tokens:
        raise ValueError("Touchstone file contained no numeric data.")

    # S2P row width: frequency + 4 S-parameters * 2 values each = 9 numbers per point.
    if len(numeric_tokens) % 9 != 0:
        raise ValueError(
            f"Touchstone numeric token count {len(numeric_tokens)} is not divisible by 9 for S2P data."
        )
    arr = np.asarray(numeric_tokens, dtype=float).reshape(-1, 9)
    if arr.shape[0] < 3:
        raise ValueError("VNA data must contain at least 3 points.")

    freq = np.asarray(arr[:, 0], dtype=float)
    scale_by_unit = {
        "HZ": 1.0,
        "KHZ": 1.0e3,
        "MHZ": 1.0e6,
        "GHZ": 1.0e9,
    }
    if freq_unit not in scale_by_unit:
        raise ValueError(f"Unsupported Touchstone frequency unit: {freq_unit}")
    freq_hz = freq * float(scale_by_unit[freq_unit])

    # Touchstone 1.0 default order: S11, S21, S12, S22.
    s21_a = np.asarray(arr[:, 3], dtype=float)
    s21_b = np.asarray(arr[:, 4], dtype=float)
    s21_complex = np.asarray(
        [_touchstone_sparam_to_complex(a, b, data_format) for a, b in zip(s21_a, s21_b)],
        dtype=np.complex128,
    )

    loaded_at = datetime.now().isoformat(timespec="seconds")
    file_timestamp = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    scan = VNAScan(
        filename=str(path.resolve()),
        source_dir=str(path.resolve().parent),
        loaded_at=loaded_at,
        file_timestamp=file_timestamp,
        freq=freq_hz,
        s21_complex_raw=s21_complex,
        s21_phase_deg_unwrapped=None,
    )
    scan.processing_history.append(
        _make_event(
            "load_vna_touchstone_s2p",
            {
                "filename": scan.filename,
                "source_dir": scan.source_dir,
                "points": int(freq_hz.size),
                "frequency_units_in": freq_unit,
                "frequency_units_stored": "Hz",
                "data_format_in": data_format,
                "option_line": option_line or "",
                "stored_param": "S21",
                "file_timestamp": scan.file_timestamp,
            },
        )
    )
    return scan


def _load_vna_file(path: Path) -> tuple[VNAScan, str | None]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _load_vna_npy(path), None
    if suffix == ".s2p":
        return _load_vna_touchstone_s2p(path), None
    if suffix in {".txt", ".dat", ".csv"}:
        try:
            scan = _load_vna_text_complex_hz(path)
        except ValueError as exc:
            if "Expected 3-column text data" not in str(exc):
                raise
        else:
            return scan, None

        warning = (
            f"{path.name}: assumed 2-column text format [frequency in MHz, amplitude in dB]. "
            "Converted frequency to Hz and set phase to 0 deg."
        )
        return _load_vna_text_db_phase0(path), warning
    raise ValueError(f"Unsupported VNA file type: {path.suffix or '<no extension>'}")


def _try_load_vna_npy_pair(path_a: Path, path_b: Path) -> tuple[VNAScan | None, str | None]:
    """Attempt loading two .npy files as one scan: 1D real freq + 1D complex S21."""
    if path_a.suffix.lower() != ".npy" or path_b.suffix.lower() != ".npy":
        return None, None

    arr_a = np.asarray(np.load(path_a))
    arr_b = np.asarray(np.load(path_b))
    if arr_a.ndim != 1 or arr_b.ndim != 1:
        return None, None

    a_is_complex = bool(np.iscomplexobj(arr_a))
    b_is_complex = bool(np.iscomplexobj(arr_b))
    if a_is_complex == b_is_complex:
        return None, None

    if a_is_complex:
        s21_path, s21_arr = path_a, arr_a
        freq_path, freq_arr = path_b, arr_b
    else:
        s21_path, s21_arr = path_b, arr_b
        freq_path, freq_arr = path_a, arr_a

    if not np.issubdtype(freq_arr.dtype, np.number):
        raise ValueError(f"{freq_path.name} is not numeric 1D frequency data.")

    freq = np.asarray(np.real(freq_arr), dtype=float)
    s21_complex = np.asarray(s21_arr, dtype=np.complex128)
    if freq.size != s21_complex.size:
        raise ValueError(
            "Frequency and complex S21 arrays must have the same length: "
            f"{freq_path.name} has {freq.size}, {s21_path.name} has {s21_complex.size}."
        )
    if freq.size < 3:
        raise ValueError("VNA data must contain at least 3 points.")

    loaded_at = datetime.now().isoformat(timespec="seconds")
    file_timestamp = datetime.fromtimestamp(s21_path.stat().st_mtime).isoformat(timespec="seconds")
    source_dir = s21_path.resolve().parent
    scan = VNAScan(
        filename=str(s21_path.resolve()),
        source_dir=str(source_dir),
        loaded_at=loaded_at,
        file_timestamp=file_timestamp,
        freq=freq,
        s21_complex_raw=s21_complex,
        s21_phase_deg_unwrapped=None,
    )
    scan.processing_history.append(
        _make_event(
            "load_vna_npy_pair_1d_freq_complex",
            {
                "filename": scan.filename,
                "source_dir": scan.source_dir,
                "frequency_file": str(freq_path.resolve()),
                "s21_complex_file": str(s21_path.resolve()),
                "frequency_shape": list(freq_arr.shape),
                "s21_complex_shape": list(s21_arr.shape),
                "file_timestamp": scan.file_timestamp,
            },
        )
    )
    return scan, None
