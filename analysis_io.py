from __future__ import annotations

import json
import pickle
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from analysis_models import Dataset, VNAScan, _make_event


APP_STATE_FILE = Path("analysis_gui_state.json")
DATASETS_DIR = Path("data_sets")
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
        with dataset_path.open("rb") as f:
            data = pickle.load(f)
        if not isinstance(data, Dataset):
            raise TypeError(f"File does not contain a Dataset object: {dataset_path}")
        return _normalize_dataset(data, dataset_path)
    return Dataset(source_file=str(dataset_path.resolve()))


def _normalize_dataset(dataset: Dataset, dataset_path: Path) -> Dataset:
    # Development mode: assume current schema and avoid compatibility shims.
    dataset.source_file = str(dataset_path.resolve())

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
            dataset.dataset_name = match.group("name").replace("_", " ")

    return dataset


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
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2D array with freq/real/imag data, got shape {arr.shape}"
        )

    # Supported formats:
    # - row format:    (3, N) rows = [freq, real, imag]
    # - column format: (N, 3) cols = [freq, real, imag]
    # - complex row:   (2, N) rows = [freq, complex_s21]
    # - complex col:   (N, 2) cols = [freq, complex_s21]
    if arr.shape[0] >= 3 and arr.shape[1] >= 3:
        freq = arr[0, :]
        s21_real = arr[1, :]
        s21_imag = arr[2, :]
    elif arr.shape[1] >= 3 and arr.shape[0] >= 3:
        freq = arr[:, 0]
        s21_real = arr[:, 1]
        s21_imag = arr[:, 2]
    elif arr.shape[0] == 2:
        freq = np.real(arr[0, :])
        s21_complex = arr[1, :]
        s21_real = np.real(s21_complex)
        s21_imag = np.imag(s21_complex)
    elif arr.shape[1] == 2:
        freq = np.real(arr[:, 0])
        s21_complex = arr[:, 1]
        s21_real = np.real(s21_complex)
        s21_imag = np.imag(s21_complex)
    else:
        raise ValueError(
            "Expected shape (3, N) rows [freq, real, imag], (N, 3) columns [freq, real, imag], "
            "(2, N) rows [freq, complex_s21], or (N, 2) columns [freq, complex_s21], "
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
    s21_amp = np.abs(s21_complex)
    s21_phase_deg_unwrapped = np.degrees(np.unwrap(np.angle(s21_complex)))
    loaded_at = datetime.now().isoformat(timespec="seconds")
    scan = VNAScan(
        filename=str(path.resolve()),
        source_dir=str(path.resolve().parent),
        loaded_at=loaded_at,
        freq=freq,
        s21_amp=s21_amp,
        s21_phase_deg_unwrapped=s21_phase_deg_unwrapped,
    )
    scan.processing_history.append(
        _make_event(
            "load_vna_npy",
            {
                "filename": scan.filename,
                "source_dir": scan.source_dir,
                "shape": list(arr.shape),
            },
        )
    )
    return scan
