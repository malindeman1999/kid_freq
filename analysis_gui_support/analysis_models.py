from __future__ import annotations

import getpass
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np


@dataclass
class VNAScan:
    filename: str
    loaded_at: str
    freq: np.ndarray
    s21_complex_raw: np.ndarray
    file_timestamp: str = ""
    plot_group: Optional[int] = None
    s21_phase_deg_unwrapped: Optional[np.ndarray] = None
    source_dir: str = ""
    baseline_filter: Dict[str, object] = field(default_factory=dict)
    candidate_resonators: Dict[str, object] = field(default_factory=dict)
    processing_history: List[Dict[str, object]] = field(default_factory=list)

    def amplitude(self) -> np.ndarray:
        return np.abs(self.s21_complex_raw)

    def phase_deg_wrapped_raw(self) -> np.ndarray:
        return np.degrees(np.angle(self.s21_complex_raw))

    def has_dewrapped_phase(self) -> bool:
        if self.s21_phase_deg_unwrapped is None:
            return False
        phase = np.asarray(self.s21_phase_deg_unwrapped, dtype=float)
        return phase.shape == self.freq.shape

    def phase_deg_unwrapped(self) -> np.ndarray:
        if not self.has_dewrapped_phase():
            raise ValueError("Dewrapped phase is not attached for this scan.")
        return np.asarray(self.s21_phase_deg_unwrapped, dtype=float)

    def complex_s21(self) -> np.ndarray:
        return np.asarray(self.s21_complex_raw, dtype=np.complex128)


@dataclass
class Dataset:
    source_file: str
    vna_scans: List[VNAScan] = field(default_factory=list)
    selected_scan_keys: List[str] = field(default_factory=list)
    last_saved_at: str = ""
    dataset_name: str = ""
    created_at: str = ""
    res_neighbor_initial_date: str = ""
    transcript: List[Dict[str, str]] = field(default_factory=list)
    processing_history: List[Dict[str, object]] = field(default_factory=list)


def _complex_from_polar(amplitude: np.ndarray, phase_deg: np.ndarray) -> np.ndarray:
    amp = np.asarray(amplitude, dtype=float)
    phase = np.asarray(phase_deg, dtype=float)
    if amp.shape != phase.shape:
        raise ValueError("Amplitude and phase arrays must have the same shape.")
    return np.asarray(amp * np.exp(1j * np.radians(phase)), dtype=np.complex128)


def _polar_from_complex(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(values, dtype=np.complex128)
    return np.abs(z), np.degrees(np.unwrap(np.angle(z)))


def _read_polar_series(
    payload: Dict[str, object],
    *,
    amplitude_key: str,
    phase_key: str,
    complex_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    amp_raw = payload.get(amplitude_key)
    phase_raw = payload.get(phase_key)
    amp = np.asarray(amp_raw, dtype=float) if amp_raw is not None else np.asarray([], dtype=float)
    phase = np.asarray(phase_raw, dtype=float) if phase_raw is not None else np.asarray([], dtype=float)
    if amp.size and phase.size and amp.shape == phase.shape:
        return amp, phase
    if complex_key is not None:
        z_raw = payload.get(complex_key)
        z = np.asarray(z_raw, dtype=np.complex128) if z_raw is not None else np.asarray([], dtype=np.complex128)
        if z.size:
            return _polar_from_complex(z)
    return np.asarray([], dtype=float), np.asarray([], dtype=float)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _current_user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def _make_event(action: str, details: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    return {
        "timestamp": _now_iso(),
        "user": _current_user(),
        "action": action,
        "details": details or {},
    }
