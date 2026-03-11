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
    s21_amp: np.ndarray
    s21_phase_deg_unwrapped: np.ndarray
    source_dir: str = ""
    baseline_filter: Dict[str, object] = field(default_factory=dict)
    candidate_resonators: Dict[str, object] = field(default_factory=dict)
    processing_history: List[Dict[str, object]] = field(default_factory=list)

    def amplitude(self) -> np.ndarray:
        return self.s21_amp

    def phase_deg_unwrapped(self) -> np.ndarray:
        return self.s21_phase_deg_unwrapped

    def complex_s21(self) -> np.ndarray:
        return self.s21_amp * np.exp(1j * np.radians(self.s21_phase_deg_unwrapped))


@dataclass
class Dataset:
    source_file: str
    vna_scans: List[VNAScan] = field(default_factory=list)
    selected_scan_keys: List[str] = field(default_factory=list)
    last_saved_at: str = ""
    dataset_name: str = ""
    created_at: str = ""
    transcript: List[Dict[str, str]] = field(default_factory=list)
    processing_history: List[Dict[str, object]] = field(default_factory=list)


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
