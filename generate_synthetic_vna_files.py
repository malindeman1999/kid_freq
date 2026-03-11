"""
Generate synthetic 3-resonator VNA files on the frequency grid of a real scan.

Workflow:
1) Open file picker (starts in "VNA data" if present) and select a real .npy file.
2) Read its frequency grid.
3) Generate 3 synthetic scans with Al MKID-like parameters using ComplexResonanceDirect.
4) Each subsequent synthetic scan shifts all resonance frequencies down by 0.1%.
5) Save files in row format: [freq, real(S21), imag(S21)].
"""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np

from ComplexResonance import ComplexResonanceDirect


def _load_frequency_grid(path: Path) -> np.ndarray:
    arr = np.load(path)
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")

    # Primary expected format in this project: (3, N), first row is frequency.
    if arr.shape[0] >= 1 and arr.shape[0] <= 6 and arr.shape[1] > arr.shape[0]:
        freq = np.asarray(arr[0, :], dtype=float)
    # Alternate layout: (N, 3), first column is frequency.
    elif arr.shape[1] >= 1 and arr.shape[0] > arr.shape[1]:
        freq = np.asarray(arr[:, 0], dtype=float)
    else:
        # Fallback: use first row.
        freq = np.asarray(arr[0, :], dtype=float)

    if freq.size < 10:
        raise ValueError("Frequency grid is too small.")
    if not np.all(np.isfinite(freq)):
        raise ValueError("Frequency grid contains non-finite values.")
    return freq


def _mk_loaded_q(qi: float, qcom: complex) -> float:
    return 1.0 / (1.0 / qi + np.real(1.0 / qcom))


def _synthesize_scan(freq: np.ndarray, frs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Al MKID-like resonator parameters (realistic order of magnitude).
    qis = np.array([2.8e5, 3.4e5, 2.5e5], dtype=float)
    qcmags = np.array([1.2e5, 1.6e5, 1.1e5], dtype=float)
    phi_deg = np.array([6.0, 11.0, 8.0], dtype=float)

    s21 = np.ones_like(freq, dtype=complex)
    for fr, qi, qcmag, phdeg in zip(frs, qis, qcmags, phi_deg):
        qcom = qcmag * np.exp(-1j * np.deg2rad(phdeg))
        q_loaded = _mk_loaded_q(qi, qcom)
        # Per-resonator contribution only; global baseline added after product.
        s21 *= ComplexResonanceDirect(freq, fr, q_loaded, qcom, 1.0 + 0j, 0.0)

    # Add realistic global gain/phase and cable delay.
    a = 0.97 * np.exp(1j * np.deg2rad(9.0))
    tau = 30e-9
    s21 *= a * np.exp(1j * 2.0 * np.pi * freq * tau)

    # Small complex measurement-like noise.
    noise = 2e-4 * (rng.normal(size=freq.size) + 1j * rng.normal(size=freq.size))
    s21_noisy = s21 + noise
    return np.vstack((freq, np.real(s21_noisy), np.imag(s21_noisy)))


def main() -> None:
    root = tk.Tk()
    root.withdraw()

    start_dir = Path("VNA data")
    if not start_dir.exists():
        start_dir = Path.cwd()

    src = filedialog.askopenfilename(
        title="Select real VNA .npy file for frequency grid",
        initialdir=str(start_dir.resolve()),
        filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
    )
    if not src:
        return

    src_path = Path(src).resolve()
    try:
        freq = _load_frequency_grid(src_path)
    except Exception as exc:
        messagebox.showerror("Load error", f"Could not load frequency grid:\n{exc}")
        return

    fmin = float(np.min(freq))
    fmax = float(np.max(freq))
    span = fmax - fmin
    if span <= 0:
        messagebox.showerror("Invalid grid", "Frequency span must be positive.")
        return

    # Place 3 resonances across the measured span.
    fr0 = np.array([fmin + 0.22 * span, fmin + 0.53 * span, fmin + 0.80 * span], dtype=float)

    project_root = Path(__file__).resolve().parent
    out_dir = project_root / "synthetic data"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260311)

    shift_factor = 0.999  # down by 0.1%
    out_paths = []
    for i in range(3):
        frs = fr0 * (shift_factor**i)
        arr = _synthesize_scan(freq, frs, rng)
        out_path = out_dir / f"synthetic_vna_3res_shift_{i:02d}.npy"
        np.save(out_path, arr)
        out_paths.append(out_path)

    msg = "Generated synthetic files:\n" + "\n".join(str(p) for p in out_paths)
    print(msg)
    messagebox.showinfo("Done", msg)


if __name__ == "__main__":
    main()
