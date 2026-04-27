from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter import filedialog, messagebox

from resonator.ComplexResonance import ComplexResonanceQi

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VNA_DATA_DIR = PROJECT_ROOT.parent / "VNA data"
SYNTHETIC_VNA_OUTPUT_DIR = VNA_DATA_DIR / "synthetic_vna"


def _load_frequency_grid(path: Path) -> np.ndarray:
    arr = np.load(path)
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    if arr.shape[0] >= 1 and arr.shape[0] <= 6 and arr.shape[1] > arr.shape[0]:
        freq = np.asarray(arr[0, :], dtype=float)
    elif arr.shape[1] >= 1 and arr.shape[0] > arr.shape[1]:
        freq = np.asarray(arr[:, 0], dtype=float)
    else:
        freq = np.asarray(arr[0, :], dtype=float)
    if freq.size < 10:
        raise ValueError("Frequency grid is too small.")
    if not np.all(np.isfinite(freq)):
        raise ValueError("Frequency grid contains non-finite values.")
    return freq


class SyntheticGeneratorMixin:
    def _synth_autoscale_amp_y_for_visible_x(self) -> None:
        if (
            self.synth_auto_y_var is None
            or not bool(self.synth_auto_y_var.get())
            or self.synth_amp_ax is None
        ):
            return
        ax = self.synth_amp_ax
        lines = [ln for ln in ax.get_lines() if ln.get_visible()]
        if not lines:
            return
        x0, x1 = ax.get_xlim()
        lo, hi = (x0, x1) if x0 <= x1 else (x1, x0)
        y_chunks = []
        for ln in lines:
            x = np.asarray(ln.get_xdata(), dtype=float)
            y = np.asarray(ln.get_ydata(), dtype=float)
            if x.size == 0 or y.size == 0 or x.size != y.size:
                continue
            mask = np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi)
            if np.any(mask):
                y_chunks.append(y[mask])
        if not y_chunks:
            return
        y_all = np.concatenate(y_chunks)
        y_min = float(np.min(y_all))
        y_max = float(np.max(y_all))
        pad = 1.0 if y_max <= y_min else 0.05 * (y_max - y_min)
        ax.set_ylim(y_min - pad, y_max + pad)

    def open_synthetic_generator_window(self) -> None:
        if self.synth_window is not None and self.synth_window.winfo_exists():
            self.synth_window.lift()
            return

        self.synth_window = tk.Toplevel(self.root)
        self.synth_window.title("Generate Synthetic VNA Data")
        self.synth_window.geometry("1250x860")
        self.synth_window.protocol("WM_DELETE_WINDOW", self._synth_close)

        controls = tk.Frame(self.synth_window, padx=8, pady=8)
        controls.pack(side="top", fill="x")
        tk.Button(
            controls,
            text="Select Source VNA Grid",
            command=self._synth_select_source_file,
            width=22,
        ).pack(side="left", padx=(0, 10))
        self.synth_source_var = tk.StringVar(value="No source file selected.")
        tk.Label(controls, textvariable=self.synth_source_var, anchor="w").pack(
            side="left", fill="x", expand=True
        )

        sliders = tk.Frame(self.synth_window, padx=8, pady=4)
        sliders.pack(side="top", fill="x")

        self.synth_num_res_slider = tk.Scale(
            sliders,
            from_=1,
            to=10,
            resolution=1,
            orient="horizontal",
            label="Number of Resonances",
            command=lambda _v: self._synth_on_slider_changed(),
            length=180,
        )
        self.synth_num_res_slider.set(3)
        self.synth_num_res_slider.pack(side="left", padx=(0, 8))
        self.synth_num_res_slider.bind("<ButtonRelease-1>", self._synth_on_slider_released)
        self.synth_num_res_slider.bind("<KeyRelease>", self._synth_on_slider_released)

        self.synth_freq_offset_slider = tk.Scale(
            sliders,
            from_=0.0,
            to=1.0,
            resolution=0.01,
            orient="horizontal",
            label="Freq Shift Between Files (%)",
            command=lambda _v: self._synth_on_slider_changed(),
            length=220,
        )
        self.synth_freq_offset_slider.set(0.1)
        self.synth_freq_offset_slider.pack(side="left", padx=(0, 8))
        self.synth_freq_offset_slider.bind("<ButtonRelease-1>", self._synth_on_slider_released)
        self.synth_freq_offset_slider.bind("<KeyRelease>", self._synth_on_slider_released)

        self.synth_num_files_slider = tk.Scale(
            sliders,
            from_=1,
            to=10,
            resolution=1,
            orient="horizontal",
            label="Number of Files",
            command=lambda _v: self._synth_on_slider_changed(),
            length=160,
        )
        self.synth_num_files_slider.set(3)
        self.synth_num_files_slider.pack(side="left", padx=(0, 8))
        self.synth_num_files_slider.bind("<ButtonRelease-1>", self._synth_on_slider_released)
        self.synth_num_files_slider.bind("<KeyRelease>", self._synth_on_slider_released)
        self.synth_auto_y_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            sliders,
            text="Auto-scale |S21| in window",
            variable=self.synth_auto_y_var,
            command=self._synth_on_auto_y_toggled,
        ).pack(side="left", padx=(8, 0))

        self.synth_qc_slider = tk.Scale(
            sliders,
            from_=1000.0,
            to=100000.0,
            resolution=100.0,
            orient="horizontal",
            label="Coupling Q Magnitude",
            command=lambda _v: self._synth_on_slider_changed(),
            length=200,
        )
        self.synth_qc_slider.set(15000.0)
        self.synth_qc_slider.pack(side="left", padx=(0, 8))
        self.synth_qc_slider.bind("<ButtonRelease-1>", self._synth_on_slider_released)
        self.synth_qc_slider.bind("<KeyRelease>", self._synth_on_slider_released)

        self.synth_qi_slider = tk.Scale(
            sliders,
            from_=1.0e5,
            to=5.0e6,
            resolution=1.0e4,
            orient="horizontal",
            label="Resonator Q Magnitude",
            command=lambda _v: self._synth_on_slider_changed(),
            length=220,
        )
        self.synth_qi_slider.set(1.0e6)
        self.synth_qi_slider.pack(side="left")
        self.synth_qi_slider.bind("<ButtonRelease-1>", self._synth_on_slider_released)
        self.synth_qi_slider.bind("<KeyRelease>", self._synth_on_slider_released)

        actions = tk.Frame(self.synth_window, padx=8, pady=6)
        actions.pack(side="top", fill="x")
        tk.Button(actions, text="Close", width=12, command=self._synth_close).pack(side="right")
        self.synth_generate_button = tk.Button(
            actions, text="Generate Files", width=14, command=self._synth_generate_files
        )
        self.synth_generate_button.pack(side="right", padx=(8, 0))

        self.synth_status_var = tk.StringVar(
            value="Select source file (sets synthetic frequency range and spacing), then adjust sliders."
        )
        tk.Label(actions, textvariable=self.synth_status_var, anchor="w").pack(
            side="left", fill="x", expand=True
        )

        self.synth_figure = Figure(figsize=(12, 7))
        self.synth_canvas = FigureCanvasTkAgg(self.synth_figure, master=self.synth_window)
        self.synth_toolbar = NavigationToolbar2Tk(self.synth_canvas, self.synth_window)
        self.synth_toolbar.update()
        self.synth_toolbar.pack(side="top", fill="x")
        self.synth_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.synth_canvas.mpl_connect("button_release_event", lambda _e: self._synth_on_zoom_release())

        self.synth_preview_files = []
        self.synth_source_path = None
        self.synth_freq = None
        self._synth_render_placeholder()

    def _synth_on_slider_changed(self) -> None:
        if self.synth_status_var is not None:
            self.synth_status_var.set("Adjusting parameters...")

    def _synth_on_slider_released(self, _event: tk.Event) -> None:
        self._synth_update_preview()

    def _synth_on_auto_y_toggled(self) -> None:
        if self.synth_auto_y_var is None:
            return
        if bool(self.synth_auto_y_var.get()):
            self._synth_autoscale_amp_y_for_visible_x()
            if self.synth_canvas is not None:
                self.synth_canvas.draw_idle()

    def _synth_on_zoom_release(self) -> None:
        if self.synth_auto_y_var is None or not bool(self.synth_auto_y_var.get()):
            return
        self._synth_autoscale_amp_y_for_visible_x()
        if self.synth_canvas is not None:
            self.synth_canvas.draw_idle()

    def _synth_render_placeholder(self) -> None:
        if self.synth_figure is None or self.synth_canvas is None:
            return
        self.synth_figure.clear()
        ax = self.synth_figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "Select a source VNA file to preview synthetic data.\n"
            "Source file sets synthetic frequency range and spacing.",
            ha="center",
            va="center",
        )
        ax.axis("off")
        self.synth_canvas.draw_idle()

    def _synth_select_source_file(self) -> None:
        initial = VNA_DATA_DIR
        if not initial.exists():
            initial = Path.cwd()
        path_text = filedialog.askopenfilename(
            title="Select source VNA file (.npy)",
            initialdir=str(initial.resolve()),
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
        )
        if not path_text:
            return
        path = Path(path_text).resolve()
        try:
            freq = _load_frequency_grid(path)
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            return
        self.synth_source_path = path
        self.synth_freq = freq
        if self.synth_source_var is not None:
            self.synth_source_var.set(str(path))
        self._synth_update_preview()

    def _synth_build_preview_data(self) -> list[np.ndarray]:
        if self.synth_freq is None:
            return []
        freq = np.asarray(self.synth_freq, dtype=float)
        n_res = int(self.synth_num_res_slider.get())
        n_files = int(self.synth_num_files_slider.get())
        shift_pct = float(self.synth_freq_offset_slider.get())
        qc_mag = float(self.synth_qc_slider.get())
        qi = float(self.synth_qi_slider.get())

        fmin = float(np.min(freq))
        fmax = float(np.max(freq))
        span = fmax - fmin
        fr0 = np.linspace(fmin + 0.15 * span, fmin + 0.85 * span, n_res)
        shift_factor = 1.0 - (shift_pct / 100.0)
        if shift_factor <= 0:
            shift_factor = 1e-6

        rng = np.random.default_rng(20260311)
        out: list[np.ndarray] = []
        for i_file in range(n_files):
            frs = fr0 * (shift_factor**i_file)
            s21 = np.ones_like(freq, dtype=complex)
            for j, fr in enumerate(frs):
                phi_deg = 6.0 + 2.5 * (j % 5)
                qcom_eff = qc_mag * np.exp(-1j * np.deg2rad(phi_deg))
                s21 *= ComplexResonanceQi(freq, fr, qi, qcom_eff, 1.0 + 0j, 0.0)

            a = 0.97 * np.exp(1j * np.deg2rad(9.0))
            tau = 30e-9
            s21 *= a * np.exp(1j * 2.0 * np.pi * freq * tau)
            noise = 2e-4 * (rng.normal(size=freq.size) + 1j * rng.normal(size=freq.size))
            s21 += noise
            out.append(np.vstack((freq, np.real(s21), np.imag(s21))))
        return out

    def _synth_update_preview(self) -> None:
        if self.synth_freq is None:
            self._synth_render_placeholder()
            return
        self.synth_preview_files = self._synth_build_preview_data()
        self._synth_render_preview()
        if self.synth_status_var is not None:
            self.synth_status_var.set("Preview updated. Use Generate Files to export.")

    def _synth_render_preview(self) -> None:
        if self.synth_figure is None or self.synth_canvas is None:
            return
        amp_xlim = self.synth_amp_ax.get_xlim() if self.synth_amp_ax is not None else None
        amp_ylim = self.synth_amp_ax.get_ylim() if self.synth_amp_ax is not None else None
        self.synth_figure.clear()
        ax1 = self.synth_figure.add_subplot(1, 2, 1)
        ax2 = self.synth_figure.add_subplot(1, 2, 2)
        self.synth_amp_ax = ax1
        self.synth_iq_ax = ax2
        if not self.synth_preview_files:
            ax1.text(0.5, 0.5, "No preview data", ha="center", va="center")
            ax1.axis("off")
            ax2.axis("off")
            self.synth_canvas.draw_idle()
            return

        for i, arr in enumerate(self.synth_preview_files):
            freq = np.asarray(arr[0, :], dtype=float) / 1.0e9
            z = arr[1, :] + 1j * arr[2, :]
            amp = np.abs(z)
            ax1.plot(freq, amp, linewidth=1.0, label=f"File {i+1}")
            if i == 0:
                ax2.plot(np.real(z), np.imag(z), linewidth=1.0, label=f"File {i+1}")
        ax1.set_title("Amplitude Preview")
        ax1.set_xlabel("Frequency (GHz)")
        ax1.set_ylabel("|S21|")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best", fontsize=8)
        ax2.set_title("IQ Preview (File 1)")
        ax2.set_xlabel("Re(S21)")
        ax2.set_ylabel("Im(S21)")
        ax2.grid(True, alpha=0.3)
        ax2.axis("equal")
        if amp_xlim is not None and amp_ylim is not None:
            ax1.set_xlim(amp_xlim)
            if self.synth_auto_y_var is not None and bool(self.synth_auto_y_var.get()):
                self._synth_autoscale_amp_y_for_visible_x()
            else:
                ax1.set_ylim(amp_ylim)
        ax1.callbacks.connect("xlim_changed", lambda _ax: self._synth_on_zoom_release())
        self.synth_figure.tight_layout()
        self.synth_canvas.draw_idle()

    def _synth_generate_files(self) -> None:
        if self.synth_freq is None:
            messagebox.showwarning("No source file", "Select a source VNA file first.")
            return
        if not self.synth_preview_files:
            self.synth_preview_files = self._synth_build_preview_data()
        out_dir = SYNTHETIC_VNA_OUTPUT_DIR.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved = []
        for i, arr in enumerate(self.synth_preview_files):
            out_path = out_dir / f"synthetic_vna_{ts}_{i:02d}.npy"
            np.save(out_path, arr)
            saved.append(out_path)

        self._mark_dirty()
        self._refresh_status()
        self._log(f"Generated {len(saved)} synthetic VNA files in {out_dir}")
        self._autosave_dataset()
        if self.synth_status_var is not None:
            self.synth_status_var.set(f"Generated {len(saved)} file(s).")
        messagebox.showinfo(
            "Synthetic data generated",
            "Generated files:\n" + "\n".join(str(p) for p in saved),
        )

    def _synth_close(self) -> None:
        if self.synth_window is not None and self.synth_window.winfo_exists():
            self.synth_window.destroy()
        self.synth_window = None
        self.synth_canvas = None
        self.synth_toolbar = None
        self.synth_figure = None
        self.synth_source_var = None
        self.synth_status_var = None
        self.synth_auto_y_var = None
        self.synth_generate_button = None
        self.synth_num_res_slider = None
        self.synth_freq_offset_slider = None
        self.synth_num_files_slider = None
        self.synth_qc_slider = None
        self.synth_qi_slider = None
        self.synth_source_path = None
        self.synth_freq = None
        self.synth_preview_files = []
        self.synth_amp_ax = None
        self.synth_iq_ax = None
