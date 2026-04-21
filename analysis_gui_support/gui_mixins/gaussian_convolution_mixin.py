from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter import messagebox

from ..analysis_filters import _estimate_frequency_resolution_mhz, _window_width_in_freq_units
from ..analysis_models import _current_user, _make_event, _read_polar_series

_FWHM_TO_SIGMA = 1.0 / 2.3548200450309493


def _fill_nans_linear(y: np.ndarray) -> np.ndarray:
    out = np.asarray(y, dtype=float).copy()
    good = np.isfinite(out)
    if np.all(good):
        return out
    if not np.any(good):
        return np.zeros_like(out)
    idx = np.arange(out.size)
    out[~good] = np.interp(idx[~good], idx[good], out[good])
    return out


def _gaussian_fft_convolve(y: np.ndarray, dx_freq_units: float, width_freq_units: float) -> np.ndarray:
    if width_freq_units <= 0 or dx_freq_units <= 0 or y.size < 3:
        return np.asarray(y, dtype=float).copy()

    sigma_samples = width_freq_units / dx_freq_units
    if sigma_samples < 1e-6:
        return np.asarray(y, dtype=float).copy()

    half = int(max(3, np.ceil(6.0 * sigma_samples)))
    x = np.arange(-half, half + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_samples) ** 2)
    kernel /= np.sum(kernel)

    yin = _fill_nans_linear(np.asarray(y, dtype=float))
    p = half
    # Reflect-pad before convolution to suppress endpoint artifacts.
    yin_pad = np.pad(yin, (p, p), mode="reflect")
    n = yin_pad.size + kernel.size - 1
    nfft = 1 << int(np.ceil(np.log2(max(2, n))))
    yf = np.fft.rfft(yin_pad, n=nfft)
    kf = np.fft.rfft(kernel, n=nfft)
    conv_full = np.fft.irfft(yf * kf, n=nfft)[:n]
    start = kernel.size // 2
    same_pad = conv_full[start : start + yin_pad.size]
    return same_pad[p : p + yin.size]


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, val in enumerate(mask):
        if val and not in_run:
            start = i
            in_run = True
        elif not val and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(mask) - 1))
    return runs


class GaussianConvolutionMixin:
    def _gauss_autoscale_y_for_visible_x(self, ax) -> None:
        if self.gauss_auto_y_var is None or not bool(self.gauss_auto_y_var.get()):
            return
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

    def open_gaussian_convolution_window(self) -> None:
        if not self._selected_scans_have_attached_normalized():
            messagebox.showwarning(
                "Missing normalized data",
                "All selected scans must have attached normalized data first.",
            )
            return

        if self.gauss_window is not None and self.gauss_window.winfo_exists():
            self.gauss_window.lift()
            return

        scans = self._selected_scans()
        resolution_mhz = _estimate_frequency_resolution_mhz(scans)
        max_khz = 100.0
        min_khz = min(max(resolution_mhz * 1000.0, 1e-3), max_khz)
        default_khz = min(max(15.0, min_khz), max_khz)
        default_thresh = 0.8
        default_min_region_khz = default_khz

        saved_settings = []
        for scan in scans:
            norm = scan.baseline_filter.get("normalized", {})
            if not isinstance(norm, dict):
                continue
            g = norm.get("gaussian_conv")
            if not isinstance(g, dict):
                continue
            keys = ("gaussian_fwhm_ghz", "threshold", "min_region_width_ghz")
            if not all(k in g for k in keys):
                continue
            saved_settings.append(
                (
                    float(g["gaussian_fwhm_ghz"]),
                    float(g["threshold"]),
                    float(g["min_region_width_ghz"]),
                )
            )

        if saved_settings:
            uniq = list(dict.fromkeys(saved_settings))
            chosen = None
            if len(uniq) > 1:
                labels = [
                    (
                        f"{i+1}. FWHM={s[0]*1e6:.3f} kHz, "
                        f"threshold={s[1]:.3f}, min region={s[2]*1e6:.3f} kHz"
                    )
                    for i, s in enumerate(uniq)
                ]
                pick = self._select_setting_option(
                    "Gaussian Setting",
                    "Selected scans have different saved Gaussian settings. Choose defaults:",
                    labels,
                )
                if pick is not None:
                    chosen = uniq[pick]
                    self._log(f"Loaded chosen saved Gaussian setting #{pick + 1} into defaults.")
            else:
                chosen = uniq[0]
                self._log("Loaded saved Gaussian setting into defaults.")
            if chosen is not None:
                default_khz = min(max(chosen[0] * 1e6, min_khz), max_khz)
                default_thresh = min(max(chosen[1], 0.0), 1.0)
                default_min_region_khz = min(max(chosen[2] * 1e6, min_khz), max_khz)

        self.gauss_window = tk.Toplevel(self.root)
        self.gauss_window.title("Gaussian Convolution on Normalized |S21|")
        self.gauss_window.geometry("1200x820")
        self.gauss_window.protocol("WM_DELETE_WINDOW", self._gauss_close)

        controls = tk.Frame(self.gauss_window, padx=8, pady=8)
        controls.pack(side="top", fill="x")
        self.gauss_slider = tk.Scale(
            controls,
            from_=min_khz,
            to=max_khz,
            resolution=min_khz,
            orient="horizontal",
            label="Gaussian FWHM (kHz)",
            command=lambda _v: self._gauss_on_slider_changed(),
            length=360,
        )
        self.gauss_slider.set(default_khz)
        self.gauss_slider.pack(side="left", padx=(0, 12))
        self.gauss_slider.bind("<ButtonRelease-1>", self._gauss_on_slider_released)
        self.gauss_slider.bind("<KeyRelease>", self._gauss_on_slider_released)
        self.gauss_threshold_slider = tk.Scale(
            controls,
            from_=0.0,
            to=1.0,
            resolution=0.01,
            orient="horizontal",
            label="Threshold",
            command=lambda _v: self._gauss_on_slider_changed(),
            length=220,
        )
        self.gauss_threshold_slider.set(default_thresh)
        self.gauss_threshold_slider.pack(side="left", padx=(0, 12))
        self.gauss_threshold_slider.bind("<ButtonRelease-1>", self._gauss_on_slider_released)
        self.gauss_threshold_slider.bind("<KeyRelease>", self._gauss_on_slider_released)
        self.gauss_min_region_slider = tk.Scale(
            controls,
            from_=min_khz,
            to=max_khz,
            resolution=min_khz,
            orient="horizontal",
            label="Min Region Width (kHz)",
            command=lambda _v: self._gauss_on_slider_changed(),
            length=280,
        )
        self.gauss_min_region_slider.set(default_min_region_khz)
        self.gauss_min_region_slider.pack(side="left", padx=(0, 12))
        self.gauss_min_region_slider.bind("<ButtonRelease-1>", self._gauss_on_slider_released)
        self.gauss_min_region_slider.bind("<KeyRelease>", self._gauss_on_slider_released)
        self.gauss_auto_y_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            controls,
            text="Auto-scale |S21| in window",
            variable=self.gauss_auto_y_var,
            command=self._gauss_on_auto_y_toggled,
        ).pack(side="left", padx=(0, 12))

        self.gauss_status_var = tk.StringVar(value="Adjust width and release slider to update.")
        tk.Label(controls, textvariable=self.gauss_status_var, anchor="w").pack(
            side="left", fill="x", expand=True
        )

        action_frame = tk.Frame(self.gauss_window, padx=8, pady=6)
        action_frame.pack(side="top", fill="x")
        tk.Button(action_frame, text="Cancel", width=12, command=self._gauss_close).pack(side="right")
        tk.Button(
            action_frame,
            text="Reset View",
            width=12,
            command=self._gauss_reset_view,
        ).pack(side="right", padx=(8, 0))
        self.gauss_attach_button = tk.Button(
            action_frame,
            text="Attach, Save, and Close",
            width=24,
            command=self._attach_save_and_close_gauss,
        )
        self.gauss_attach_button.pack(side="right", padx=(8, 0))
        self._gauss_set_attach_state(attached=False)

        self.gauss_figure = Figure(figsize=(12, 7))
        self.gauss_canvas = FigureCanvasTkAgg(self.gauss_figure, master=self.gauss_window)
        self.gauss_toolbar = NavigationToolbar2Tk(self.gauss_canvas, self.gauss_window)
        self.gauss_toolbar.update()
        def _home_gauss(*_args) -> None:
            if self.gauss_figure is None or self.gauss_canvas is None:
                return
            for ax in self.gauss_figure.axes:
                ax.relim()
                ax.autoscale(enable=True, axis="both", tight=False)
            self.gauss_canvas.draw_idle()
        self.gauss_toolbar.home = _home_gauss
        self.gauss_toolbar.pack(side="top", fill="x")
        self.gauss_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.gauss_canvas.mpl_connect(
            "button_release_event", lambda _evt: self._gauss_autoscale_all_y()
        )

        self.gauss_preview = {}
        self._gauss_compute_preview()
        self._gauss_render()
        self._log(
            f"Opened Gaussian convolution window. Slider range {min_khz:.3f}-{max_khz:.3f} kHz."
        )

    def _gauss_set_attach_state(self, attached: bool) -> None:
        if self.gauss_attach_button is None:
            return
        if attached:
            self.gauss_attach_button.configure(bg="light green", activebackground="light green")
        else:
            self.gauss_attach_button.configure(bg="pink", activebackground="pink")

    def _gauss_on_slider_changed(self) -> None:
        if self.gauss_status_var is not None:
            self.gauss_status_var.set("Adjusting Gaussian width...")
        self._gauss_set_attach_state(attached=False)

    def _gauss_on_slider_released(self, _event: tk.Event) -> None:
        self._gauss_compute_preview()
        self._gauss_render()
        self._gauss_set_attach_state(attached=False)
        if self.gauss_status_var is not None:
            self.gauss_status_var.set("Preview updated. Attach to save.")

    def _gauss_on_auto_y_toggled(self) -> None:
        if self.gauss_auto_y_var is None:
            return
        if bool(self.gauss_auto_y_var.get()):
            self._gauss_autoscale_all_y()
            if self.gauss_status_var is not None:
                self.gauss_status_var.set("Auto Y enabled.")
        else:
            if self.gauss_status_var is not None:
                self.gauss_status_var.set("Auto Y disabled. Free X/Y zoom enabled.")

    def _gauss_autoscale_all_y(self) -> None:
        if self.gauss_figure is None or self.gauss_canvas is None:
            return
        if self.gauss_auto_y_var is None or not bool(self.gauss_auto_y_var.get()):
            return
        for ax in self.gauss_figure.axes:
            self._gauss_autoscale_y_for_visible_x(ax)
        self.gauss_canvas.draw_idle()

    def _gauss_reset_view(self) -> None:
        if self.gauss_figure is None or self.gauss_canvas is None:
            return
        for ax in self.gauss_figure.axes:
            ax.relim()
            ax.autoscale(enable=True, axis="both", tight=False)
        self.gauss_canvas.draw_idle()

    def _gauss_compute_preview(self) -> None:
        self.gauss_preview = {}
        scans = self._selected_scans()
        if self.gauss_slider is None or self.gauss_threshold_slider is None or self.gauss_min_region_slider is None:
            return
        fwhm_ghz = float(self.gauss_slider.get()) / 1e6
        sigma_ghz = fwhm_ghz * _FWHM_TO_SIGMA
        threshold = float(self.gauss_threshold_slider.get())
        min_region_width_ghz = float(self.gauss_min_region_slider.get()) / 1e6
        for scan in scans:
            norm = scan.baseline_filter.get("normalized", {})
            if not isinstance(norm, dict):
                continue
            amp, _phase = _read_polar_series(
                norm,
                amplitude_key="norm_amp",
                phase_key="norm_phase_deg_unwrapped",
            )
            if amp.shape != scan.freq.shape:
                continue
            f_sorted = np.sort(np.asarray(scan.freq, dtype=float))
            diffs = np.diff(f_sorted)
            diffs = diffs[np.isfinite(diffs)]
            diffs = np.abs(diffs[diffs > 0])
            dx = float(np.median(diffs)) if diffs.size else 0.0
            width_freq_units = float(_window_width_in_freq_units(scan.freq, sigma_ghz))
            smooth = _gaussian_fft_convolve(
                amp, dx_freq_units=dx, width_freq_units=width_freq_units
            )
            min_region_width_freq = float(
                _window_width_in_freq_units(scan.freq, min_region_width_ghz)
            )
            below = smooth < threshold
            accepted_regions: list[tuple[int, int]] = []
            minima_idx: list[int] = []
            for i0, i1 in _true_runs(below):
                if i1 <= i0:
                    continue
                region_width = float(scan.freq[i1] - scan.freq[i0])
                if region_width < min_region_width_freq:
                    continue
                accepted_regions.append((i0, i1))
                local = smooth[i0 : i1 + 1]
                if local.size:
                    minima_idx.append(i0 + int(np.argmin(local)))
            self.gauss_preview[self._scan_key(scan)] = {
                "orig_amp": amp,
                "smooth_amp": smooth,
                "gaussian_fwhm_ghz": fwhm_ghz,
                "gaussian_sigma_ghz": sigma_ghz,
                "threshold": threshold,
                "min_region_width_ghz": min_region_width_ghz,
                "accepted_regions": accepted_regions,
                "minima_idx": np.asarray(minima_idx, dtype=int),
                "dx_ghz": dx,
                "width_freq_units": width_freq_units,
                "sigma_samples": (width_freq_units / dx) if dx > 0 else 0.0,
            }

    def _gauss_render(self) -> None:
        if self.gauss_figure is None or self.gauss_canvas is None:
            return
        saved_limits = []
        for ax_old in self.gauss_figure.axes:
            saved_limits.append((ax_old.get_xlim(), ax_old.get_ylim()))
        scans = self._selected_scans()
        self.gauss_figure.clear()
        if not scans:
            ax = self.gauss_figure.add_subplot(111)
            ax.text(0.5, 0.5, "No selected scans", ha="center", va="center")
            ax.axis("off")
            self.gauss_canvas.draw_idle()
            return

        n = len(scans)
        axes = self.gauss_figure.subplots(n, 1, sharex=False)
        axes = np.atleast_1d(axes)
        for i, scan in enumerate(scans):
            freq_ghz = np.asarray(scan.freq, dtype=float) / 1.0e9
            ax = axes[i]
            prev = self.gauss_preview.get(self._scan_key(scan))
            if prev is None:
                ax.text(0.5, 0.5, "Missing normalized data", ha="center", va="center")
                ax.axis("off")
                continue
            ax.plot(freq_ghz, prev["orig_amp"], color="0.45", linewidth=0.8, label="Original normalized |S21|")
            ax.plot(
                freq_ghz,
                prev["smooth_amp"],
                color="tab:blue",
                linewidth=1.3,
                linestyle="--",
                label="Smoothed (non-accepted)",
            )
            for r0, r1 in prev["accepted_regions"]:
                ax.plot(
                    freq_ghz[r0 : r1 + 1],
                    prev["smooth_amp"][r0 : r1 + 1],
                    color="tab:green",
                    linewidth=1.5,
                    label="Accepted region" if (r0, r1) == prev["accepted_regions"][0] else None,
                )
            min_idx = prev["minima_idx"]
            if min_idx.size:
                ax.plot(
                    freq_ghz[min_idx],
                    prev["orig_amp"][min_idx],
                    linestyle="none",
                    marker="o",
                    markersize=4,
                    color="tab:red",
                    label="Region minima (raw)",
                )
                ax.plot(
                    freq_ghz[min_idx],
                    prev["smooth_amp"][min_idx],
                    linestyle="none",
                    marker="x",
                    markersize=5,
                    color="black",
                    label="Region minima (smooth)",
                )
            ax.set_ylabel("|S21|")
            ax.grid(True, alpha=0.3)
            ax.set_title(Path(scan.filename).name, fontsize=9)
            if i == 0:
                ax.legend(loc="upper right", fontsize=8)
            if i == n - 1:
                ax.set_xlabel("Frequency (GHz)")

        width_khz = float(self.gauss_slider.get()) if self.gauss_slider is not None else 0.0
        threshold = (
            float(self.gauss_threshold_slider.get()) if self.gauss_threshold_slider is not None else 0.0
        )
        min_region_khz = (
            float(self.gauss_min_region_slider.get()) if self.gauss_min_region_slider is not None else 0.0
        )
        self.gauss_figure.suptitle(
            f"Gaussian Convolution | FWHM={width_khz:.3f} kHz | threshold={threshold:.2f} | min width={min_region_khz:.3f} kHz",
            fontsize=11,
        )
        if len(saved_limits) == len(axes):
            for ax_new, (xlim, ylim) in zip(axes, saved_limits):
                ax_new.set_xlim(xlim)
                if self.gauss_auto_y_var is not None and bool(self.gauss_auto_y_var.get()):
                    self._gauss_autoscale_y_for_visible_x(ax_new)
                else:
                    ax_new.set_ylim(ylim)
        for ax_new in axes:
            ax_new.callbacks.connect(
                "xlim_changed",
                lambda changed_ax: self._gauss_autoscale_y_for_visible_x(changed_ax),
            )
        self.gauss_figure.tight_layout()
        self.gauss_canvas.draw_idle()

    def _gauss_attach(self) -> None:
        scans = self._selected_scans()
        if not scans:
            return
        attached_at = datetime.now().isoformat(timespec="seconds")
        count = 0
        overwritten = 0
        for scan in scans:
            prev = self.gauss_preview.get(self._scan_key(scan))
            if prev is None:
                continue
            norm = scan.baseline_filter.get("normalized", {})
            if not isinstance(norm, dict):
                continue
            if isinstance(norm.get("gaussian_conv"), dict) and norm["gaussian_conv"]:
                overwritten += 1
            norm["gaussian_conv"] = {}
            norm["gaussian_conv"] = {
                "attached_at": attached_at,
                "attached_by": _current_user(),
                "gaussian_fwhm_ghz": prev["gaussian_fwhm_ghz"],
                "gaussian_fwhm_khz": prev["gaussian_fwhm_ghz"] * 1e6,
                "gaussian_fwhm_mhz": prev["gaussian_fwhm_ghz"] * 1000.0,
                "gaussian_sigma_ghz": prev["gaussian_sigma_ghz"],
                "gaussian_sigma_khz": prev["gaussian_sigma_ghz"] * 1e6,
                "gaussian_sigma_mhz": prev["gaussian_sigma_ghz"] * 1000.0,
                "threshold": prev["threshold"],
                "min_region_width_ghz": prev["min_region_width_ghz"],
                "min_region_width_khz": prev["min_region_width_ghz"] * 1e6,
                "min_region_width_mhz": prev["min_region_width_ghz"] * 1000.0,
                "accepted_regions_indices": prev["accepted_regions"],
                "accepted_regions_freq": [
                    (float(scan.freq[i0]), float(scan.freq[i1])) for i0, i1 in prev["accepted_regions"]
                ],
                "minima_indices": prev["minima_idx"],
                "minima_freq": scan.freq[prev["minima_idx"]],
                "smooth_amp": prev["smooth_amp"],
                "gaussian_data": np.vstack((scan.freq, prev["smooth_amp"])),
                "gaussian_data_format": "(2, N) rows = [freq, smooth_amp]",
            }
            scan.candidate_resonators["gaussian_convolution"] = {
                "attached_at": attached_at,
                "attached_by": _current_user(),
                "method": "gaussian_convolution",
                "candidate_freq": np.asarray(scan.freq[prev["minima_idx"]], dtype=float),
                "candidate_indices": np.asarray(prev["minima_idx"], dtype=int),
                "threshold": prev["threshold"],
                "min_region_width_ghz": prev["min_region_width_ghz"],
                "gaussian_fwhm_ghz": prev["gaussian_fwhm_ghz"],
            }
            scan.processing_history.append(
                _make_event(
                    "attach_gaussian_convolution",
                    {
                        "gaussian_fwhm_ghz": prev["gaussian_fwhm_ghz"],
                        "gaussian_fwhm_khz": prev["gaussian_fwhm_ghz"] * 1e6,
                        "gaussian_fwhm_mhz": prev["gaussian_fwhm_ghz"] * 1000.0,
                        "gaussian_sigma_ghz": prev["gaussian_sigma_ghz"],
                        "gaussian_sigma_khz": prev["gaussian_sigma_ghz"] * 1e6,
                        "gaussian_sigma_mhz": prev["gaussian_sigma_ghz"] * 1000.0,
                        "threshold": prev["threshold"],
                        "min_region_width_ghz": prev["min_region_width_ghz"],
                        "min_region_width_khz": prev["min_region_width_ghz"] * 1e6,
                        "accepted_region_count": int(len(prev["accepted_regions"])),
                        "candidate_count": int(len(prev["minima_idx"])),
                        "points": int(scan.freq.size),
                    },
                )
            )
            count += 1

        self.dataset.processing_history.append(
            _make_event("attach_gaussian_convolution_selected", {"selected_count": count})
        )
        self._mark_dirty()
        self._refresh_status()
        self._gauss_set_attach_state(attached=True)
        if self.gauss_status_var is not None:
            self.gauss_status_var.set(
                f"Attached Gaussian-smoothed data to {count} selected scan(s). Overwrote {overwritten}."
            )
        self._log(
            f"Attached Gaussian-smoothed normalized |S21| to {count} selected scan(s); overwrote {overwritten} prior attachment(s)."
        )
        self._autosave_dataset()

    def _gauss_close(self) -> None:
        if self.gauss_window is not None and self.gauss_window.winfo_exists():
            self.gauss_window.destroy()
        self.gauss_window = None
        self.gauss_canvas = None
        self.gauss_toolbar = None
        self.gauss_figure = None
        self.gauss_slider = None
        self.gauss_threshold_slider = None
        self.gauss_min_region_slider = None
        self.gauss_auto_y_var = None
        self.gauss_status_var = None
        self.gauss_attach_button = None
        self.gauss_preview = {}
