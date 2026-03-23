from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import AutoLocator, AutoMinorLocator, MultipleLocator
from tkinter import messagebox, ttk

from ..analysis_models import _complex_from_polar, _current_user, _make_event
from phase_correction import process_phase_data

DEFAULT_PHASE2_THRESHOLD_DEG = 10.0
DEFAULT_PHASE2_CORRECT_OTHER_DISCONTINUITIES = False
DEFAULT_PHASE2_MAX_PASSES = 3
DEFAULT_PHASE2_MIN_SEPARATION_HZ = 15e3
DEFAULT_PHASE2_P_RANDOM_CUTOFF = 1e-3


class PhaseCorrection2Mixin:
    def _phase2_values_at_freqs(self, sample_freqs, ref_freqs, ref_values) -> np.ndarray:
        sample = np.asarray(sample_freqs, dtype=float)
        ref_f = np.asarray(ref_freqs, dtype=float)
        ref_v = np.asarray(ref_values, dtype=float)
        if sample.size == 0:
            return np.asarray([], dtype=float)
        idx = np.searchsorted(ref_f, sample)
        idx = np.clip(idx, 0, max(ref_f.size - 1, 0))
        left = np.clip(idx - 1, 0, max(ref_f.size - 1, 0))
        use_left = np.abs(sample - ref_f[left]) <= np.abs(sample - ref_f[idx])
        chosen = np.where(use_left, left, idx)
        return ref_v[chosen]

    def _phase2_get_step1_class_points(self, scan) -> dict:
        points = scan.candidate_resonators.get("phase_class_points", {})
        if not isinstance(points, dict):
            raise ValueError("Phase Correction 1 classification points are missing.")
        return {
            "irregular_congruent_freqs": np.asarray(points["irregular_congruent_freqs"], dtype=float),
            "irregular_noncongruent_freqs": np.asarray(points["irregular_noncongruent_freqs"], dtype=float),
        }

    def open_second_phase_correction_window(self) -> None:
        scans = self._selected_scans()
        if not scans:
            messagebox.showwarning("No selection", "Select scans for analysis first.")
            return
        if any(not scan.has_dewrapped_phase() for scan in scans):
            messagebox.showwarning(
                "Missing Phase Correction 1 output",
                "Run Phase Correction 1 and click Attach for all selected scans.",
            )
            return
        if self.phase2_window is not None and self.phase2_window.winfo_exists():
            self.phase2_window.lift()
            return

        self.phase2_window = tk.Toplevel(self.root)
        self.phase2_window.title("Phase Correction 2")
        self.phase2_window.geometry("1280x900")
        self.phase2_window.protocol("WM_DELETE_WINDOW", self._phase2_close)

        controls = tk.Frame(self.phase2_window, padx=8, pady=8)
        controls.pack(side="top", fill="x")

        self.phase2_threshold_slider = tk.Scale(controls, from_=1.0, to=120.0, resolution=1.0, orient="horizontal", label="Threshold (deg)", command=lambda _v: self._phase2_on_control_changed(), length=220)
        self.phase2_threshold_slider.set(DEFAULT_PHASE2_THRESHOLD_DEG)
        self.phase2_threshold_slider.grid(row=0, column=0, padx=(0, 8), sticky="w")
        self.phase2_threshold_slider.bind("<ButtonRelease-1>", self._phase2_on_control_released)
        self.phase2_threshold_slider.bind("<KeyRelease>", self._phase2_on_control_released)

        self.phase2_max_passes_slider = tk.Scale(controls, from_=1, to=8, resolution=1, orient="horizontal", label="Max Passes", command=lambda _v: self._phase2_on_control_changed(), length=180)
        self.phase2_max_passes_slider.set(DEFAULT_PHASE2_MAX_PASSES)
        self.phase2_max_passes_slider.grid(row=0, column=1, padx=(0, 8), sticky="w")
        self.phase2_max_passes_slider.bind("<ButtonRelease-1>", self._phase2_on_control_released)
        self.phase2_max_passes_slider.bind("<KeyRelease>", self._phase2_on_control_released)

        self.phase2_min_sep_slider = tk.Scale(controls, from_=1.0, to=500.0, resolution=1.0, orient="horizontal", label="Min Separation (kHz)", command=lambda _v: self._phase2_on_control_changed(), length=220)
        self.phase2_min_sep_slider.set(DEFAULT_PHASE2_MIN_SEPARATION_HZ / 1e3)
        self.phase2_min_sep_slider.grid(row=0, column=2, padx=(0, 8), sticky="w")
        self.phase2_min_sep_slider.bind("<ButtonRelease-1>", self._phase2_on_control_released)
        self.phase2_min_sep_slider.bind("<KeyRelease>", self._phase2_on_control_released)

        self.phase2_correct_other_var = tk.BooleanVar(value=DEFAULT_PHASE2_CORRECT_OTHER_DISCONTINUITIES)
        tk.Checkbutton(controls, text="Correct other phase discontinuities", variable=self.phase2_correct_other_var, command=self._phase2_on_toggle_changed).grid(row=0, column=3, padx=(0, 12), sticky="w")

        self.phase2_auto_y_var = tk.BooleanVar(value=True)
        tk.Checkbutton(controls, text="Auto-scale phase in window", variable=self.phase2_auto_y_var, command=self._phase2_on_auto_y_toggled).grid(row=0, column=4, padx=(0, 12), sticky="w")
        self.phase2_mod360_var = tk.BooleanVar(value=False)
        tk.Checkbutton(controls, text="Plot mod 360", variable=self.phase2_mod360_var, command=self._phase2_on_toggle_changed).grid(row=0, column=5, padx=(0, 12), sticky="w")

        tk.Label(controls, text="p-random cutoff:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.phase2_p_random_var = tk.StringVar(value=f"{DEFAULT_PHASE2_P_RANDOM_CUTOFF:g}")
        self.phase2_p_random_entry = tk.Entry(controls, textvariable=self.phase2_p_random_var, width=12)
        self.phase2_p_random_entry.grid(row=1, column=1, sticky="w", pady=(6, 0))
        self.phase2_update_button = tk.Button(controls, text="Update Preview", width=16, command=self._phase2_update_preview)
        self.phase2_update_button.grid(row=1, column=2, sticky="w", pady=(6, 0))

        self.phase2_status_var = tk.StringVar(value="Opening window and preparing preview...")
        tk.Label(controls, textvariable=self.phase2_status_var, anchor="w").grid(row=1, column=3, columnspan=2, sticky="we", pady=(6, 0))
        self.phase2_progress = ttk.Progressbar(controls, orient="horizontal", mode="determinate", length=280)
        self.phase2_progress.grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))

        actions = tk.Frame(self.phase2_window, padx=8, pady=6)
        actions.pack(side="top", fill="x")
        tk.Button(actions, text="Cancel", width=12, command=self._phase2_close).pack(side="right")
        tk.Button(actions, text="Reset View", width=12, command=self._phase2_reset_view).pack(side="right", padx=(8, 0))
        self.phase2_attach_button = tk.Button(actions, text="Attach, Save, and Close", width=24, command=self._attach_save_and_close_phase2)
        self.phase2_attach_button.pack(side="right", padx=(8, 0))
        self._phase2_set_attach_state(attached=False)

        self.phase2_figure = Figure(figsize=(12, 7))
        self.phase2_canvas = FigureCanvasTkAgg(self.phase2_figure, master=self.phase2_window)
        self.phase2_toolbar = NavigationToolbar2Tk(self.phase2_canvas, self.phase2_window)
        self.phase2_toolbar.update()
        self.phase2_toolbar.pack(side="top", fill="x")
        self.phase2_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.phase2_canvas.mpl_connect("button_release_event", lambda _e: self._phase2_autoscale_all_y())

        self.phase2_preview = {}
        self.phase2_window.update_idletasks()
        self.phase2_window.after(10, self._phase2_update_preview)

    def _phase2_get_settings(self) -> dict:
        p_cutoff_text = self.phase2_p_random_var.get().strip() if self.phase2_p_random_var is not None else "1e-3"
        p_random_cutoff = float(p_cutoff_text)
        if p_random_cutoff <= 0.0:
            raise ValueError("p-random cutoff must be > 0.")
        return {
            "threshold_deg": float(self.phase2_threshold_slider.get()),
            "max_passes": int(self.phase2_max_passes_slider.get()),
            "correct_other_discontinuities": bool(self.phase2_correct_other_var.get()),
            "min_separation_hz": float(self.phase2_min_sep_slider.get()) * 1e3,
            "p_random_cutoff": p_random_cutoff,
        }

    def _phase2_set_attach_state(self, attached: bool) -> None:
        if self.phase2_attach_button is not None:
            self.phase2_attach_button.configure(bg="light green" if attached else "pink", activebackground="light green" if attached else "pink")

    def _phase2_on_control_changed(self, *_args) -> None:
        self._phase2_set_attach_state(attached=False)
        if self.phase2_status_var is not None:
            self.phase2_status_var.set("Settings changed. Click Update Preview.")

    def _phase2_on_control_released(self, _event: tk.Event) -> None:
        self._phase2_on_control_changed()

    def _phase2_on_toggle_changed(self) -> None:
        self._phase2_on_control_changed()
        self._phase2_update_preview()
        self._phase2_reset_view()

    def _phase2_update_preview(self) -> None:
        scans = self._selected_scans()
        if not scans:
            return
        try:
            settings = self._phase2_get_settings()
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return
        self.phase2_preview = {}
        if self.phase2_update_button is not None:
            self.phase2_update_button.configure(state="disabled")
        if self.phase2_progress is not None:
            self.phase2_progress.configure(maximum=max(len(scans), 1), value=0)
        failed = []
        for idx, scan in enumerate(scans, start=1):
            if self.phase2_status_var is not None:
                self.phase2_status_var.set(
                    f"Processing phase correction 2 preview: {idx}/{len(scans)} | {Path(scan.filename).name}"
                )
            if self.phase2_progress is not None:
                self.phase2_progress.configure(value=idx - 1)
            if self.phase2_window is not None and self.phase2_window.winfo_exists():
                self.phase2_window.update_idletasks()
            try:
                step1_points = self._phase2_get_step1_class_points(scan)
                freq = np.asarray(scan.freq, dtype=float)
                phase_input = np.asarray(scan.phase_deg_unwrapped(), dtype=float)
                phase_input_unwrapped = np.degrees(np.unwrap(np.radians(phase_input)))
                s21 = _complex_from_polar(np.asarray(scan.amplitude(), dtype=float), phase_input)
                order = np.argsort(freq)
                inv = np.empty_like(order)
                inv[order] = np.arange(order.size)
                processed = process_phase_data(
                    freq[order],
                    s21[order],
                    threshold_deg=settings["threshold_deg"],
                    apply_exact_360=False,
                    max_passes=settings["max_passes"],
                    min_separation_hz=settings["min_separation_hz"],
                    p_random_cutoff=settings["p_random_cutoff"],
                    correct_non_congruent=settings["correct_other_discontinuities"],
                    verbose=False,
                )
                phase_corrected_sorted = np.asarray(processed["phase_corrected"], dtype=float)
                phase_corrected_unwrapped_sorted = np.degrees(np.unwrap(np.radians(phase_corrected_sorted)))
                self.phase2_preview[self._scan_key(scan)] = {
                    "freq_sorted": freq[order],
                    "phase_input_sorted": phase_input_unwrapped[order],
                    "phase_input_mod360_sorted": np.mod(phase_input[order], 360.0),
                    "phase_corrected_initial_sorted": np.asarray(processed["phase_corrected_initial"], dtype=float),
                    "phase_corrected_initial_mod360_sorted": np.asarray(processed["phase_corrected_initial_mod360"], dtype=float),
                    "phase_corrected_sorted": phase_corrected_unwrapped_sorted,
                    "phase_corrected_mod360_sorted": np.asarray(processed["phase_corrected_mod360"], dtype=float),
                    "corrected_phase_deg": phase_corrected_unwrapped_sorted[inv],
                    "corrected_amp": np.asarray(scan.amplitude(), dtype=float),
                    "correction_360_freqs": np.asarray(processed["correction_360_freqs"], dtype=float),
                    "correction_360_phases_mod360": np.asarray(processed["correction_360_phases_mod360"], dtype=float),
                    "non_congruent_freqs": np.asarray(processed["non_congruent_freqs"], dtype=float),
                    "non_congruent_phases": np.asarray(processed["non_congruent_phases"], dtype=float),
                    "non_congruent_phases_mod360": np.asarray(processed["non_congruent_phases_mod360"], dtype=float),
                    "step1_non_congruent_freqs": step1_points["irregular_noncongruent_freqs"],
                    "settings": settings,
                }
            except Exception as exc:
                failed.append(f"{Path(scan.filename).name}: {exc}")
            if self.phase2_progress is not None:
                self.phase2_progress.configure(value=idx)
            if self.phase2_window is not None and self.phase2_window.winfo_exists():
                self.phase2_window.update_idletasks()
        self._phase2_render()
        if self.phase2_update_button is not None:
            self.phase2_update_button.configure(state="normal")
        self._phase2_set_attach_state(attached=False)
        if self.phase2_status_var is not None:
            self.phase2_status_var.set(f"Preview updated for {len(self.phase2_preview)} scan(s)." if not failed else f"Preview updated for {len(self.phase2_preview)} scan(s); {len(failed)} failed.")
        if failed:
            messagebox.showwarning("Phase Correction 2 preview", "\n".join(failed[:10]))

    def _phase2_on_auto_y_toggled(self) -> None:
        if self.phase2_auto_y_var is not None and bool(self.phase2_auto_y_var.get()):
            self._phase2_autoscale_all_y()

    def _phase2_autoscale_y_for_visible_x(self, ax) -> None:
        if self.phase2_auto_y_var is None or not bool(self.phase2_auto_y_var.get()):
            return
        lines = [ln for ln in ax.get_lines() if ln.get_visible()]
        collections = [col for col in ax.collections if col.get_visible()]
        x0, x1 = ax.get_xlim()
        lo, hi = (x0, x1) if x0 <= x1 else (x1, x0)
        y_chunks = []
        for ln in lines:
            x = np.asarray(ln.get_xdata(), dtype=float)
            y = np.asarray(ln.get_ydata(), dtype=float)
            mask = np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi)
            if np.any(mask):
                y_chunks.append(y[mask])
        for col in collections:
            offsets = np.asarray(col.get_offsets(), dtype=float)
            if offsets.ndim == 2 and offsets.shape[1] == 2 and offsets.shape[0] > 0:
                x = offsets[:, 0]
                y = offsets[:, 1]
                mask = np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi)
                if np.any(mask):
                    y_chunks.append(y[mask])
        if y_chunks:
            y_all = np.concatenate(y_chunks)
            pad = 1.0 if float(np.max(y_all)) <= float(np.min(y_all)) else 0.05 * (float(np.max(y_all)) - float(np.min(y_all)))
            ax.set_ylim(float(np.min(y_all)) - pad, float(np.max(y_all)) + pad)

    def _phase2_autoscale_all_y(self) -> None:
        if self.phase2_figure is None or self.phase2_canvas is None or self.phase2_auto_y_var is None or not bool(self.phase2_auto_y_var.get()):
            return
        for ax in self.phase2_figure.axes:
            self._phase2_autoscale_y_for_visible_x(ax)
        self.phase2_canvas.draw_idle()

    def _phase2_reset_view(self) -> None:
        if self.phase2_figure is None or self.phase2_canvas is None:
            return
        for ax in self.phase2_figure.axes:
            ax.relim()
            ax.autoscale(enable=True, axis="both", tight=False)
        self.phase2_canvas.draw_idle()

    def _phase2_render(self) -> None:
        if self.phase2_figure is None or self.phase2_canvas is None:
            return
        saved = [(ax.get_xlim(), ax.get_ylim()) for ax in self.phase2_figure.axes]
        scans = self._selected_scans()
        self.phase2_figure.clear()
        axes = np.atleast_1d(self.phase2_figure.subplots(len(scans), 1, sharex=False))
        plot_mod360 = self.phase2_mod360_var is not None and bool(self.phase2_mod360_var.get())
        for i, scan in enumerate(scans):
            ax = axes[i]
            prev = self.phase2_preview.get(self._scan_key(scan))
            if prev is None:
                ax.text(0.5, 0.5, "No preview", ha="center", va="center")
                ax.axis("off")
                continue
            y_initial = prev["phase_input_mod360_sorted"] if plot_mod360 else prev["phase_input_sorted"]
            y_final = prev["phase_corrected_mod360_sorted"] if plot_mod360 else prev["phase_corrected_sorted"]
            x_ghz = np.asarray(prev["freq_sorted"], dtype=float) / 1.0e9
            ax.plot(x_ghz, y_final, color="darkblue", linewidth=1.0, alpha=0.95, label="Final corrected", zorder=2)
            ax.plot(
                x_ghz,
                y_initial,
                color="darkorange",
                linewidth=1.4,
                linestyle="--",
                alpha=0.95,
                label="Input from Phase Correction 1",
                zorder=3,
            )
            if prev["non_congruent_freqs"].size > 0:
                freq_pts = prev["step1_non_congruent_freqs"]
                y_pts = self._phase2_values_at_freqs(
                    freq_pts,
                    prev["freq_sorted"],
                    y_initial,
                )
                ax.scatter(np.asarray(freq_pts, dtype=float) / 1.0e9, y_pts, s=16, color="blue", label="Other phase discontinuities", zorder=4)
            ax.set_ylabel("Phase mod 360 (deg)" if plot_mod360 else "Phase (deg)")
            if plot_mod360:
                ax.yaxis.set_major_locator(MultipleLocator(90))
                ax.yaxis.set_minor_locator(MultipleLocator(10))
            else:
                ax.yaxis.set_major_locator(AutoLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.grid(True, which="major", alpha=0.45)
            ax.grid(True, which="minor", alpha=0.2)
            ax.set_title(Path(scan.filename).name, fontsize=9)
            if i == 0:
                ax.legend(loc="upper right", fontsize=8)
            if i == len(scans) - 1:
                ax.set_xlabel("Frequency (GHz)")
        if len(saved) == len(axes):
            for ax, (xlim, ylim) in zip(axes, saved):
                ax.set_xlim(xlim)
                if self.phase2_auto_y_var is not None and bool(self.phase2_auto_y_var.get()):
                    self._phase2_autoscale_y_for_visible_x(ax)
                else:
                    ax.set_ylim(ylim)
        for ax in axes:
            ax.callbacks.connect("xlim_changed", lambda changed_ax: self._phase2_autoscale_y_for_visible_x(changed_ax))
        settings = next((prev.get("settings") for prev in self.phase2_preview.values() if isinstance(prev, dict)), None)
        title = "Phase Correction 2"
        if isinstance(settings, dict):
            title += f" | threshold={settings['threshold_deg']:.1f} deg | max_passes={settings['max_passes']} | min_sep={settings['min_separation_hz'] / 1e3:.1f} kHz"
        self.phase2_figure.suptitle(title, fontsize=11)
        self.phase2_figure.tight_layout()
        self.phase2_canvas.draw_idle()

    def _phase2_attach(self) -> None:
        scans = self._selected_scans()
        if not scans:
            return
        attached_at = datetime.now().isoformat(timespec="seconds")
        count = 0
        for scan in scans:
            prev = self.phase2_preview.get(self._scan_key(scan))
            if prev is None:
                continue
            settings = prev.get("settings", {})
            scan.candidate_resonators["phase_correction_2"] = {
                "attached_at": attached_at,
                "attached_by": _current_user(),
                "threshold_deg": float(settings.get("threshold_deg", DEFAULT_PHASE2_THRESHOLD_DEG)),
                "max_passes": int(settings.get("max_passes", DEFAULT_PHASE2_MAX_PASSES)),
                "correct_other_discontinuities": bool(settings.get("correct_other_discontinuities", DEFAULT_PHASE2_CORRECT_OTHER_DISCONTINUITIES)),
                "min_separation_hz": float(settings.get("min_separation_hz", DEFAULT_PHASE2_MIN_SEPARATION_HZ)),
                "p_random_cutoff": float(settings.get("p_random_cutoff", DEFAULT_PHASE2_P_RANDOM_CUTOFF)),
                "corrected_amp": np.asarray(prev["corrected_amp"], dtype=float),
                "corrected_phase_deg": np.asarray(prev["corrected_phase_deg"], dtype=float),
                "data_polar": np.vstack((scan.freq, np.asarray(prev["corrected_amp"], dtype=float), np.asarray(prev["corrected_phase_deg"], dtype=float))),
                "data_polar_format": "(3, N) rows = [freq, amplitude, unwrapped_phase_deg]",
            }
            scan.processing_history.append(_make_event("attach_phase_correction_2", {"threshold_deg": float(settings.get("threshold_deg", DEFAULT_PHASE2_THRESHOLD_DEG)), "max_passes": int(settings.get("max_passes", DEFAULT_PHASE2_MAX_PASSES)), "correct_other_discontinuities": bool(settings.get("correct_other_discontinuities", DEFAULT_PHASE2_CORRECT_OTHER_DISCONTINUITIES)), "min_separation_hz": float(settings.get("min_separation_hz", DEFAULT_PHASE2_MIN_SEPARATION_HZ)), "p_random_cutoff": float(settings.get("p_random_cutoff", DEFAULT_PHASE2_P_RANDOM_CUTOFF))}))
            count += 1
        self.dataset.processing_history.append(_make_event("attach_phase_correction_2_selected", {"selected_count": count}))
        self._mark_dirty()
        self._refresh_status()
        self._phase2_set_attach_state(attached=True)
        if self.phase2_status_var is not None:
            self.phase2_status_var.set(f"Attached phase correction 2 to {count} selected scan(s).")
        self._log(f"Attached phase correction 2 to {count} selected scan(s).")
        self._autosave_dataset()

    def _phase2_close(self) -> None:
        if self.phase2_window is not None and self.phase2_window.winfo_exists():
            self.phase2_window.destroy()
        self.phase2_window = None
        self.phase2_canvas = None
        self.phase2_toolbar = None
        self.phase2_figure = None
        self.phase2_threshold_slider = None
        self.phase2_max_passes_slider = None
        self.phase2_min_sep_slider = None
        self.phase2_correct_other_var = None
        self.phase2_p_random_var = None
        self.phase2_p_random_entry = None
        self.phase2_auto_y_var = None
        self.phase2_mod360_var = None
        self.phase2_status_var = None
        self.phase2_progress = None
        self.phase2_update_button = None
        self.phase2_attach_button = None
        self.phase2_preview = {}
