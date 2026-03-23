from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import AutoLocator, AutoMinorLocator, MultipleLocator
from tkinter import messagebox, ttk

from ..analysis_models import _current_user, _make_event
from phase_correction import process_phase_data

DEFAULT_PHASE_THRESHOLD_DEG = 10.0
DEFAULT_APPLY_EXACT_360 = True
DEFAULT_CORRECT_CONGRUENT = True
DEFAULT_MAX_PASSES = 3
DEFAULT_MIN_SEPARATION_HZ = 15e3
DEFAULT_P_RANDOM_CUTOFF = 1e-3


class UnwrapPhaseMixin:
    def open_unwrap_phase_window(self) -> None:
        scans = self._selected_scans()
        if not scans:
            messagebox.showwarning("No selection", "Select scans for analysis first.")
            return
        if self.unwrap_window is not None and self.unwrap_window.winfo_exists():
            self.unwrap_window.lift()
            return

        self.unwrap_window = tk.Toplevel(self.root)
        self.unwrap_window.title("Phase Correction")
        self.unwrap_window.geometry("1280x900")
        self.unwrap_window.protocol("WM_DELETE_WINDOW", self._unwrap_close)

        controls = tk.Frame(self.unwrap_window, padx=8, pady=8)
        controls.pack(side="top", fill="x")

        self.unwrap_threshold_slider = tk.Scale(
            controls,
            from_=1.0,
            to=120.0,
            resolution=1.0,
            orient="horizontal",
            label="Threshold (deg)",
            command=lambda _v: self._unwrap_on_control_changed(),
            length=220,
        )
        self.unwrap_threshold_slider.set(DEFAULT_PHASE_THRESHOLD_DEG)
        self.unwrap_threshold_slider.grid(row=0, column=0, padx=(0, 8), sticky="w")
        self.unwrap_threshold_slider.bind("<ButtonRelease-1>", self._unwrap_on_control_released)
        self.unwrap_threshold_slider.bind("<KeyRelease>", self._unwrap_on_control_released)

        self.unwrap_max_passes_slider = tk.Scale(
            controls,
            from_=1,
            to=8,
            resolution=1,
            orient="horizontal",
            label="Max Passes",
            command=lambda _v: self._unwrap_on_control_changed(),
            length=180,
        )
        self.unwrap_max_passes_slider.set(DEFAULT_MAX_PASSES)
        self.unwrap_max_passes_slider.grid(row=0, column=1, padx=(0, 8), sticky="w")
        self.unwrap_max_passes_slider.bind("<ButtonRelease-1>", self._unwrap_on_control_released)
        self.unwrap_max_passes_slider.bind("<KeyRelease>", self._unwrap_on_control_released)

        self.unwrap_min_sep_slider = tk.Scale(
            controls,
            from_=1.0,
            to=500.0,
            resolution=1.0,
            orient="horizontal",
            label="Min Separation (kHz)",
            command=lambda _v: self._unwrap_on_control_changed(),
            length=220,
        )
        self.unwrap_min_sep_slider.set(DEFAULT_MIN_SEPARATION_HZ / 1e3)
        self.unwrap_min_sep_slider.grid(row=0, column=2, padx=(0, 8), sticky="w")
        self.unwrap_min_sep_slider.bind("<ButtonRelease-1>", self._unwrap_on_control_released)
        self.unwrap_min_sep_slider.bind("<KeyRelease>", self._unwrap_on_control_released)

        self.unwrap_apply_exact_360_var = tk.BooleanVar(value=DEFAULT_APPLY_EXACT_360)
        tk.Checkbutton(
            controls,
            text="Snap near 2*pi phase wraps",
            variable=self.unwrap_apply_exact_360_var,
            command=self._unwrap_on_toggle_changed,
        ).grid(row=0, column=3, padx=(0, 12), sticky="w")

        self.unwrap_correct_congruent_var = tk.BooleanVar(value=DEFAULT_CORRECT_CONGRUENT)
        tk.Checkbutton(
            controls,
            text="Correct VNA phase errors",
            variable=self.unwrap_correct_congruent_var,
            command=self._unwrap_on_toggle_changed,
        ).grid(row=0, column=4, padx=(0, 12), sticky="w")

        self.unwrap_auto_y_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            controls,
            text="Auto-scale phase in window",
            variable=self.unwrap_auto_y_var,
            command=self._unwrap_on_auto_y_toggled,
        ).grid(row=0, column=5, padx=(0, 12), sticky="w")
        self.unwrap_mod360_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            controls,
            text="Plot mod 360",
            variable=self.unwrap_mod360_var,
            command=self._unwrap_on_toggle_changed,
        ).grid(row=0, column=6, padx=(0, 12), sticky="w")

        tk.Label(controls, text="p-random cutoff:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.unwrap_p_random_var = tk.StringVar(value=f"{DEFAULT_P_RANDOM_CUTOFF:g}")
        self.unwrap_p_random_entry = tk.Entry(controls, textvariable=self.unwrap_p_random_var, width=12)
        self.unwrap_p_random_entry.grid(row=1, column=1, sticky="w", pady=(6, 0))

        self.unwrap_update_button = tk.Button(
            controls,
            text="Update Preview",
            width=16,
            command=self._unwrap_update_preview,
        )
        self.unwrap_update_button.grid(row=1, column=2, sticky="w", pady=(6, 0))

        self.unwrap_status_var = tk.StringVar(value="Opening window and preparing preview...")
        tk.Label(controls, textvariable=self.unwrap_status_var, anchor="w").grid(
            row=1, column=3, columnspan=2, sticky="we", pady=(6, 0)
        )
        self.unwrap_progress = ttk.Progressbar(
            controls,
            orient="horizontal",
            mode="determinate",
            length=280,
        )
        self.unwrap_progress.grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))

        actions = tk.Frame(self.unwrap_window, padx=8, pady=6)
        actions.pack(side="top", fill="x")
        tk.Button(actions, text="Cancel", width=12, command=self._unwrap_close).pack(side="right")
        tk.Button(actions, text="Reset View", width=12, command=self._unwrap_reset_view).pack(
            side="right", padx=(8, 0)
        )
        self.unwrap_attach_button = tk.Button(
            actions,
            text="Attach, Save, and Close",
            width=24,
            command=self._attach_save_and_close_unwrap,
        )
        self.unwrap_attach_button.pack(side="right", padx=(8, 0))
        self._unwrap_set_attach_state(attached=False)

        self.unwrap_figure = Figure(figsize=(12, 7))
        self.unwrap_canvas = FigureCanvasTkAgg(self.unwrap_figure, master=self.unwrap_window)
        self.unwrap_toolbar = NavigationToolbar2Tk(self.unwrap_canvas, self.unwrap_window)
        self.unwrap_toolbar.update()
        self.unwrap_toolbar.pack(side="top", fill="x")
        self.unwrap_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.unwrap_canvas.mpl_connect("button_release_event", lambda _e: self._unwrap_autoscale_all_y())

        self.unwrap_preview = {}
        self.unwrap_window.update_idletasks()
        self.unwrap_window.after(10, self._unwrap_update_preview)

    def _unwrap_get_settings(self) -> dict:
        if self.unwrap_threshold_slider is None or self.unwrap_max_passes_slider is None:
            raise ValueError("Phase correction controls are not initialized.")
        threshold_deg = float(self.unwrap_threshold_slider.get())
        max_passes = int(self.unwrap_max_passes_slider.get())
        min_sep_khz = float(self.unwrap_min_sep_slider.get()) if self.unwrap_min_sep_slider is not None else 15.0
        p_cutoff_text = self.unwrap_p_random_var.get().strip() if self.unwrap_p_random_var is not None else "1e-3"
        try:
            p_random_cutoff = float(p_cutoff_text)
        except ValueError as exc:
            raise ValueError("p-random cutoff must be numeric (e.g., 1e-3).") from exc
        if p_random_cutoff <= 0.0:
            raise ValueError("p-random cutoff must be > 0.")
        return {
            "threshold_deg": threshold_deg,
            "max_passes": max_passes,
            "apply_exact_360": bool(self.unwrap_apply_exact_360_var.get()) if self.unwrap_apply_exact_360_var is not None else True,
            "correct_congruent": bool(self.unwrap_correct_congruent_var.get()) if self.unwrap_correct_congruent_var is not None else True,
            "min_separation_hz": min_sep_khz * 1e3,
            "p_random_cutoff": p_random_cutoff,
        }

    def _unwrap_set_attach_state(self, attached: bool) -> None:
        if self.unwrap_attach_button is None:
            return
        self.unwrap_attach_button.configure(
            bg="light green" if attached else "pink",
            activebackground="light green" if attached else "pink",
        )

    def _unwrap_on_control_changed(self, *_args) -> None:
        self._unwrap_set_attach_state(attached=False)
        if self.unwrap_status_var is not None:
            self.unwrap_status_var.set("Settings changed. Click Update Preview.")

    def _unwrap_on_control_released(self, _event: tk.Event) -> None:
        self._unwrap_on_control_changed()

    def _unwrap_on_toggle_changed(self) -> None:
        self._unwrap_on_control_changed()
        self._unwrap_update_preview()
        self._unwrap_reset_view()

    def _unwrap_update_preview(self) -> None:
        scans = self._selected_scans()
        if not scans:
            return

        try:
            settings = self._unwrap_get_settings()
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        self.unwrap_preview = {}
        if self.unwrap_update_button is not None:
            self.unwrap_update_button.configure(state="disabled")
        if self.unwrap_progress is not None:
            self.unwrap_progress.configure(maximum=max(len(scans), 1), value=0)
        failed = []
        for idx, scan in enumerate(scans, start=1):
            if self.unwrap_status_var is not None:
                self.unwrap_status_var.set(
                    f"Processing phase correction preview: {idx}/{len(scans)} | {Path(scan.filename).name}"
                )
            if self.unwrap_progress is not None:
                self.unwrap_progress.configure(value=idx - 1)
            if self.unwrap_window is not None and self.unwrap_window.winfo_exists():
                self.unwrap_window.update_idletasks()
            try:
                freq = np.asarray(scan.freq, dtype=float)
                s21 = np.asarray(scan.s21_complex_raw, dtype=complex)
                order = np.argsort(freq)
                inv = np.empty_like(order)
                inv[order] = np.arange(order.size)
                freq_sorted = freq[order]
                s21_sorted = s21[order]

                processed = process_phase_data(
                    freq_sorted,
                    s21_sorted,
                    threshold_deg=settings["threshold_deg"],
                    apply_exact_360=settings["apply_exact_360"],
                    max_passes=settings["max_passes"],
                    min_separation_hz=settings["min_separation_hz"],
                    p_random_cutoff=settings["p_random_cutoff"],
                    correct_congruent=settings["correct_congruent"],
                    correct_non_congruent=False,
                    verbose=False,
                )

                corrected_sorted = np.asarray(processed["phase_corrected"], dtype=float)
                corrected_unsorted = corrected_sorted[inv]
                self.unwrap_preview[self._scan_key(scan)] = {
                    "freq_sorted": freq_sorted,
                    "phase_corrected_initial_sorted": np.asarray(processed["phase_corrected_initial"], dtype=float),
                    "phase_corrected_initial_mod360_sorted": np.asarray(
                        processed["phase_corrected_initial_mod360"], dtype=float
                    ),
                    "phase_corrected_sorted": corrected_sorted,
                    "phase_corrected_mod360_sorted": np.asarray(
                        processed["phase_corrected_mod360"], dtype=float
                    ),
                    "phase_corrected_unsorted": corrected_unsorted,
                    "correction_360_freqs": np.asarray(processed["correction_360_freqs"], dtype=float),
                    "correction_360_phases_mod360": np.asarray(
                        processed["correction_360_phases_mod360"], dtype=float
                    ),
                    "congruent_freqs": np.asarray(processed["congruent_freqs"], dtype=float),
                    "congruent_phases": np.asarray(
                        processed["congruent_phases"], dtype=float
                    ),
                    "congruent_phases_mod360": np.asarray(
                        processed["congruent_phases_mod360"], dtype=float
                    ),
                    "non_congruent_freqs": np.asarray(processed["non_congruent_freqs"], dtype=float),
                    "non_congruent_phases": np.asarray(
                        processed["non_congruent_phases"], dtype=float
                    ),
                    "non_congruent_phases_mod360": np.asarray(
                        processed["non_congruent_phases_mod360"], dtype=float
                    ),
                    "settings": settings,
                }
            except Exception as exc:
                failed.append(f"{Path(scan.filename).name}: {exc}")
            if self.unwrap_progress is not None:
                self.unwrap_progress.configure(value=idx)
            if self.unwrap_window is not None and self.unwrap_window.winfo_exists():
                self.unwrap_window.update_idletasks()

        self._unwrap_render()
        if self.unwrap_update_button is not None:
            self.unwrap_update_button.configure(state="normal")
        self._unwrap_set_attach_state(attached=False)
        if self.unwrap_status_var is not None:
            ok_count = len(self.unwrap_preview)
            if failed:
                self.unwrap_status_var.set(
                    f"Preview updated for {ok_count} scan(s); {len(failed)} failed."
                )
            else:
                self.unwrap_status_var.set(f"Preview updated for {ok_count} scan(s).")
        if failed:
            messagebox.showwarning("Phase correction preview", "\n".join(failed[:10]))

    def _unwrap_on_auto_y_toggled(self) -> None:
        if self.unwrap_auto_y_var is not None and bool(self.unwrap_auto_y_var.get()):
            self._unwrap_autoscale_all_y()

    def _unwrap_autoscale_y_for_visible_x(self, ax) -> None:
        if self.unwrap_auto_y_var is None or not bool(self.unwrap_auto_y_var.get()):
            return
        lines = [ln for ln in ax.get_lines() if ln.get_visible()]
        collections = [col for col in ax.collections if col.get_visible()]
        if not lines and not collections:
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
        for col in collections:
            offsets = np.asarray(col.get_offsets(), dtype=float)
            if offsets.ndim != 2 or offsets.shape[1] != 2 or offsets.shape[0] == 0:
                continue
            x = offsets[:, 0]
            y = offsets[:, 1]
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

    def _unwrap_autoscale_all_y(self) -> None:
        if self.unwrap_figure is None or self.unwrap_canvas is None:
            return
        if self.unwrap_auto_y_var is None or not bool(self.unwrap_auto_y_var.get()):
            return
        for ax in self.unwrap_figure.axes:
            self._unwrap_autoscale_y_for_visible_x(ax)
        self.unwrap_canvas.draw_idle()

    def _unwrap_reset_view(self) -> None:
        if self.unwrap_figure is None or self.unwrap_canvas is None:
            return
        for ax in self.unwrap_figure.axes:
            ax.relim()
            ax.autoscale(enable=True, axis="both", tight=False)
        self.unwrap_canvas.draw_idle()

    def _unwrap_render(self) -> None:
        if self.unwrap_figure is None or self.unwrap_canvas is None:
            return
        saved = [(ax.get_xlim(), ax.get_ylim()) for ax in self.unwrap_figure.axes]
        scans = self._selected_scans()
        self.unwrap_figure.clear()
        if not scans:
            ax = self.unwrap_figure.add_subplot(111)
            ax.text(0.5, 0.5, "No selected scans", ha="center", va="center")
            ax.axis("off")
            self.unwrap_canvas.draw_idle()
            return

        axes = np.atleast_1d(self.unwrap_figure.subplots(len(scans), 1, sharex=False))
        plot_mod360 = self.unwrap_mod360_var is not None and bool(self.unwrap_mod360_var.get())
        for i, scan in enumerate(scans):
            ax = axes[i]
            prev = self.unwrap_preview.get(self._scan_key(scan))
            if prev is None:
                ax.text(0.5, 0.5, "No preview", ha="center", va="center")
                ax.axis("off")
                continue
            y_initial = prev["phase_corrected_initial_mod360_sorted"] if plot_mod360 else prev["phase_corrected_initial_sorted"]
            y_final = prev["phase_corrected_mod360_sorted"] if plot_mod360 else prev["phase_corrected_sorted"]
            ax.plot(
                np.asarray(prev["freq_sorted"], dtype=float) / 1.0e9,
                y_initial,
                color="lightgray",
                linewidth=0.7,
                alpha=0.9,
                label="Initial corrected (context)",
            )
            ax.plot(
                np.asarray(prev["freq_sorted"], dtype=float) / 1.0e9,
                y_final,
                color="darkblue",
                linewidth=0.9,
                label="Final corrected",
            )
            if prev["correction_360_freqs"].size > 0:
                y_pts = prev["correction_360_phases_mod360"] if plot_mod360 else y_final[np.searchsorted(np.asarray(prev["freq_sorted"], dtype=float), np.asarray(prev["correction_360_freqs"], dtype=float)).clip(0, len(y_final) - 1)]
                ax.scatter(
                    np.asarray(prev["correction_360_freqs"], dtype=float) / 1.0e9,
                    y_pts,
                    s=10,
                    color="black",
                    label="2*pi phase wrap corrections",
                    zorder=3,
                )
            if prev["non_congruent_freqs"].size > 0:
                y_pts = prev["non_congruent_phases_mod360"] if plot_mod360 else prev["non_congruent_phases"]
                ax.scatter(
                    np.asarray(prev["non_congruent_freqs"], dtype=float) / 1.0e9,
                    y_pts,
                    s=16,
                    color="blue",
                    label="Other phase discontinuities",
                    zorder=4,
                )
            if prev["congruent_freqs"].size > 0:
                y_pts = prev["congruent_phases_mod360"] if plot_mod360 else prev["congruent_phases"]
                ax.scatter(
                    np.asarray(prev["congruent_freqs"], dtype=float) / 1.0e9,
                    y_pts,
                    s=20,
                    color="pink",
                    label="VNA phase corrections",
                    zorder=5,
                )
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
                if self.unwrap_auto_y_var is not None and bool(self.unwrap_auto_y_var.get()):
                    self._unwrap_autoscale_y_for_visible_x(ax)
                else:
                    ax.set_ylim(ylim)
        for ax in axes:
            ax.callbacks.connect(
                "xlim_changed", lambda changed_ax: self._unwrap_autoscale_y_for_visible_x(changed_ax)
            )

        title = "Phase Correction"
        settings = None
        if scans:
            first = self.unwrap_preview.get(self._scan_key(scans[0]))
            if isinstance(first, dict):
                settings = first.get("settings")
        if isinstance(settings, dict):
            title = (
                "Phase Correction"
                f" | threshold={settings['threshold_deg']:.1f} deg"
                f" | max_passes={settings['max_passes']}"
                f" | min_sep={settings['min_separation_hz'] / 1e3:.1f} kHz"
            )
        self.unwrap_figure.suptitle(title, fontsize=11)
        self.unwrap_figure.tight_layout()
        self.unwrap_canvas.draw_idle()

    def _unwrap_attach(self) -> None:
        scans = self._selected_scans()
        if not scans:
            return
        count = 0
        attached_at = datetime.now().isoformat(timespec="seconds")
        for scan in scans:
            prev = self.unwrap_preview.get(self._scan_key(scan))
            if prev is None:
                continue
            scan.s21_phase_deg_unwrapped = np.asarray(prev["phase_corrected_unsorted"], dtype=float)
            settings = prev.get("settings", {})
            scan.candidate_resonators["phase_class_points"] = {
                "regular_freqs": np.asarray(prev["correction_360_freqs"], dtype=float),
                "irregular_congruent_freqs": np.asarray(prev["congruent_freqs"], dtype=float),
                "irregular_noncongruent_freqs": np.asarray(prev["non_congruent_freqs"], dtype=float),
                "attached_at": attached_at,
                "attached_by": _current_user(),
            }
            scan.processing_history.append(
                _make_event(
                    "attach_phase_correction",
                    {
                        "threshold_deg": float(settings.get("threshold_deg", DEFAULT_PHASE_THRESHOLD_DEG)),
                        "max_passes": int(settings.get("max_passes", DEFAULT_MAX_PASSES)),
                        "apply_exact_360": bool(settings.get("apply_exact_360", DEFAULT_APPLY_EXACT_360)),
                        "correct_congruent": bool(settings.get("correct_congruent", DEFAULT_CORRECT_CONGRUENT)),
                        "min_separation_hz": float(settings.get("min_separation_hz", DEFAULT_MIN_SEPARATION_HZ)),
                        "p_random_cutoff": float(settings.get("p_random_cutoff", DEFAULT_P_RANDOM_CUTOFF)),
                        "regular_count": int(np.asarray(prev["correction_360_freqs"]).size),
                        "irregular_congruent_count": int(np.asarray(prev["congruent_freqs"]).size),
                        "irregular_noncongruent_count": int(np.asarray(prev["non_congruent_freqs"]).size),
                        "attached_at": attached_at,
                        "attached_by": _current_user(),
                    },
                )
            )
            count += 1

        self.dataset.processing_history.append(
            _make_event("attach_phase_correction_selected", {"selected_count": count})
        )
        self._mark_dirty()
        self._refresh_status()
        self._unwrap_set_attach_state(attached=True)
        if self.unwrap_status_var is not None:
            self.unwrap_status_var.set(f"Attached phase correction to {count} selected scan(s).")
        self._log(f"Attached phase correction to {count} selected scan(s).")
        self._autosave_dataset()

    def _unwrap_close(self) -> None:
        if self.unwrap_window is not None and self.unwrap_window.winfo_exists():
            self.unwrap_window.destroy()
        self.unwrap_window = None
        self.unwrap_canvas = None
        self.unwrap_toolbar = None
        self.unwrap_figure = None
        self.unwrap_threshold_slider = None
        self.unwrap_max_passes_slider = None
        self.unwrap_min_sep_slider = None
        self.unwrap_apply_exact_360_var = None
        self.unwrap_correct_congruent_var = None
        self.unwrap_p_random_var = None
        self.unwrap_p_random_entry = None
        self.unwrap_auto_y_var = None
        self.unwrap_mod360_var = None
        self.unwrap_status_var = None
        self.unwrap_progress = None
        self.unwrap_update_button = None
        self.unwrap_attach_button = None
        self.unwrap_preview = {}
