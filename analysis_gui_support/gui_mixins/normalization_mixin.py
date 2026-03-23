from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter import messagebox

from ..analysis_models import _current_user, _make_event, _read_polar_series


class NormalizationMixin:
    def open_normalization_window(self) -> None:
        if not self._selected_scans_have_attached_interp_data():
            messagebox.showwarning(
                "Missing interp data",
                "Run pipeline in order:\n"
                "Phase Correction -> Baseline Filtering -> Interp+Smooth -> Normalize Baseline.\n\n"
                "All selected scans must have attached interpolation data first.",
            )
            return

        if self.norm_window is not None and self.norm_window.winfo_exists():
            self.norm_window.lift()
            return

        self.norm_window = tk.Toplevel(self.root)
        self.norm_window.title("Normalize Baseline")
        self.norm_window.geometry("1200x820")
        self.norm_window.protocol("WM_DELETE_WINDOW", self._norm_close)

        controls = tk.Frame(self.norm_window, padx=8, pady=8)
        controls.pack(side="top", fill="x")

        self.norm_status_var = tk.StringVar(
            value="Preview shows S21 / interpolated baseline. Click Attach to store."
        )
        tk.Label(controls, textvariable=self.norm_status_var, anchor="w").pack(
            side="left", fill="x", expand=True
        )

        action_frame = tk.Frame(self.norm_window, padx=8, pady=6)
        action_frame.pack(side="top", fill="x")
        tk.Button(action_frame, text="Cancel", width=12, command=self._norm_close).pack(side="right")
        self.norm_attach_button = tk.Button(
            action_frame,
            text="Attach, Save, and Close",
            width=24,
            command=self._attach_save_and_close_norm,
        )
        self.norm_attach_button.pack(side="right", padx=(8, 0))
        self._norm_set_attach_state(attached=False)

        self.norm_figure = Figure(figsize=(12, 7))
        self.norm_canvas = FigureCanvasTkAgg(self.norm_figure, master=self.norm_window)
        self.norm_toolbar = NavigationToolbar2Tk(self.norm_canvas, self.norm_window)
        self.norm_toolbar.update()
        self.norm_toolbar.pack(side="top", fill="x")
        self.norm_canvas.get_tk_widget().pack(fill="both", expand=True)

        self.norm_preview = {}
        self._norm_compute_preview()
        self._norm_render()
        self._log("Opened normalization window.")

    def _norm_set_attach_state(self, attached: bool) -> None:
        if self.norm_attach_button is None:
            return
        if attached:
            self.norm_attach_button.configure(bg="light green", activebackground="light green")
        else:
            self.norm_attach_button.configure(bg="pink", activebackground="pink")

    def _norm_compute_preview(self) -> None:
        scans = self._selected_scans()
        self.norm_preview = {}
        for scan in scans:
            interp = scan.baseline_filter.get("interp_smooth", {})
            interp_amp, interp_phase = _read_polar_series(
                interp,
                amplitude_key="interp_amp",
                phase_key="interp_phase",
            )
            if interp_amp.shape != scan.freq.shape or interp_phase.shape != scan.freq.shape:
                continue

            phase3 = scan.candidate_resonators.get("phase_correction_3", {})
            corrected_amp, corrected_phase = _read_polar_series(
                phase3,
                amplitude_key="corrected_amp",
                phase_key="corrected_phase_deg",
            )
            if corrected_amp.shape != scan.freq.shape or corrected_phase.shape != scan.freq.shape:
                continue

            with np.errstate(divide="ignore", invalid="ignore"):
                norm_amp = np.divide(
                    corrected_amp,
                    interp_amp,
                    out=np.full(corrected_amp.shape, np.nan, dtype=float),
                    where=np.abs(interp_amp) > 0,
                )
            norm_phase = corrected_phase - interp_phase
            self.norm_preview[self._scan_key(scan)] = {
                "norm_amp": norm_amp,
                "norm_phase_deg_unwrapped": norm_phase,
            }

    def _norm_render(self) -> None:
        if self.norm_figure is None or self.norm_canvas is None:
            return
        scans = self._selected_scans()
        if not scans:
            self.norm_figure.clear()
            self.norm_canvas.draw_idle()
            return

        n = len(scans)
        self.norm_figure.clear()
        axes = self.norm_figure.subplots(n, 2, sharex=False)
        axes = np.atleast_2d(axes)

        for i, scan in enumerate(scans):
            freq_ghz = np.asarray(scan.freq, dtype=float) / 1.0e9
            ax_a = axes[i, 0]
            ax_p = axes[i, 1]
            prev = self.norm_preview.get(self._scan_key(scan))
            if prev is None:
                ax_a.text(0.5, 0.5, "Missing interp data", ha="center", va="center")
                ax_p.text(0.5, 0.5, "Missing interp data", ha="center", va="center")
                ax_a.set_axis_off()
                ax_p.set_axis_off()
                continue

            ax_a.plot(freq_ghz, prev["norm_amp"], color="tab:blue", linewidth=1.0, label="|S21/interp|")
            ax_p.plot(
                freq_ghz,
                prev["norm_phase_deg_unwrapped"],
                color="tab:orange",
                linewidth=1.0,
                label="Phase(S21/interp)",
            )
            ax_a.set_ylabel("Normalized |S21|")
            ax_p.set_ylabel("Normalized Phase (deg)")
            ax_a.grid(True, alpha=0.3)
            ax_p.grid(True, alpha=0.3)
            ax_a.set_title(Path(scan.filename).name, fontsize=9)
            if i == 0:
                ax_a.legend(loc="upper right", fontsize=8)
                ax_p.legend(loc="upper right", fontsize=8)
            if i == n - 1:
                ax_a.set_xlabel("Frequency (GHz)")
                ax_p.set_xlabel("Frequency (GHz)")

        self.norm_figure.suptitle("Baseline Normalization: S21 / Interpolated Baseline", fontsize=11)
        self.norm_figure.tight_layout()
        self.norm_canvas.draw_idle()

    def _norm_attach(self) -> None:
        scans = self._selected_scans()
        if not scans:
            return

        attached_at = datetime.now().isoformat(timespec="seconds")
        count = 0
        overwritten = 0
        for scan in scans:
            prev = self.norm_preview.get(self._scan_key(scan))
            if prev is None:
                continue
            if isinstance(scan.baseline_filter.get("normalized"), dict) and scan.baseline_filter["normalized"]:
                overwritten += 1
            # Exactly one attached normalized result per scan; overwrite prior attachment.
            scan.baseline_filter["normalized"] = {}
            scan.baseline_filter["normalized"] = {
                "attached_at": attached_at,
                "attached_by": _current_user(),
                "source": "phase2_polar / interp_polar",
                "norm_amp": np.asarray(prev["norm_amp"], dtype=float),
                "norm_phase_deg_unwrapped": np.asarray(prev["norm_phase_deg_unwrapped"], dtype=float),
                "normalized_data_polar": np.vstack(
                    (scan.freq, prev["norm_amp"], prev["norm_phase_deg_unwrapped"])
                ),
                "normalized_data_polar_format": "(3, N) rows = [freq, amplitude, unwrapped_phase_deg]",
            }
            scan.processing_history.append(
                _make_event(
                    "attach_normalized_baseline",
                    {"source": "phase2_polar / interp_polar", "points": int(scan.freq.size)},
                )
            )
            count += 1

        self.dataset.processing_history.append(
            _make_event("attach_normalized_baseline_selected", {"selected_count": count})
        )
        self._mark_dirty()
        self._refresh_status()
        self._norm_set_attach_state(attached=True)
        if self.norm_status_var is not None:
            self.norm_status_var.set(
                f"Attached normalized data to {count} selected scan(s). Overwrote {overwritten}."
            )
        self._log(
            f"Attached normalized baseline data to {count} selected scan(s); overwrote {overwritten} prior attachment(s)."
        )
        self._autosave_dataset()

    def _norm_close(self) -> None:
        if self.norm_window is not None and self.norm_window.winfo_exists():
            self.norm_window.destroy()
        self.norm_window = None
        self.norm_canvas = None
        self.norm_toolbar = None
        self.norm_figure = None
        self.norm_status_var = None
        self.norm_attach_button = None
        self.norm_preview = {}
