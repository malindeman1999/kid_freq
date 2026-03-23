from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter import messagebox

from ..analysis_models import _current_user, _make_event

DEFAULT_PHASE2_THRESHOLD_DEG = 100.0


def _find_expected_diff(source_diffs: np.ndarray, i: int, threshold_deg: float) -> tuple[float, bool]:
    left_idx = None
    for j in range(i - 1, -1, -1):
        if abs(source_diffs[j]) <= threshold_deg:
            left_idx = j
            break

    right_idx = None
    for j in range(i + 1, source_diffs.size):
        if abs(source_diffs[j]) <= threshold_deg:
            right_idx = j
            break

    if left_idx is None or right_idx is None:
        return 0.0, False

    left_val = float(source_diffs[left_idx])
    right_val = float(source_diffs[right_idx])
    span = right_idx - left_idx
    if span <= 0:
        return 0.0, False

    t = (i - left_idx) / span
    return left_val + t * (right_val - left_val), True


def _simple_phase2_correction(phase_deg: np.ndarray, threshold_deg: float) -> dict:
    phase = np.asarray(phase_deg, dtype=float).ravel()
    if phase.size <= 1:
        return {
            "phase_in": phase.copy(),
            "phase_out": phase.copy(),
            "corrected_idx": np.empty((0,), dtype=int),
            "phase_out_mod360": np.mod(phase, 360.0),
        }

    source_diffs = np.diff(phase)
    corrected_diffs = source_diffs.copy()
    corrected_idx = []

    for i in range(source_diffs.size):
        expected, valid = _find_expected_diff(source_diffs, i, threshold_deg)
        if not valid:
            continue
        if abs(source_diffs[i] - expected) > threshold_deg:
            corrected_diffs[i] = expected
            corrected_idx.append(i + 1)

    phase_out = np.empty_like(phase, dtype=float)
    phase_out[0] = phase[0]
    phase_out[1:] = phase[0] + np.cumsum(corrected_diffs)
    return {
        "phase_in": phase,
        "phase_out": phase_out,
        "corrected_idx": np.asarray(corrected_idx, dtype=int),
        "phase_out_mod360": np.mod(phase_out, 360.0),
    }


class SecondPhaseCorrectionMixin:
    def open_second_phase_correction_window(self) -> None:
        scans = self._selected_scans()
        if not scans:
            messagebox.showwarning("No selection", "Select scans for analysis first.")
            return
        if any(not scan.has_dewrapped_phase() for scan in scans):
            messagebox.showwarning(
                "Missing phase correction",
                "Run the first Phase Correction step and click Attach for all selected scans.",
            )
            return
        if self.phase2_window is not None and self.phase2_window.winfo_exists():
            self.phase2_window.lift()
            return

        self.phase2_window = tk.Toplevel(self.root)
        self.phase2_window.title("Phase Correction 3")
        self.phase2_window.geometry("1280x900")
        self.phase2_window.protocol("WM_DELETE_WINDOW", self._phase2_close)

        controls = tk.Frame(self.phase2_window, padx=8, pady=8)
        controls.pack(side="top", fill="x")

        self.phase2_threshold_slider = tk.Scale(
            controls,
            from_=1.0,
            to=300.0,
            resolution=1.0,
            orient="horizontal",
            label="Threshold (deg)",
            command=lambda _v: self._phase2_on_control_changed(),
            length=260,
        )
        self.phase2_threshold_slider.set(DEFAULT_PHASE2_THRESHOLD_DEG)
        self.phase2_threshold_slider.pack(side="left", padx=(0, 12))
        self.phase2_threshold_slider.bind("<ButtonRelease-1>", self._phase2_on_control_released)
        self.phase2_threshold_slider.bind("<KeyRelease>", self._phase2_on_control_released)

        self.phase2_auto_y_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            controls,
            text="Auto-scale phase in window",
            variable=self.phase2_auto_y_var,
            command=self._phase2_on_auto_y_toggled,
        ).pack(side="left", padx=(0, 12))

        self.phase2_status_var = tk.StringVar(
            value="Adjust threshold and release slider to update. Attach to save corrected amplitude+phase output."
        )
        tk.Label(controls, textvariable=self.phase2_status_var, anchor="w").pack(
            side="left", fill="x", expand=True
        )

        actions = tk.Frame(self.phase2_window, padx=8, pady=6)
        actions.pack(side="top", fill="x")
        tk.Button(actions, text="Cancel", width=12, command=self._phase2_close).pack(side="right")
        tk.Button(actions, text="Reset View", width=12, command=self._phase2_reset_view).pack(
            side="right", padx=(8, 0)
        )
        self.phase2_attach_button = tk.Button(
            actions,
            text="Attach, Save, and Close",
            width=24,
            command=self._attach_save_and_close_phase2,
        )
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
        self._phase2_update_preview()

    def _phase2_set_attach_state(self, attached: bool) -> None:
        if self.phase2_attach_button is None:
            return
        self.phase2_attach_button.configure(
            bg="light green" if attached else "pink",
            activebackground="light green" if attached else "pink",
        )

    def _phase2_on_control_changed(self) -> None:
        self._phase2_set_attach_state(attached=False)
        if self.phase2_status_var is not None:
            self.phase2_status_var.set("Settings changed. Release slider to update preview.")

    def _phase2_on_control_released(self, _event: tk.Event) -> None:
        self._phase2_on_control_changed()
        self._phase2_update_preview()

    def _phase2_update_preview(self) -> None:
        scans = self._selected_scans()
        if not scans:
            return
        if self.phase2_threshold_slider is None:
            return

        threshold_deg = float(self.phase2_threshold_slider.get())
        self.phase2_preview = {}
        failed = []
        for scan in scans:
            try:
                phase_in = np.asarray(scan.phase_deg_unwrapped(), dtype=float)
                corrected = _simple_phase2_correction(phase_in, threshold_deg)
                phase_out = corrected["phase_out"]
                self.phase2_preview[self._scan_key(scan)] = {
                    "corrected_amp": np.asarray(scan.amplitude(), dtype=float),
                    "phase_in": corrected["phase_in"],
                    "phase_out": phase_out,
                    "phase_out_mod360": corrected["phase_out_mod360"],
                    "corrected_idx": corrected["corrected_idx"],
                    "threshold_deg": threshold_deg,
                }
            except Exception as exc:
                failed.append(f"{Path(scan.filename).name}: {exc}")

        self._phase2_render()
        self._phase2_set_attach_state(attached=False)
        if self.phase2_status_var is not None:
            ok_count = len(self.phase2_preview)
            if failed:
                self.phase2_status_var.set(
                    f"Preview updated for {ok_count} scan(s); {len(failed)} failed."
                )
            else:
                self.phase2_status_var.set(f"Preview updated for {ok_count} scan(s).")
        if failed:
            messagebox.showwarning("Phase Correction 3 preview", "\n".join(failed[:10]))

    def _phase2_on_auto_y_toggled(self) -> None:
        if self.phase2_auto_y_var is not None and bool(self.phase2_auto_y_var.get()):
            self._phase2_autoscale_all_y()

    def _phase2_autoscale_y_for_visible_x(self, ax) -> None:
        if self.phase2_auto_y_var is None or not bool(self.phase2_auto_y_var.get()):
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

    def _phase2_autoscale_all_y(self) -> None:
        if self.phase2_figure is None or self.phase2_canvas is None:
            return
        if self.phase2_auto_y_var is None or not bool(self.phase2_auto_y_var.get()):
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
        if not scans:
            ax = self.phase2_figure.add_subplot(111)
            ax.text(0.5, 0.5, "No selected scans", ha="center", va="center")
            ax.axis("off")
            self.phase2_canvas.draw_idle()
            return

        axes = np.atleast_1d(self.phase2_figure.subplots(len(scans), 1, sharex=False))
        for i, scan in enumerate(scans):
            freq_ghz = np.asarray(scan.freq, dtype=float) / 1.0e9
            ax = axes[i]
            prev = self.phase2_preview.get(self._scan_key(scan))
            if prev is None:
                ax.text(0.5, 0.5, "No preview", ha="center", va="center")
                ax.axis("off")
                continue
            ax.plot(freq_ghz, prev["phase_in"], color="0.75", linewidth=0.8, label="Input phase (step 1)")
            ax.plot(freq_ghz, prev["phase_out"], color="tab:blue", linewidth=1.0, label="Corrected phase (step 2)")
            idx = np.asarray(prev["corrected_idx"], dtype=int)
            if idx.size > 0:
                ax.scatter(
                    freq_ghz[idx],
                    prev["phase_out"][idx],
                    s=14,
                    color="tab:red",
                    label="Corrected points",
                    zorder=3,
                )
            ax.set_ylabel("Phase (deg)")
            ax.grid(True, alpha=0.3)
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
            ax.callbacks.connect(
                "xlim_changed", lambda changed_ax: self._phase2_autoscale_y_for_visible_x(changed_ax)
            )

        th = float(self.phase2_threshold_slider.get()) if self.phase2_threshold_slider is not None else 0.0
        self.phase2_figure.suptitle(
            f"Phase Correction 3 | threshold={th:.1f} deg | simple diff correction",
            fontsize=11,
        )
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
            scan.candidate_resonators["phase_correction_2"] = {
                "attached_at": attached_at,
                "attached_by": _current_user(),
                "threshold_deg": float(prev["threshold_deg"]),
                "corrected_amp": np.asarray(prev["corrected_amp"], dtype=float),
                "corrected_phase_deg": np.asarray(prev["phase_out"], dtype=float),
                "data_polar": np.vstack(
                    (
                        scan.freq,
                        np.asarray(prev["corrected_amp"], dtype=float),
                        np.asarray(prev["phase_out"], dtype=float),
                    )
                ),
                "data_polar_format": "(3, N) rows = [freq, amplitude, unwrapped_phase_deg]",
            }
            scan.processing_history.append(
                _make_event(
                    "attach_phase_correction_2",
                    {
                        "threshold_deg": float(prev["threshold_deg"]),
                        "corrected_points": int(np.asarray(prev["corrected_idx"]).size),
                        "attached_at": attached_at,
                        "attached_by": _current_user(),
                    },
                )
            )
            count += 1

        self.dataset.processing_history.append(
            _make_event("attach_phase_correction_2_selected", {"selected_count": count})
        )
        self._mark_dirty()
        self._refresh_status()
        self._phase2_set_attach_state(attached=True)
        if self.phase2_status_var is not None:
            self.phase2_status_var.set(f"Attached phase correction 2 output to {count} selected scan(s).")
        self._log(f"Attached phase correction 2 output to {count} selected scan(s).")
        self._autosave_dataset()

    def _phase2_close(self) -> None:
        if self.phase2_window is not None and self.phase2_window.winfo_exists():
            self.phase2_window.destroy()
        self.phase2_window = None
        self.phase2_canvas = None
        self.phase2_toolbar = None
        self.phase2_figure = None
        self.phase2_threshold_slider = None
        self.phase2_auto_y_var = None
        self.phase2_status_var = None
        self.phase2_attach_button = None
        self.phase2_preview = {}
