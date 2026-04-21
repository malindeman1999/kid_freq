from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter import messagebox

from ..analysis_models import _current_user, _make_event, _read_polar_series


class NormalizationMixin:
    @staticmethod
    def _scan_freq_bounds_hz(scan) -> tuple[float, float]:
        freq = np.asarray(scan.freq, dtype=float)
        finite = freq[np.isfinite(freq)]
        if finite.size == 0:
            return 0.0, 0.0
        return float(np.min(finite)), float(np.max(finite))

    def _scan_span_mhz(self, scan) -> float:
        lo_hz, hi_hz = self._scan_freq_bounds_hz(scan)
        return max(0.0, hi_hz - lo_hz) / 1.0e6

    def _scan_is_contained_in(self, inner_scan, outer_scan) -> bool:
        inner_lo, inner_hi = self._scan_freq_bounds_hz(inner_scan)
        outer_lo, outer_hi = self._scan_freq_bounds_hz(outer_scan)
        return inner_lo >= outer_lo and inner_hi <= outer_hi

    @staticmethod
    def _store_normalized_payload(scan, *, norm_amp: np.ndarray, norm_phase: np.ndarray, attached_at: str, source: str) -> bool:
        bf = scan.baseline_filter if isinstance(scan.baseline_filter, dict) else {}
        overwritten = isinstance(bf.get("normalized"), dict) and bool(bf.get("normalized"))
        bf["normalized"] = {
            "attached_at": attached_at,
            "attached_by": _current_user(),
            "source": source,
            "norm_amp": np.asarray(norm_amp, dtype=float),
            "norm_phase_deg_unwrapped": np.asarray(norm_phase, dtype=float),
            "normalized_data_polar": np.vstack((scan.freq, norm_amp, norm_phase)),
            "normalized_data_polar_format": "(3, N) rows = [freq, amplitude, unwrapped_phase_deg]",
        }
        scan.baseline_filter = bf
        return bool(overwritten)

    @staticmethod
    def _store_interp_payload(
        scan,
        *,
        interp_amp: np.ndarray,
        interp_phase: np.ndarray,
        smooth_amp: np.ndarray,
        smooth_phase: np.ndarray,
        attached_at: str,
        source: str,
    ) -> bool:
        bf = scan.baseline_filter if isinstance(scan.baseline_filter, dict) else {}
        overwritten = isinstance(bf.get("interp_smooth"), dict) and bool(bf.get("interp_smooth"))
        bf["interp_smooth"] = {
            "attached_at": attached_at,
            "attached_by": _current_user(),
            "source": source,
            "smoothing_width_ghz": 0.0,
            "smoothing_width_mhz": 0.0,
            "interp_amp": np.asarray(interp_amp, dtype=float),
            "interp_phase": np.asarray(interp_phase, dtype=float),
            "interp_data_polar": np.vstack((scan.freq, interp_amp, interp_phase)),
            "interp_data_polar_format": "(3, N) rows = [freq, amplitude, unwrapped_phase_deg]",
            "smooth_amp": np.asarray(smooth_amp, dtype=float),
            "smooth_phase": np.asarray(smooth_phase, dtype=float),
            "smooth_data_polar": np.vstack((scan.freq, smooth_amp, smooth_phase)),
            "smooth_data_polar_format": "(3, N) rows = [freq, amplitude, unwrapped_phase_deg]",
        }
        scan.baseline_filter = bf
        return bool(overwritten)

    def _borrowed_baseline_preview(self, source_scan, target_scan) -> Optional[dict[str, np.ndarray]]:
        source_interp = source_scan.baseline_filter.get("interp_smooth", {})
        interp_amp, interp_phase = _read_polar_series(
            source_interp,
            amplitude_key="interp_amp",
            phase_key="interp_phase",
        )
        if interp_amp.shape != source_scan.freq.shape or interp_phase.shape != source_scan.freq.shape:
            return None
        smooth_amp = np.asarray(source_interp.get("smooth_amp"), dtype=float)
        smooth_phase = np.asarray(source_interp.get("smooth_phase"), dtype=float)
        if smooth_amp.shape != source_scan.freq.shape:
            smooth_amp = np.asarray(interp_amp, dtype=float)
        if smooth_phase.shape != source_scan.freq.shape:
            smooth_phase = np.asarray(interp_phase, dtype=float)

        phase3 = target_scan.candidate_resonators.get("phase_correction_3", {})
        corrected_amp, corrected_phase = _read_polar_series(
            phase3,
            amplitude_key="corrected_amp",
            phase_key="corrected_phase_deg",
        )
        if corrected_amp.shape != target_scan.freq.shape or corrected_phase.shape != target_scan.freq.shape:
            return None

        source_freq = np.asarray(source_scan.freq, dtype=float)
        order = np.argsort(source_freq)
        source_freq = source_freq[order]
        interp_amp_sorted = np.asarray(interp_amp, dtype=float)[order]
        interp_phase_sorted = np.asarray(interp_phase, dtype=float)[order]
        smooth_amp_sorted = np.asarray(smooth_amp, dtype=float)[order]
        smooth_phase_sorted = np.asarray(smooth_phase, dtype=float)[order]
        target_freq = np.asarray(target_scan.freq, dtype=float)

        borrowed_interp_amp = np.interp(target_freq, source_freq, interp_amp_sorted)
        borrowed_interp_phase = np.interp(target_freq, source_freq, interp_phase_sorted)
        borrowed_smooth_amp = np.interp(target_freq, source_freq, smooth_amp_sorted)
        borrowed_smooth_phase = np.interp(target_freq, source_freq, smooth_phase_sorted)

        with np.errstate(divide="ignore", invalid="ignore"):
            norm_amp = np.divide(
                corrected_amp,
                borrowed_interp_amp,
                out=np.full(corrected_amp.shape, np.nan, dtype=float),
                where=np.abs(borrowed_interp_amp) > 0,
            )
        norm_phase = corrected_phase - borrowed_interp_phase
        return {
            "interp_amp": borrowed_interp_amp,
            "interp_phase": borrowed_interp_phase,
            "smooth_amp": borrowed_smooth_amp,
            "smooth_phase": borrowed_smooth_phase,
            "norm_amp": norm_amp,
            "norm_phase_deg_unwrapped": norm_phase,
        }

    def open_normalization_window(self) -> None:
        if not self._selected_scans_have_attached_interp_data():
            omitted = self._baseline_pipeline_omitted_selected_scans()
            omit_msg = ""
            if omitted:
                omit_msg = (
                    "\n\nSmall scans currently omitted from this step:\n"
                    + "\n".join(Path(scan.filename).name for scan in omitted[:10])
                )
            messagebox.showwarning(
                "Missing interp data",
                "Run pipeline in order:\n"
                "Phase Correction -> Baseline Filtering -> Interp+Smooth -> Normalize Baseline.\n\n"
                "All baseline-eligible selected scans must have attached interpolation data first."
                + omit_msg,
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

        omitted = self._baseline_pipeline_omitted_selected_scans()
        omitted_text = ""
        if omitted:
            omitted_text = " Omitted small scans: " + ", ".join(Path(scan.filename).name for scan in omitted[:4])
            if len(omitted) > 4:
                omitted_text += f", ... (+{len(omitted) - 4} more)"
        self.norm_status_var = tk.StringVar(
            value="Preview shows S21 / interpolated baseline. Click Attach to store." + omitted_text
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

        toolbar_frame, plot_parent = self._ensure_scrollable_plot_host("norm", self.norm_window)
        self.norm_figure = Figure(figsize=(12, 7))
        self.norm_canvas = FigureCanvasTkAgg(self.norm_figure, master=plot_parent)
        self.norm_toolbar = NavigationToolbar2Tk(self.norm_canvas, toolbar_frame)
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
        scans = self._baseline_pipeline_selected_scans()
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
        scans = self._baseline_pipeline_selected_scans()
        if not scans:
            self.norm_figure.clear()
            self.norm_canvas.draw_idle()
            return

        n = len(scans)
        self.norm_figure.clear()
        self._set_scrollable_figure_size(
            "norm",
            self.norm_figure,
            canvas_agg=self.norm_canvas,
            width_in=13.0,
            row_count=max(n, 1),
            row_height_in=2.6,
            min_height_in=7.0,
        )
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
        scans = self._baseline_pipeline_selected_scans()
        if not scans:
            return

        attached_at = datetime.now().isoformat(timespec="seconds")
        count = 0
        overwritten = 0
        for scan in scans:
            prev = self.norm_preview.get(self._scan_key(scan))
            if prev is None:
                continue
            overwritten += int(
                self._store_normalized_payload(
                    scan,
                    norm_amp=np.asarray(prev["norm_amp"], dtype=float),
                    norm_phase=np.asarray(prev["norm_phase_deg_unwrapped"], dtype=float),
                    attached_at=attached_at,
                    source="phase3_polar / interp_polar",
                )
            )
            scan.processing_history.append(
                _make_event(
                    "attach_normalized_baseline",
                    {"source": "phase3_polar / interp_polar", "points": int(scan.freq.size)},
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

    def apply_large_scan_baseline_to_selected(self) -> None:
        scans = self._selected_scans()
        if not scans:
            messagebox.showwarning(
                "No selection",
                "No scans selected for analysis.\nUse 'Select Scans for Analysis' first.",
            )
            return
        source_candidates = [scan for scan in scans if self._has_valid_interp_output(scan)]
        if not source_candidates:
            messagebox.showwarning(
                "Missing interp data",
                "At least one selected scan must already have attached interpolation data to act as the large-scan baseline source.",
            )
            return
        invalid_phase = [Path(scan.filename).name for scan in scans if not self._has_valid_phase3_output(scan)]
        if invalid_phase:
            messagebox.showwarning(
                "Missing Phase Correction 3 output",
                "All selected scans must have Phase Correction 3 attached before applying a large-scan baseline.\n\n"
                + "\n".join(invalid_phase[:10]),
            )
            return

        source_candidates = sorted(source_candidates, key=lambda scan: self._scan_span_mhz(scan), reverse=True)
        source_labels = [
            f"{Path(scan.filename).name} | span={self._scan_span_mhz(scan):.6f} MHz"
            for scan in source_candidates
        ]
        source_pick = self._select_setting_option(
            "Baseline Source Scan",
            "Choose the scan whose attached interpolated baseline should be applied to smaller scans.\n"
            "Default is the widest eligible selected scan.",
            source_labels,
            default_index=0,
        )
        if source_pick is None:
            return
        source_scan = source_candidates[source_pick]

        target_candidates = [scan for scan in scans if self._scan_key(scan) != self._scan_key(source_scan)]
        if not target_candidates:
            messagebox.showwarning(
                "No target scans",
                "There are no other selected scans to normalize with the chosen source baseline.",
            )
            return
        default_target_indices = [
            idx for idx, scan in enumerate(target_candidates) if self._scan_is_contained_in(scan, source_scan)
        ]
        target_labels = [
            f"{Path(scan.filename).name} | span={self._scan_span_mhz(scan):.6f} MHz"
            for scan in target_candidates
        ]
        target_picks = self._select_multiple_setting_options(
            "Target Scans",
            "Choose the scans that should borrow the selected large-scan baseline.\n"
            "Default selections are scans whose frequency range is fully contained within the source scan.",
            target_labels,
            default_indices=default_target_indices,
        )
        if not target_picks:
            return

        chosen_targets = [target_candidates[idx] for idx in target_picks]
        valid_targets = [scan for scan in chosen_targets if self._scan_is_contained_in(scan, source_scan)]
        invalid_targets = [scan for scan in chosen_targets if not self._scan_is_contained_in(scan, source_scan)]
        if invalid_targets:
            messagebox.showwarning(
                "Source range does not contain all targets",
                "Some selected target scans extend outside the source scan frequency range and will be skipped.\n\n"
                + "\n".join(Path(scan.filename).name for scan in invalid_targets[:10]),
            )
        if not valid_targets:
            messagebox.showwarning(
                "No compatible targets",
                "None of the chosen target scans fit within the selected source scan frequency range.",
            )
            return

        attached_at = datetime.now().isoformat(timespec="seconds")
        source_name = Path(source_scan.filename).name
        interp_overwritten = 0
        norm_overwritten = 0
        applied_count = 0
        failed_targets: list[str] = []
        for target_scan in valid_targets:
            preview = self._borrowed_baseline_preview(source_scan, target_scan)
            if preview is None:
                failed_targets.append(Path(target_scan.filename).name)
                continue
            interp_overwritten += int(
                self._store_interp_payload(
                    target_scan,
                    interp_amp=np.asarray(preview["interp_amp"], dtype=float),
                    interp_phase=np.asarray(preview["interp_phase"], dtype=float),
                    smooth_amp=np.asarray(preview["smooth_amp"], dtype=float),
                    smooth_phase=np.asarray(preview["smooth_phase"], dtype=float),
                    attached_at=attached_at,
                    source=f"borrowed interp baseline from {source_name}",
                )
            )
            norm_overwritten += int(
                self._store_normalized_payload(
                    target_scan,
                    norm_amp=np.asarray(preview["norm_amp"], dtype=float),
                    norm_phase=np.asarray(preview["norm_phase_deg_unwrapped"], dtype=float),
                    attached_at=attached_at,
                    source=f"phase3_polar / borrowed interp baseline from {source_name}",
                )
            )
            target_scan.processing_history.append(
                _make_event(
                    "apply_large_scan_baseline",
                    {
                        "source_scan": source_name,
                        "points": int(target_scan.freq.size),
                    },
                )
            )
            applied_count += 1

        self.dataset.processing_history.append(
            _make_event(
                "apply_large_scan_baseline_selected",
                {
                    "source_scan": source_name,
                    "selected_target_count": len(chosen_targets),
                    "applied_target_count": applied_count,
                    "skipped_out_of_range_count": len(invalid_targets),
                    "failed_count": len(failed_targets),
                },
            )
        )
        self._mark_dirty()
        self._refresh_status()
        self._update_norm_button_state()
        self._autosave_dataset()

        message = (
            f"Applied {source_name} baseline to {applied_count} scan(s). "
            f"Overwrote {interp_overwritten} interp attachment(s) and {norm_overwritten} normalized attachment(s)."
        )
        if failed_targets:
            message += "\n\nFailed targets:\n" + "\n".join(failed_targets[:10])
        messagebox.showinfo("Large-scan baseline applied", message)
        self._log(
            f"Applied borrowed baseline from {source_name} to {applied_count} selected scan(s); "
            f"skipped {len(invalid_targets)} out-of-range target(s), failed {len(failed_targets)}."
        )

    def _norm_close(self) -> None:
        if self.norm_window is not None and self.norm_window.winfo_exists():
            self.norm_window.destroy()
        self._destroy_scrollable_plot_host("norm")
        self.norm_window = None
        self.norm_canvas = None
        self.norm_toolbar = None
        self.norm_figure = None
        self.norm_status_var = None
        self.norm_attach_button = None
        self.norm_preview = {}
