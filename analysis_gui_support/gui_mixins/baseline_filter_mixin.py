from __future__ import annotations

import os
import queue
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter import messagebox, ttk

from ..analysis_filters import _compute_one_scan_filter, _estimate_frequency_resolution_mhz
from ..analysis_models import VNAScan, _current_user, _make_event, _read_polar_series


class BaselineFilterMixin:
    @staticmethod
    def _freq_span_mhz(freq: np.ndarray) -> float:
        freq_arr = np.asarray(freq, dtype=float)
        if freq_arr.size < 2:
            return 0.0
        finite = freq_arr[np.isfinite(freq_arr)]
        if finite.size < 2:
            return 0.0
        span = float(np.max(finite) - np.min(finite))
        median_abs = float(np.nanmedian(np.abs(finite)))
        if median_abs > 1e6:
            return span / 1.0e6
        if median_abs > 1e3:
            return span
        return span * 1.0e3

    def _baseline_partition_narrow_scans(
        self, scans: List[VNAScan]
    ) -> tuple[List[VNAScan], List[tuple[VNAScan, float]]]:
        eligible: List[VNAScan] = []
        narrow: List[tuple[VNAScan, float]] = []
        for scan in scans:
            span_mhz = self._freq_span_mhz(scan.freq)
            if span_mhz < 2.0:
                narrow.append((scan, span_mhz))
            else:
                eligible.append(scan)
        return eligible, narrow

    def _autoscale_y_for_visible_x(self, ax) -> None:
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

    def _set_attach_button_state(self, attached: bool) -> None:
        if self.baseline_attach_button is None:
            return
        if attached:
            self.baseline_attach_button.configure(
                bg="light green", activebackground="light green"
            )
        else:
            self.baseline_attach_button.configure(bg="pink", activebackground="pink")

    def open_baseline_filter_window(self) -> None:
        if not self.dataset.vna_scans:
            messagebox.showwarning("No data", "No VNA scans are loaded in this dataset.")
            return

        if not self._selected_scans():
            self.dataset.selected_scan_keys = [
                self._scan_key(scan) for scan in self.dataset.vna_scans
            ]
            self._refresh_status()
            self._log("No prior selection. Auto-selected all scans for baseline filtering.")
        scans = self._baseline_target_scans()
        if not scans:
            return
        missing_phase3 = []
        for scan in scans:
            phase3 = scan.candidate_resonators.get("phase_correction_3")
            if not isinstance(phase3, dict):
                missing_phase3.append(Path(scan.filename).name)
                continue
            amp3, phase3_deg = _read_polar_series(
                phase3,
                amplitude_key="corrected_amp",
                phase_key="corrected_phase_deg",
            )
            if amp3.shape != scan.freq.shape or phase3_deg.shape != scan.freq.shape:
                missing_phase3.append(Path(scan.filename).name)
        if missing_phase3:
            messagebox.showwarning(
                "Missing Phase Correction 3 output",
                "Run pipeline in order:\n"
                "Phase Correction 1 -> Phase Correction 2 -> Phase Correction 3 -> Baseline Filtering.\n\n"
                "Use 'Phase Correction 3' and click Attach for all selected scans before baseline filtering.",
            )
            return

        eligible_scans, narrow_scans = self._baseline_partition_narrow_scans(scans)
        if narrow_scans:
            names = "\n".join(
                f"{Path(scan.filename).name} ({span_mhz:.6f} MHz wide)"
                for scan, span_mhz in narrow_scans[:10]
            )
            ok = messagebox.askyesno(
                "Narrow scans detected",
                "Some selected scans are narrower than 2 MHz and may be too small to get a reliable baseline fit.\n\n"
                "Ignore those scans and continue baseline filtering for the wider scans?\n\n"
                + names,
            )
            if not ok:
                return
            if not eligible_scans:
                messagebox.showwarning(
                    "No eligible scans",
                    "All selected scans are narrower than 2 MHz, so baseline filtering was not started.",
                )
                return
            omitted_at = datetime.now().isoformat(timespec="seconds")
            eligible_keys = [self._scan_key(scan) for scan in eligible_scans]
            for scan in eligible_scans:
                bf = scan.baseline_filter if isinstance(scan.baseline_filter, dict) else {}
                bf.pop("omit_from_baseline_fit", None)
                bf.pop("omit_from_baseline_fit_reason", None)
                bf.pop("omit_from_baseline_fit_at", None)
                scan.baseline_filter = bf
            for scan, span_mhz in narrow_scans:
                bf = scan.baseline_filter if isinstance(scan.baseline_filter, dict) else {}
                bf["omit_from_baseline_fit"] = True
                bf["omit_from_baseline_fit_reason"] = (
                    f"Scan span {span_mhz:.6f} MHz is below the 2 MHz baseline-fit threshold."
                )
                bf["omit_from_baseline_fit_at"] = omitted_at
                scan.baseline_filter = bf
            self._baseline_target_scan_keys_override = set(eligible_keys)
            self._refresh_status()
            self._log(
                f"Ignoring {len(narrow_scans)} narrow scan(s) under 2 MHz for baseline filtering while leaving the active selection unchanged."
            )
            scans = eligible_scans
        else:
            for scan in scans:
                bf = scan.baseline_filter if isinstance(scan.baseline_filter, dict) else {}
                bf.pop("omit_from_baseline_fit", None)
                bf.pop("omit_from_baseline_fit_reason", None)
                bf.pop("omit_from_baseline_fit_at", None)
                scan.baseline_filter = bf
            self._baseline_target_scan_keys_override = None

        if self.baseline_window is not None and self.baseline_window.winfo_exists():
            self.baseline_window.lift()
            self._schedule_baseline_preview()
            return

        self.baseline_window = tk.Toplevel(self.root)
        self.baseline_window.title("Baseline Filtering")
        self.baseline_window.geometry("1200x800")
        self.baseline_window.protocol("WM_DELETE_WINDOW", self._close_baseline_window)

        controls = tk.Frame(self.baseline_window, padx=8, pady=8)
        controls.pack(side="top", fill="x")

        scans = self._baseline_target_scans()
        resolution_mhz = _estimate_frequency_resolution_mhz(scans)
        min_mhz = min(max(resolution_mhz, 1e-6), 10.0)
        max_mhz = 10.0
        default_width_mhz = min(max(1.0, min_mhz), max_mhz)
        default_step_mhz = min(max(0.5, min_mhz), max_mhz)
        default_low_slope = 10.0
        default_retain = 10.0
        default_center = 70.0

        saved_settings = []
        for scan in scans:
            bf = scan.baseline_filter
            if not isinstance(bf, dict):
                continue
            keys = (
                "window_width_ghz",
                "compute_step_ghz",
                "low_slope_percent",
                "retain_percent",
                "center_percent",
            )
            if not all(k in bf for k in keys):
                continue
            saved_settings.append(
                (
                    float(bf["window_width_ghz"]),
                    float(bf["compute_step_ghz"]),
                    float(bf["low_slope_percent"]),
                    float(bf["retain_percent"]),
                    float(bf["center_percent"]),
                )
            )

        if saved_settings:
            uniq = list(dict.fromkeys(saved_settings))
            chosen = None
            if len(uniq) > 1:
                labels = [
                    (
                        f"{idx+1}. width={s[0]*1000:.3f} MHz, step={s[1]*1000:.3f} MHz, "
                        f"low|dS21/df|={s[2]:.1f}%, retain={s[3]:.1f}%, center={s[4]:.1f}%"
                    )
                    for idx, s in enumerate(uniq)
                ]
                pick = self._select_setting_option(
                    "Baseline Setting",
                    "Selected scans have different saved baseline-filter settings. Choose defaults:",
                    labels,
                )
                if pick is not None:
                    chosen = uniq[pick]
                    self._log(f"Loaded chosen saved baseline setting #{pick + 1} into defaults.")
            else:
                chosen = uniq[0]
                self._log("Loaded saved baseline setting into defaults.")

            if chosen is not None:
                default_width_mhz = min(max(chosen[0] * 1000.0, min_mhz), max_mhz)
                default_step_mhz = min(max(chosen[1] * 1000.0, min_mhz), max_mhz)
                default_low_slope = chosen[2]
                default_retain = chosen[3]
                default_center = chosen[4]

        self.width_slider = tk.Scale(
            controls,
            from_=min_mhz,
            to=max_mhz,
            resolution=min_mhz,
            orient="horizontal",
            label="Window Width (MHz)",
            command=lambda _value: self._on_baseline_params_changed(),
            length=320,
        )
        self.width_slider.set(default_width_mhz)
        self.width_slider.pack(side="left", padx=(0, 12))
        self.width_slider.bind("<ButtonRelease-1>", self._on_baseline_slider_released)
        self.width_slider.bind("<KeyRelease>", self._on_baseline_slider_released)

        self.step_slider = tk.Scale(
            controls,
            from_=min_mhz,
            to=max_mhz,
            resolution=min_mhz,
            orient="horizontal",
            label="Compute Step (MHz)",
            command=lambda _value: self._on_baseline_params_changed(),
            length=250,
        )
        self.step_slider.set(default_step_mhz)
        self.step_slider.pack(side="left", padx=(0, 12))
        self.step_slider.bind("<ButtonRelease-1>", self._on_baseline_slider_released)
        self.step_slider.bind("<KeyRelease>", self._on_baseline_slider_released)

        self.low_slope_slider = tk.Scale(
            controls,
            from_=1,
            to=100,
            resolution=1,
            orient="horizontal",
            label="Lowest |dS21/df| (%)",
            command=lambda _value: self._on_baseline_params_changed(),
            length=220,
        )
        self.low_slope_slider.set(default_low_slope)
        self.low_slope_slider.pack(side="left", padx=(0, 12))
        self.low_slope_slider.bind("<ButtonRelease-1>", self._on_baseline_slider_released)
        self.low_slope_slider.bind("<KeyRelease>", self._on_baseline_slider_released)

        self.retain_slider = tk.Scale(
            controls,
            from_=1,
            to=100,
            resolution=1,
            orient="horizontal",
            label="Retain (%)",
            command=lambda _value: self._on_baseline_params_changed(),
            length=220,
        )
        self.retain_slider.set(default_retain)
        self.retain_slider.pack(side="left", padx=(0, 12))
        self.retain_slider.bind("<ButtonRelease-1>", self._on_baseline_slider_released)
        self.retain_slider.bind("<KeyRelease>", self._on_baseline_slider_released)

        self.center_slider = tk.Scale(
            controls,
            from_=0,
            to=100,
            resolution=1,
            orient="horizontal",
            label="Center (%)",
            command=lambda _value: self._on_baseline_params_changed(),
            length=220,
        )
        self.center_slider.set(default_center)
        self.center_slider.pack(side="left", padx=(0, 12))
        self.center_slider.bind("<ButtonRelease-1>", self._on_baseline_slider_released)
        self.center_slider.bind("<KeyRelease>", self._on_baseline_slider_released)

        status_frame = tk.Frame(self.baseline_window, padx=8, pady=6)
        status_frame.pack(side="top", fill="x")
        self.baseline_status_var = tk.StringVar(value="Ready. Auto-computing on parameter changes.")
        tk.Label(status_frame, textvariable=self.baseline_status_var, anchor="w").pack(
            side="left", fill="x", expand=True
        )
        self.baseline_progress = ttk.Progressbar(
            status_frame, orient="horizontal", mode="determinate", length=280
        )
        self.baseline_progress.pack(side="right")
        self.baseline_progress["value"] = 0

        action_frame = tk.Frame(self.baseline_window, padx=8, pady=6)
        action_frame.pack(side="top", fill="x")
        tk.Button(action_frame, text="Cancel", width=12, command=self._close_baseline_window).pack(
            side="right"
        )
        tk.Button(
            action_frame,
            text="Reset View",
            width=12,
            command=self._baseline_reset_view,
        ).pack(side="right", padx=(8, 0))
        self.baseline_attach_button = tk.Button(
            action_frame,
            text="Attach, Save, and Close",
            width=24,
            command=self._attach_save_and_close_baseline,
        )
        self.baseline_attach_button.pack(side="right", padx=(8, 0))
        self._set_attach_button_state(attached=False)

        toolbar_frame, plot_parent = self._ensure_scrollable_plot_host("baseline", self.baseline_window)
        self.baseline_figure = Figure(figsize=(12, 7))
        self.baseline_canvas = FigureCanvasTkAgg(self.baseline_figure, master=plot_parent)
        self.baseline_toolbar = NavigationToolbar2Tk(self.baseline_canvas, toolbar_frame)
        self.baseline_toolbar.update()
        def _home_baseline(*_args) -> None:
            if self.baseline_figure is None or self.baseline_canvas is None:
                return
            for ax in self.baseline_figure.axes:
                ax.relim()
                ax.autoscale(enable=True, axis="both", tight=False)
            self.baseline_canvas.draw_idle()
        self.baseline_toolbar.home = _home_baseline
        self.baseline_toolbar.pack(side="top", fill="x")
        self.baseline_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.baseline_canvas.mpl_connect(
            "button_release_event", lambda _evt: self._baseline_autoscale_all_y()
        )
        if self.baseline_status_var is not None:
            self.baseline_status_var.set(
                f"Ready. Resolution ~{resolution_mhz:.6f} MHz. Slider range: {min_mhz:.6f} to {max_mhz:.3f} MHz."
            )

        self._baseline_preview_results = {}
        self._schedule_baseline_preview()
        self._request_baseline_recompute()
        self._log(
            f"Opened baseline filtering window. Resolution ~{resolution_mhz:.6f} MHz, slider range {min_mhz:.6f}-{max_mhz:.3f} MHz."
        )

    def _close_baseline_window(self) -> None:
        if self.baseline_window is not None and self.baseline_window.winfo_exists():
            self.baseline_window.destroy()
        self._baseline_compute_running = False
        self._destroy_scrollable_plot_host("baseline")
        self.baseline_window = None
        self.baseline_canvas = None
        self.baseline_toolbar = None
        self.baseline_figure = None
        self.width_slider = None
        self.step_slider = None
        self.low_slope_slider = None
        self.retain_slider = None
        self.center_slider = None
        self.baseline_attach_button = None
        self._baseline_after_id = None
        self._baseline_preview_results = {}
        self.baseline_status_var = None
        self.baseline_progress = None
        self._baseline_recompute_pending = False

    def _schedule_baseline_preview(self) -> None:
        if self.baseline_window is None or not self.baseline_window.winfo_exists():
            return
        if self._baseline_after_id:
            self.baseline_window.after_cancel(self._baseline_after_id)
        self._baseline_after_id = self.baseline_window.after(120, self._render_baseline_preview)

    def _baseline_autoscale_all_y(self) -> None:
        if self.baseline_figure is None or self.baseline_canvas is None:
            return
        for ax in self.baseline_figure.axes:
            self._autoscale_y_for_visible_x(ax)
        self.baseline_canvas.draw_idle()

    def _baseline_reset_view(self) -> None:
        if self.baseline_figure is None or self.baseline_canvas is None:
            return
        for ax in self.baseline_figure.axes:
            ax.relim()
            ax.autoscale(enable=True, axis="both", tight=False)
        self.baseline_canvas.draw_idle()

    def _on_baseline_params_changed(self) -> None:
        if self._baseline_compute_running:
            return
        if self.baseline_status_var is not None:
            self.baseline_status_var.set("Adjusting parameters...")
        if self.baseline_progress is not None:
            self.baseline_progress["value"] = 0
        self._set_attach_button_state(attached=False)

    def _on_baseline_slider_released(self, _event: tk.Event) -> None:
        if self._baseline_compute_running:
            self._baseline_recompute_pending = True
            if self.baseline_status_var is not None:
                self.baseline_status_var.set("Parameters changed. Recompute queued...")
            return
        self._request_baseline_recompute()
        self._set_attach_button_state(attached=False)

    def _set_baseline_controls_state(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        if self.width_slider is not None:
            self.width_slider.configure(state=state)
        if self.step_slider is not None:
            self.step_slider.configure(state=state)
        if self.low_slope_slider is not None:
            self.low_slope_slider.configure(state=state)
        if self.retain_slider is not None:
            self.retain_slider.configure(state=state)
        if self.center_slider is not None:
            self.center_slider.configure(state=state)
        if self.baseline_attach_button is not None:
            self.baseline_attach_button.configure(state=state)

    def _current_baseline_params(
        self,
    ) -> Optional[tuple[List[VNAScan], float, float, float, float, float]]:
        if (
            self.width_slider is None
            or self.step_slider is None
            or self.low_slope_slider is None
            or self.retain_slider is None
            or self.center_slider is None
        ):
            return None
        scans = self._baseline_target_scans()
        if not scans:
            return None
        width_ghz = float(self.width_slider.get()) / 1000.0
        step_ghz = float(self.step_slider.get()) / 1000.0
        low_slope_pct = float(self.low_slope_slider.get())
        retain_pct = float(self.retain_slider.get())
        center_pct = float(self.center_slider.get())
        return scans, width_ghz, step_ghz, low_slope_pct, retain_pct, center_pct

    def _request_baseline_recompute(self) -> None:
        params = self._current_baseline_params()
        if params is None:
            return
        scans, width_ghz, step_ghz, low_slope_pct, retain_pct, center_pct = params
        if self._baseline_compute_running:
            self._baseline_recompute_pending = True
            if self.baseline_status_var is not None:
                self.baseline_status_var.set("Parameters changed. Recompute queued...")
            return
        self._start_baseline_worker(
            scans, width_ghz, step_ghz, low_slope_pct, retain_pct, center_pct
        )

    def _start_baseline_worker(
        self,
        scans: List[VNAScan],
        width_ghz: float,
        step_ghz: float,
        low_slope_pct: float,
        retain_pct: float,
        center_pct: float,
    ) -> None:
        self._baseline_compute_running = True
        self._baseline_recompute_pending = False
        self._set_baseline_controls_state(enabled=False)
        self._log(
            f"Baseline compute started: scans={len(scans)}, width={width_ghz:.6f} GHz, step={step_ghz:.6f} GHz, low_slope={low_slope_pct:.1f}%, retain={retain_pct:.1f}%, center={center_pct:.1f}%"
        )
        self._baseline_compute_context = {
            "selected_count": len(scans),
            "window_width_ghz": width_ghz,
            "compute_step_ghz": step_ghz,
            "low_slope_percent": low_slope_pct,
            "retain_percent": retain_pct,
            "center_percent": center_pct,
        }
        self._baseline_preview_results = {}
        if self.baseline_status_var is not None:
            self.baseline_status_var.set("Starting compute...")
        if self.baseline_progress is not None:
            self.baseline_progress["value"] = 0
            self.baseline_progress["maximum"] = max(1, len(scans))
        while True:
            try:
                self._baseline_worker_queue.get_nowait()
            except queue.Empty:
                break

        self._baseline_worker_thread = threading.Thread(
            target=self._baseline_worker_main,
            args=(scans, width_ghz, step_ghz, low_slope_pct, retain_pct, center_pct),
            daemon=True,
        )
        self._baseline_worker_thread.start()
        if self.baseline_window is not None and self.baseline_window.winfo_exists():
            self.baseline_window.after(50, self._poll_baseline_worker_queue)

    def _baseline_worker_main(
        self,
        scans: List[VNAScan],
        width_ghz: float,
        step_ghz: float,
        low_slope_pct: float,
        retain_pct: float,
        center_pct: float,
    ) -> None:
        try:
            results: Dict[str, Dict[str, np.ndarray]] = {}
            total = len(scans)
            if total > 1:
                max_workers = min(total, max(1, os.cpu_count() or 1))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_scan = {
                        executor.submit(
                            _compute_one_scan_filter,
                            scan,
                            width_ghz,
                            step_ghz,
                            retain_pct,
                            center_pct,
                            low_slope_pct,
                        ): scan
                        for scan in scans
                    }
                    done = 0
                    for fut in as_completed(future_to_scan):
                        scan = future_to_scan[fut]
                        result = fut.result()
                        results[self._scan_key(scan)] = result
                        done += 1
                        self._baseline_worker_queue.put(
                            {
                                "type": "progress",
                                "done": done,
                                "total": total,
                                "name": Path(scan.filename).name,
                            }
                        )
            else:
                scan = scans[0]

                def _progress(done: int, total_pts: int) -> None:
                    self._baseline_worker_queue.put(
                        {
                            "type": "progress_points",
                            "done_pts": done,
                            "total_pts": max(total_pts, 1),
                            "name": Path(scan.filename).name,
                        }
                    )

                result = _compute_one_scan_filter(
                    scan,
                    width_ghz=width_ghz,
                    step_ghz=step_ghz,
                    retain_pct=retain_pct,
                    center_pct=center_pct,
                    low_slope_pct=low_slope_pct,
                    progress_cb=_progress,
                )
                results[self._scan_key(scan)] = result
                self._baseline_worker_queue.put(
                    {"type": "progress", "done": 1, "total": 1, "name": Path(scan.filename).name}
                )

            self._baseline_worker_queue.put({"type": "done", "results": results})
        except Exception:
            self._baseline_worker_queue.put(
                {"type": "error", "message": traceback.format_exc(limit=5)}
            )

    def _poll_baseline_worker_queue(self) -> None:
        had_event = False
        while True:
            try:
                event = self._baseline_worker_queue.get_nowait()
            except queue.Empty:
                break
            had_event = True
            etype = event.get("type")
            if etype == "progress":
                done = int(event.get("done", 0))
                total = int(event.get("total", 1))
                name = str(event.get("name", "scan"))
                if self.baseline_progress is not None:
                    self.baseline_progress["maximum"] = max(1, total)
                    self.baseline_progress["value"] = min(done, total)
                if self.baseline_status_var is not None:
                    self.baseline_status_var.set(f"Computing {done}/{total}: {name}")
            elif etype == "progress_points":
                done_pts = int(event.get("done_pts", 0))
                total_pts = int(event.get("total_pts", 1))
                name = str(event.get("name", "scan"))
                if self.baseline_progress is not None:
                    self.baseline_progress["maximum"] = max(1, total_pts)
                    self.baseline_progress["value"] = min(done_pts, total_pts)
                if self.baseline_status_var is not None:
                    pct = 100.0 * done_pts / max(total_pts, 1)
                    self.baseline_status_var.set(f"Computing {name} ({pct:.1f}%)")
            elif etype == "done":
                self._baseline_compute_running = False
                self._set_baseline_controls_state(enabled=True)
                self._baseline_preview_results = event["results"]
                if self.baseline_status_var is not None:
                    self.baseline_status_var.set(
                        f"Done. Computed {len(self._baseline_preview_results)} scan(s)."
                    )
                if self.baseline_progress is not None:
                    self.baseline_progress["value"] = self.baseline_progress["maximum"]
                self._log(
                    f"Computed baseline filter for {len(self._baseline_preview_results)} selected scan(s)."
                )
                self.dataset.processing_history.append(
                    _make_event("compute_baseline_filter", dict(self._baseline_compute_context))
                )
                self._schedule_baseline_preview()
                self._set_attach_button_state(attached=False)
                if self._baseline_recompute_pending:
                    self._request_baseline_recompute()
                return
            elif etype == "error":
                self._baseline_compute_running = False
                self._set_baseline_controls_state(enabled=True)
                if self.baseline_status_var is not None:
                    self.baseline_status_var.set("Compute failed.")
                self._log("Baseline compute failed.")
                messagebox.showerror("Baseline compute failed", str(event.get("message", "")))
                if self._baseline_recompute_pending:
                    self._request_baseline_recompute()
                return

        if self._baseline_compute_running and self.baseline_window is not None and self.baseline_window.winfo_exists():
            self.baseline_window.after(50, self._poll_baseline_worker_queue)
        elif had_event and self.baseline_window is not None and self.baseline_window.winfo_exists():
            self.baseline_window.update_idletasks()

    def _render_baseline_preview(self) -> None:
        if (
            self.baseline_window is None
            or self.baseline_figure is None
            or self.baseline_canvas is None
            or self.width_slider is None
            or self.step_slider is None
            or self.low_slope_slider is None
            or self.retain_slider is None
            or self.center_slider is None
        ):
            return
        self._baseline_after_id = None

        try:
            saved_x_limits = []
            for ax_old in self.baseline_figure.axes:
                saved_x_limits.append(ax_old.get_xlim())
            scans = self._baseline_target_scans()
            if not scans:
                self.baseline_figure.clear()
                ax = self.baseline_figure.add_subplot(111)
                ax.text(0.5, 0.5, "No selected scans", ha="center", va="center")
                ax.axis("off")
                self.baseline_canvas.draw_idle()
                return

            width_ghz = float(self.width_slider.get()) / 1000.0
            step_ghz = float(self.step_slider.get()) / 1000.0
            low_slope_pct = float(self.low_slope_slider.get())
            retain_pct = float(self.retain_slider.get())
            center_pct = float(self.center_slider.get())

            n = len(scans)
            self.baseline_figure.clear()
            self._set_scrollable_figure_size(
                "baseline",
                self.baseline_figure,
                canvas_agg=self.baseline_canvas,
                width_in=13.0,
                row_count=max(n, 1),
                row_height_in=2.8,
                min_height_in=7.0,
            )
            axes = self.baseline_figure.subplots(n, 2, sharex=False)
            axes_arr = np.atleast_2d(axes)
            axes_list = list(axes_arr.ravel())

            for i, scan in enumerate(scans):
                freq_ghz = np.asarray(scan.freq, dtype=float) / 1.0e9
                phase3 = scan.candidate_resonators.get("phase_correction_3", {})
                amp, ph = _read_polar_series(
                    phase3,
                    amplitude_key="corrected_amp",
                    phase_key="corrected_phase_deg",
                )
                ax_a = axes_arr[i, 0]
                ax_p = axes_arr[i, 1]
                ax_a.plot(freq_ghz, amp, color="0.6", linewidth=0.8, label="Amplitude input")
                ax_p.plot(freq_ghz, ph, color="0.6", linewidth=0.8, label="Phase input")

                key = self._scan_key(scan)
                result = self._baseline_preview_results.get(key)
                if result is not None:
                    baseline = result["baseline_amplitude"]
                    keep = result["retained_mask"].astype(bool)
                    ax_a.plot(freq_ghz, baseline, color="tab:blue", linewidth=0.8, label="Median")
                    ax_a.plot(
                        freq_ghz[keep],
                        amp[keep],
                        linestyle="none",
                        marker=".",
                        markersize=2.0,
                        color="tab:orange",
                        label="Retained",
                    )
                    ax_p.plot(
                        freq_ghz[keep],
                        ph[keep],
                        linestyle="none",
                        marker=".",
                        markersize=2.0,
                        color="tab:orange",
                        label="Retained-associated phase",
                    )
                ax_a.set_ylabel("|S21|")
                ax_p.set_ylabel("Phase (deg)")
                ax_a.grid(True, alpha=0.3)
                ax_p.grid(True, alpha=0.3)
                ax_a.set_title(Path(scan.filename).name + " | Amplitude", fontsize=9)
                ax_p.set_title(Path(scan.filename).name + " | Phase", fontsize=9)
                if i == 0:
                    ax_a.legend(loc="upper right", fontsize=8)
                    ax_p.legend(loc="upper right", fontsize=8)

            axes_arr[-1, 0].set_xlabel("Frequency (GHz)")
            axes_arr[-1, 1].set_xlabel("Frequency (GHz)")
            computed_count = len(self._baseline_preview_results)
            title_suffix = (
                f"Computed overlays: {computed_count}/{len(scans)}"
                if computed_count
                else "Unfiltered view"
            )
            self.baseline_figure.suptitle(
                f"Baseline Filtering | width={width_ghz*1000.0:.2f} MHz, step={step_ghz*1000.0:.2f} MHz, low |dS21/df|={low_slope_pct:.0f}%, retain={retain_pct:.0f}%, center={center_pct:.0f}% | {title_suffix}",
                fontsize=10,
            )
            if len(saved_x_limits) == len(axes_list):
                for ax_new, xlim in zip(axes_list, saved_x_limits):
                    ax_new.set_xlim(xlim)
                    self._autoscale_y_for_visible_x(ax_new)
            for ax_new in axes_list:
                ax_new.callbacks.connect(
                    "xlim_changed",
                    lambda changed_ax: self._autoscale_y_for_visible_x(changed_ax),
                )
            self.baseline_figure.tight_layout()
            self.baseline_canvas.draw_idle()
        except Exception as exc:
            self._log(f"Baseline preview failed: {exc}")
            messagebox.showerror("Baseline preview failed", str(exc))

    def attach_baseline_filter(self) -> None:
        if (
            self.width_slider is None
            or self.step_slider is None
            or self.low_slope_slider is None
            or self.retain_slider is None
            or self.center_slider is None
        ):
            return
        if self._baseline_compute_running:
            messagebox.showwarning("Compute running", "Wait for compute to finish before Attach.")
            return

        scans = self._baseline_target_scans()
        if not scans:
            messagebox.showwarning("No selection", "No scans selected for analysis.")
            return
        if not self._baseline_preview_results:
            messagebox.showwarning("Not computed", "No computed filter is available yet.")
            return

        width_ghz = float(self.width_slider.get()) / 1000.0
        step_ghz = float(self.step_slider.get()) / 1000.0
        low_slope_pct = float(self.low_slope_slider.get())
        retain_pct = float(self.retain_slider.get())
        center_pct = float(self.center_slider.get())
        attached_at = datetime.now().isoformat(timespec="seconds")

        for scan in scans:
            result = self._baseline_preview_results.get(self._scan_key(scan))
            scan.baseline_filter = {}
            if result is None:
                continue
            keep = result["retained_mask"].astype(bool)
            baseline = result["baseline_amplitude"]
            filtered_freq = scan.freq[keep]
            phase3 = scan.candidate_resonators["phase_correction_3"]
            corrected_amp = np.asarray(phase3["corrected_amp"], dtype=float)
            corrected_phase = np.asarray(phase3["corrected_phase_deg"], dtype=float)
            filtered_amp = corrected_amp[keep]
            filtered_phase = corrected_phase[keep]
            scan.baseline_filter = {
                "attached_at": attached_at,
                "attached_by": _current_user(),
                "window_width_ghz": width_ghz,
                "compute_step_ghz": step_ghz,
                "low_slope_percent": low_slope_pct,
                "retain_percent": retain_pct,
                "center_percent": center_pct,
                "slope_survivor_mask": result.get("slope_survivor_mask"),
                "retained_mask": keep,
                "baseline_amplitude": baseline,
                "filtered_amp": filtered_amp,
                "filtered_phase_deg": filtered_phase,
                "filtered_data_polar": np.vstack((filtered_freq, filtered_amp, filtered_phase)),
                "filtered_data_polar_format": "(3, N_kept) rows = [freq, amplitude, unwrapped_phase_deg]",
            }
            scan.processing_history.append(
                _make_event(
                    "attach_baseline_filter",
                    {
                        "window_width_ghz": width_ghz,
                        "compute_step_ghz": step_ghz,
                        "low_slope_percent": low_slope_pct,
                        "retain_percent": retain_pct,
                        "center_percent": center_pct,
                        "kept_points": int(np.count_nonzero(keep)),
                        "total_points": int(keep.size),
                    },
                )
            )

        self.dataset.processing_history.append(
            _make_event(
                "attach_baseline_filter_to_selected",
                {
                    "selected_count": len(scans),
                    "window_width_ghz": width_ghz,
                    "compute_step_ghz": step_ghz,
                    "low_slope_percent": low_slope_pct,
                    "retain_percent": retain_pct,
                    "center_percent": center_pct,
                },
            )
        )
        self._mark_dirty()

        self._log(f"Attached baseline filter to {len(scans)} selected scan(s).")
        self._autosave_dataset()
        if self.baseline_status_var is not None:
            self.baseline_status_var.set(f"Attached to {len(scans)} scan(s).")
        self._set_attach_button_state(attached=True)
