from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import numpy as np
import tkinter as tk
from matplotlib import colormaps
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy import optimize
from tkinter import messagebox, ttk

from ..analysis_models import _make_event
from .resonance_selection_mixin import _HZ_PER_GHZ, fit_nonlinear_iq, guess_p0_nonlinear_iq, nonlinear_iq


class BatchResonanceFitMixin:
    def open_accepted_fit_parameter_date_window(self) -> None:
        scans = self._selected_scans()
        if not scans:
            messagebox.showwarning(
                "No selection",
                "No scans selected for analysis.\nUse 'Select Scans for Analysis' first.",
            )
            return
        points, stats = self._accepted_fit_parameter_date_collect_points(scans)
        if not points:
            messagebox.showwarning(
                "No accepted fits",
                "No accepted marker fit results with valid VNA dates were found on the selected scans.",
            )
            return

        if (
            getattr(self, "accepted_fit_parameter_date_window", None) is not None
            and self.accepted_fit_parameter_date_window.winfo_exists()
        ):
            self.accepted_fit_parameter_date_points = points
            self.accepted_fit_parameter_date_stats = stats
            self.accepted_fit_parameter_date_window.lift()
            self._accepted_fit_parameter_date_render()
            return

        self.accepted_fit_parameter_date_points = points
        self.accepted_fit_parameter_date_stats = stats
        self.accepted_fit_parameter_date_window = tk.Toplevel(self.root)
        self.accepted_fit_parameter_date_window.title("Accepted Fit Parameters vs Date")
        self.accepted_fit_parameter_date_window.geometry("1160x800")
        self.accepted_fit_parameter_date_window.protocol(
            "WM_DELETE_WINDOW", self._accepted_fit_parameter_date_close
        )

        controls = tk.Frame(self.accepted_fit_parameter_date_window, padx=8, pady=6)
        controls.pack(side="top", fill="x")
        self.accepted_fit_parameter_date_status_var = tk.StringVar()
        tk.Label(controls, textvariable=self.accepted_fit_parameter_date_status_var, anchor="w").pack(
            side="top", fill="x"
        )
        option_row = tk.Frame(controls)
        option_row.pack(side="top", fill="x", pady=(6, 0))
        self.accepted_fit_parameter_date_y_var = tk.StringVar(value="a_nl")
        tk.Label(option_row, text="Y axis").pack(side="left", padx=(0, 6))
        for label, value in (
            ("nonlinear parameter a", "a_nl"),
            ("(fr0 - initial fr0) / initial fr0", "fr0_rel_change"),
            ("(marker frequency - fr0) / fr0", "marker_minus_fr0_rel"),
        ):
            tk.Radiobutton(
                option_row,
                text=label,
                value=value,
                variable=self.accepted_fit_parameter_date_y_var,
                command=self._accepted_fit_parameter_date_render,
            ).pack(side="left", padx=(0, 10))

        self.accepted_fit_parameter_date_figure = Figure(figsize=(11, 6.8))
        self.accepted_fit_parameter_date_canvas = FigureCanvasTkAgg(
            self.accepted_fit_parameter_date_figure,
            master=self.accepted_fit_parameter_date_window,
        )
        self.accepted_fit_parameter_date_toolbar = NavigationToolbar2Tk(
            self.accepted_fit_parameter_date_canvas,
            self.accepted_fit_parameter_date_window,
        )
        self.accepted_fit_parameter_date_toolbar.update()
        self.accepted_fit_parameter_date_toolbar.pack(side="top", fill="x")
        self.accepted_fit_parameter_date_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._accepted_fit_parameter_date_render()

    def _accepted_fit_parameter_date_close(self) -> None:
        if (
            getattr(self, "accepted_fit_parameter_date_window", None) is not None
            and self.accepted_fit_parameter_date_window.winfo_exists()
        ):
            self.accepted_fit_parameter_date_window.destroy()
        self.accepted_fit_parameter_date_window = None
        self.accepted_fit_parameter_date_canvas = None
        self.accepted_fit_parameter_date_toolbar = None
        self.accepted_fit_parameter_date_figure = None
        self.accepted_fit_parameter_date_status_var = None
        self.accepted_fit_parameter_date_y_var = None
        self.accepted_fit_parameter_date_points = []
        self.accepted_fit_parameter_date_stats = {}

    @staticmethod
    def _accepted_fit_parameter_date_parse(text: object) -> Optional[datetime]:
        stamp = str(text or "").strip()
        if not stamp:
            return None
        try:
            return datetime.fromisoformat(stamp)
        except Exception:
            pass
        for fmt in ("%Y-%m-%d", "%Y%m%d_%H%M%S", "%Y%m%d"):
            try:
                return datetime.strptime(stamp, fmt)
            except Exception:
                continue
        return None

    def _accepted_fit_parameter_date_collect_points(self, scans: list[object]) -> tuple[list[dict], dict]:
        points = []
        skipped_no_date = 0
        skipped_no_payload = 0
        skipped_unaccepted = 0
        skipped_invalid = 0
        for scan in scans:
            scan_date = self._accepted_fit_parameter_date_parse(getattr(scan, "file_timestamp", ""))
            if scan_date is None:
                scan_date = self._accepted_fit_parameter_date_parse(getattr(scan, "loaded_at", ""))
            if scan_date is None:
                skipped_no_date += 1
                continue
            fit_payload = scan.candidate_resonators.get("logan_nonlinear_iq_marker_fits")
            assignments = fit_payload.get("assignments") if isinstance(fit_payload, dict) else {}
            if not isinstance(assignments, dict):
                skipped_no_payload += 1
                continue
            for resonator_number, payload in assignments.items():
                if not isinstance(payload, dict) or not payload.get("success"):
                    skipped_invalid += 1
                    continue
                if not bool(payload.get("accepted", False)):
                    skipped_unaccepted += 1
                    continue
                try:
                    fr0_hz = float(payload.get("fr0_hz", np.nan))
                    a_nl = float(payload.get("a_nl", np.nan))
                    marker_hz = float(payload.get("marker_frequency_hz", np.nan))
                except Exception:
                    skipped_invalid += 1
                    continue
                if not np.isfinite(marker_hz):
                    marker_record = payload.get("source_marker_record", {})
                    try:
                        marker_hz = float(marker_record.get("frequency_hz", np.nan)) if isinstance(marker_record, dict) else np.nan
                    except Exception:
                        marker_hz = np.nan
                discrepancy = (marker_hz - fr0_hz) / fr0_hz if np.isfinite(marker_hz) and fr0_hz != 0.0 else np.nan
                if not np.isfinite(fr0_hz) or not np.isfinite(a_nl):
                    skipped_invalid += 1
                    continue
                points.append(
                    {
                        "scan": scan,
                        "scan_key": self._scan_key(scan),
                        "scan_date": scan_date,
                        "date_num": float(mdates.date2num(scan_date)),
                        "scan_label": Path(getattr(scan, "filename", "")).name,
                        "resonator_number": str(resonator_number),
                        "fr0_hz": fr0_hz,
                        "a_nl": a_nl,
                        "marker_hz": marker_hz,
                        "marker_minus_fr0_rel": float(discrepancy),
                    }
                )

        freq_by_res: dict[str, list[float]] = {}
        for point in points:
            freq_by_res.setdefault(str(point["resonator_number"]), []).append(float(point["fr0_hz"]))
        color_freq_by_res = {
            resonator: float(np.mean(np.asarray(values, dtype=float)))
            for resonator, values in freq_by_res.items()
            if len(values) > 0
        }
        for point in points:
            point["color_freq_hz"] = color_freq_by_res.get(str(point["resonator_number"]), float(point["fr0_hz"]))
        for resonator, values in color_freq_by_res.items():
            resonator_points = [point for point in points if str(point["resonator_number"]) == resonator]
            resonator_points.sort(key=lambda point: float(point["date_num"]))
            initial_fr0_hz = float(resonator_points[0]["fr0_hz"]) if resonator_points else np.nan
            for point in resonator_points:
                point["initial_fr0_hz"] = initial_fr0_hz
                point["fr0_rel_change"] = (
                    (float(point["fr0_hz"]) - initial_fr0_hz) / initial_fr0_hz
                    if np.isfinite(initial_fr0_hz) and initial_fr0_hz != 0.0
                    else np.nan
                )

        return points, {
            "selected_scan_count": int(len(scans)),
            "point_count": int(len(points)),
            "resonator_count": int(len(color_freq_by_res)),
            "skipped_no_date": int(skipped_no_date),
            "skipped_no_payload": int(skipped_no_payload),
            "skipped_unaccepted": int(skipped_unaccepted),
            "skipped_invalid": int(skipped_invalid),
        }

    def _accepted_fit_parameter_date_render(self) -> None:
        if (
            getattr(self, "accepted_fit_parameter_date_figure", None) is None
            or self.accepted_fit_parameter_date_canvas is None
        ):
            return
        points = list(getattr(self, "accepted_fit_parameter_date_points", []))
        stats = dict(getattr(self, "accepted_fit_parameter_date_stats", {}))
        self.accepted_fit_parameter_date_figure.clear()
        ax = self.accepted_fit_parameter_date_figure.add_subplot(1, 1, 1)
        if not points:
            ax.text(0.5, 0.5, "No accepted fit points.", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            self.accepted_fit_parameter_date_canvas.draw_idle()
            return

        y_mode = (
            str(self.accepted_fit_parameter_date_y_var.get())
            if getattr(self, "accepted_fit_parameter_date_y_var", None) is not None
            else "a_nl"
        )
        if y_mode == "fr0_rel_change":
            y = np.asarray([float(point.get("fr0_rel_change", np.nan)) for point in points], dtype=float)
            y_label = "(fr0 - initial fr0) / initial fr0"
        elif y_mode == "marker_minus_fr0_rel":
            y = np.asarray([float(point.get("marker_minus_fr0_rel", np.nan)) for point in points], dtype=float)
            y_label = "(marker frequency - fr0) / fr0"
        else:
            y_mode = "a_nl"
            y = np.asarray([float(point["a_nl"]) for point in points], dtype=float)
            y_label = "Nonlinear parameter a"

        x = np.asarray([float(point["date_num"]) for point in points], dtype=float)
        color_freq = np.asarray([float(point["color_freq_hz"]) / _HZ_PER_GHZ for point in points], dtype=float)
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(color_freq)

        if not np.any(finite):
            ax.text(0.5, 0.5, "No finite points for this Y axis.", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            self.accepted_fit_parameter_date_canvas.draw_idle()
            return

        grouped: dict[str, list[int]] = {}
        for index, point in enumerate(points):
            if finite[index]:
                grouped.setdefault(str(point["resonator_number"]), []).append(index)
        for indices in grouped.values():
            if len(indices) < 2:
                continue
            order = sorted(indices, key=lambda index: x[index])
            ax.plot(x[order], y[order], color="0.65", linewidth=0.9, alpha=0.7, zorder=1)

        scatter = ax.scatter(
            x[finite],
            y[finite],
            c=color_freq[finite],
            cmap="rainbow_r",
            s=34,
            edgecolors="0.2",
            zorder=2,
        )
        cbar = self.accepted_fit_parameter_date_figure.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label("Resonator mean fr0 (GHz)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.set_xlabel("VNA date")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        ax.set_title("Accepted marker fits vs VNA date", fontsize=10)
        if y_mode == "marker_minus_fr0_rel":
            y_finite = y[finite]
            if y_finite.size >= 2:
                std_text = f"STD = {float(np.std(y_finite, ddof=1)):.3e}"
            elif y_finite.size == 1:
                std_text = "STD = 0"
            else:
                std_text = ""
            if std_text:
                ax.text(
                    0.02,
                    0.98,
                    std_text,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                    bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.85},
                )
        self.accepted_fit_parameter_date_figure.autofmt_xdate()
        self.accepted_fit_parameter_date_figure.tight_layout()
        self.accepted_fit_parameter_date_canvas.draw_idle()
        if getattr(self, "accepted_fit_parameter_date_status_var", None) is not None:
            self.accepted_fit_parameter_date_status_var.set(
                f"Showing {int(np.count_nonzero(finite))}/{len(points)} accepted fit point(s) "
                f"from {stats.get('resonator_count', 0)} resonator(s). "
                f"Skipped {stats.get('skipped_no_date', 0)} scan(s) with no date, "
                f"{stats.get('skipped_no_payload', 0)} scan(s) with no fits, "
                f"{stats.get('skipped_unaccepted', 0)} unaccepted fit(s), "
                f"{stats.get('skipped_invalid', 0)} invalid fit(s)."
            )

    def open_resonance_fit_offset_window(self) -> None:
        scans = self._selected_scans()
        if not scans:
            messagebox.showwarning(
                "No selection",
                "No scans selected for analysis.\nUse 'Select Scans for Analysis' first.",
            )
            return

        power_rows = []
        missing_power = []
        for scan in scans:
            try:
                power_dbm = float(getattr(scan, "bias_power_dBm", np.nan))
            except Exception:
                power_dbm = np.nan
            if np.isfinite(power_dbm):
                power_rows.append((power_dbm, scan))
            else:
                missing_power.append(scan)
        if not power_rows:
            messagebox.showwarning(
                "Missing drive powers",
                "No selected VNA scan has a finite drive power.\nUse 'Update Drive Powers From Filename' first.",
            )
            return
        if missing_power:
            names = "\n".join(Path(getattr(scan, "filename", "")).name for scan in missing_power[:8])
            more = "" if len(missing_power) <= 8 else f"\n... and {len(missing_power) - 8} more"
            messagebox.showwarning(
                "Some powers missing",
                "Only selected scans with finite drive power will be used to find the lowest-power reference.\n\n"
                f"Missing drive power:\n{names}{more}",
            )

        lowest_power_dbm, reference_scan = min(power_rows, key=lambda row: row[0])
        points, stats = self._res_fit_offset_collect_points(scans, reference_scan)
        if not points:
            messagebox.showwarning(
                "No accepted offset points",
                "No accepted fits could be compared.\n\n"
                "The lowest-power selected scan must have accepted fits for the resonator numbers being compared.",
            )
            return

        if getattr(self, "res_fit_offset_window", None) is not None and self.res_fit_offset_window.winfo_exists():
            self.res_fit_offset_points = points
            self.res_fit_offset_stats = stats
            self.res_fit_offset_window.lift()
            self._res_fit_offset_render()
            return

        self.res_fit_offset_points = points
        self.res_fit_offset_stats = stats
        self.res_fit_offset_window = tk.Toplevel(self.root)
        self.res_fit_offset_window.title("Check Fitted Frequency Offsets")
        self.res_fit_offset_window.geometry("1180x820")
        self.res_fit_offset_window.protocol("WM_DELETE_WINDOW", self._res_fit_offset_close)

        controls = tk.Frame(self.res_fit_offset_window, padx=8, pady=6)
        controls.pack(side="top", fill="x")
        self.res_fit_offset_status_var = tk.StringVar()
        tk.Label(controls, textvariable=self.res_fit_offset_status_var, anchor="w").pack(side="top", fill="x")

        radio_row = tk.Frame(controls)
        radio_row.pack(side="top", fill="x", pady=(6, 0))
        self.res_fit_offset_y_var = tk.StringVar(value="residual_rel")
        self.res_fit_offset_x_var = tk.StringVar(value="reference_frequency")
        self.res_fit_offset_color_var = tk.StringVar(value="scan")

        x_frame = tk.LabelFrame(radio_row, text="X axis", padx=6, pady=4)
        x_frame.pack(side="left", padx=(0, 10), fill="y")
        tk.Radiobutton(
            x_frame,
            text="reference frequency",
            variable=self.res_fit_offset_x_var,
            value="reference_frequency",
            command=self._res_fit_offset_render,
        ).pack(anchor="w")
        tk.Radiobutton(
            x_frame,
            text="nonlinear parameter a",
            variable=self.res_fit_offset_x_var,
            value="a_nl",
            command=self._res_fit_offset_render,
        ).pack(anchor="w")
        tk.Radiobutton(
            x_frame,
            text="true offset / fr_r",
            variable=self.res_fit_offset_x_var,
            value="true_rel",
            command=self._res_fit_offset_render,
        ).pack(anchor="w")
        tk.Radiobutton(
            x_frame,
            text="bias power",
            variable=self.res_fit_offset_x_var,
            value="bias_power",
            command=self._res_fit_offset_render,
        ).pack(anchor="w")

        y_frame = tk.LabelFrame(radio_row, text="Y axis", padx=6, pady=4)
        y_frame.pack(side="left", padx=(0, 10), fill="y")
        tk.Radiobutton(
            y_frame,
            text="(model offset - true offset) / fr_r",
            variable=self.res_fit_offset_y_var,
            value="residual_rel",
            command=self._res_fit_offset_render,
        ).pack(anchor="w")
        tk.Radiobutton(
            y_frame,
            text="model offset / true offset",
            variable=self.res_fit_offset_y_var,
            value="ratio",
            command=self._res_fit_offset_render,
        ).pack(anchor="w")
        tk.Radiobutton(
            y_frame,
            text="model offset / fr_r",
            variable=self.res_fit_offset_y_var,
            value="model_rel",
            command=self._res_fit_offset_render,
        ).pack(anchor="w")
        tk.Radiobutton(
            y_frame,
            text="true offset / fr_r",
            variable=self.res_fit_offset_y_var,
            value="true_rel",
            command=self._res_fit_offset_render,
        ).pack(anchor="w")
        tk.Radiobutton(
            y_frame,
            text="Q internal",
            variable=self.res_fit_offset_y_var,
            value="q_internal",
            command=self._res_fit_offset_render,
        ).pack(anchor="w")
        tk.Radiobutton(
            y_frame,
            text="Q coupling",
            variable=self.res_fit_offset_y_var,
            value="q_coupling",
            command=self._res_fit_offset_render,
        ).pack(anchor="w")

        color_frame = tk.LabelFrame(radio_row, text="Color", padx=6, pady=4)
        color_frame.pack(side="left", fill="y")
        tk.Radiobutton(
            color_frame,
            text="VNA scan",
            variable=self.res_fit_offset_color_var,
            value="scan",
            command=self._res_fit_offset_render,
        ).pack(anchor="w")
        tk.Radiobutton(
            color_frame,
            text="nonlinear parameter a",
            variable=self.res_fit_offset_color_var,
            value="a_nl",
            command=self._res_fit_offset_render,
        ).pack(anchor="w")
        tk.Radiobutton(
            color_frame,
            text="reference frequency",
            variable=self.res_fit_offset_color_var,
            value="reference_frequency",
            command=self._res_fit_offset_render,
        ).pack(anchor="w")
        tk.Radiobutton(
            color_frame,
            text="fit quality rank",
            variable=self.res_fit_offset_color_var,
            value="fit_quality_rank",
            command=self._res_fit_offset_render,
        ).pack(anchor="w")

        option_row = tk.Frame(controls)
        option_row.pack(side="top", fill="x", pady=(6, 0))
        self.res_fit_offset_connect_resonators_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            option_row,
            text="Connect same resonator by increasing bias power",
            variable=self.res_fit_offset_connect_resonators_var,
            command=self._res_fit_offset_render,
        ).pack(side="left")

        self.res_fit_offset_figure = Figure(figsize=(11, 7))
        self.res_fit_offset_canvas = FigureCanvasTkAgg(self.res_fit_offset_figure, master=self.res_fit_offset_window)
        self.res_fit_offset_toolbar = NavigationToolbar2Tk(self.res_fit_offset_canvas, self.res_fit_offset_window)
        self.res_fit_offset_toolbar.update()
        self.res_fit_offset_toolbar.pack(side="top", fill="x")
        self.res_fit_offset_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.res_fit_offset_pick_points = []
        self.res_fit_offset_pick_cid = self.res_fit_offset_canvas.mpl_connect(
            "button_press_event", self._res_fit_offset_on_click
        )
        self._res_fit_offset_render()

    def _res_fit_offset_close(self) -> None:
        if (
            getattr(self, "res_fit_offset_canvas", None) is not None
            and getattr(self, "res_fit_offset_pick_cid", None) is not None
        ):
            self.res_fit_offset_canvas.mpl_disconnect(self.res_fit_offset_pick_cid)
        if getattr(self, "res_fit_offset_window", None) is not None and self.res_fit_offset_window.winfo_exists():
            self.res_fit_offset_window.destroy()
        self.res_fit_offset_window = None
        self.res_fit_offset_canvas = None
        self.res_fit_offset_toolbar = None
        self.res_fit_offset_figure = None
        self.res_fit_offset_status_var = None
        self.res_fit_offset_pick_cid = None
        self.res_fit_offset_pick_points = []
        self.res_fit_offset_points = []
        self.res_fit_offset_stats = {}

    def _res_fit_offset_collect_points(self, scans: list[object], reference_scan: object) -> tuple[list[dict], dict]:
        reference_key = self._scan_key(reference_scan)
        reference_payload = reference_scan.candidate_resonators.get("logan_nonlinear_iq_marker_fits")
        reference_assignments = reference_payload.get("assignments") if isinstance(reference_payload, dict) else {}
        if not isinstance(reference_assignments, dict):
            reference_assignments = {}

        reference_by_res = {}
        for resonator_number, payload in reference_assignments.items():
            if not isinstance(payload, dict) or not payload.get("success") or not bool(payload.get("accepted", False)):
                continue
            try:
                fr0_hz = float(payload.get("fr0_hz", np.nan))
            except Exception:
                fr0_hz = np.nan
            if np.isfinite(fr0_hz):
                reference_by_res[str(resonator_number)] = {"fr0_hz": fr0_hz, "payload": payload}

        points = []
        skipped_not_in_reference = 0
        skipped_invalid = 0
        skipped_unaccepted = 0
        scan_label_by_key = {}
        for scan_index, scan in enumerate(scans):
            scan_key = self._scan_key(scan)
            scan_label_by_key[str(scan_key)] = Path(getattr(scan, "filename", "")).name or str(scan_key)
            fit_payload = scan.candidate_resonators.get("logan_nonlinear_iq_marker_fits")
            assignments = fit_payload.get("assignments") if isinstance(fit_payload, dict) else {}
            if not isinstance(assignments, dict):
                continue
            for resonator_number, payload in assignments.items():
                if not isinstance(payload, dict) or not payload.get("success"):
                    skipped_invalid += 1
                    continue
                if not bool(payload.get("accepted", False)):
                    skipped_unaccepted += 1
                    continue
                resonator_key = str(resonator_number)
                reference_entry = reference_by_res.get(resonator_key)
                if reference_entry is None:
                    skipped_not_in_reference += 1
                    continue
                reference_fr0_hz = float(reference_entry["fr0_hz"])
                try:
                    true_fr_hz = float(payload.get("true_fr_hz", np.nan))
                    model_offset_hz = float(payload.get("delta_fr_hz", np.nan))
                    a_nl = float(payload.get("a_nl", np.nan))
                    nrmse = float(payload.get("nrmse", np.nan))
                    bias_power_dbm = float(getattr(scan, "bias_power_dBm", np.nan))
                except Exception:
                    skipped_invalid += 1
                    continue
                true_offset_hz = true_fr_hz - reference_fr0_hz
                if not all(np.isfinite(value) for value in (true_fr_hz, model_offset_hz, true_offset_hz, a_nl)):
                    skipped_invalid += 1
                    continue
                points.append(
                    {
                        "scan": scan,
                        "scan_key": str(scan_key),
                        "scan_index": int(scan_index),
                        "scan_label": scan_label_by_key[str(scan_key)],
                        "resonator_number": resonator_key,
                        "payload": payload,
                        "reference_scan": reference_scan,
                        "reference_payload": reference_entry["payload"],
                        "reference_scan_key": str(reference_key),
                        "reference_fr0_hz": float(reference_fr0_hz),
                        "true_fr_hz": float(true_fr_hz),
                        "true_offset_hz": float(true_offset_hz),
                        "model_offset_hz": float(model_offset_hz),
                        "a_nl": float(a_nl),
                        "nrmse": float(nrmse),
                        "bias_power_dbm": float(bias_power_dbm),
                    }
                )

        try:
            reference_power_dbm = float(getattr(reference_scan, "bias_power_dBm", np.nan))
        except Exception:
            reference_power_dbm = np.nan
        stats = {
            "reference_scan": reference_scan,
            "reference_scan_key": str(reference_key),
            "reference_scan_label": Path(getattr(reference_scan, "filename", "")).name or str(reference_key),
            "reference_power_dbm": float(reference_power_dbm),
            "reference_resonator_count": int(len(reference_by_res)),
            "point_count": int(len(points)),
            "skipped_not_in_reference": int(skipped_not_in_reference),
            "skipped_invalid": int(skipped_invalid),
            "skipped_unaccepted": int(skipped_unaccepted),
            "scan_label_by_key": scan_label_by_key,
        }
        return points, stats

    def _res_fit_offset_render(self) -> None:
        if getattr(self, "res_fit_offset_figure", None) is None or self.res_fit_offset_canvas is None:
            return
        points = list(getattr(self, "res_fit_offset_points", []))
        stats = dict(getattr(self, "res_fit_offset_stats", {}))
        self.res_fit_offset_figure.clear()
        ax = self.res_fit_offset_figure.add_subplot(1, 1, 1)
        if not points:
            ax.text(0.5, 0.5, "No accepted offset points.", ha="center", va="center")
            self.res_fit_offset_canvas.draw_idle()
            return

        reference_fr0 = np.asarray([point["reference_fr0_hz"] for point in points], dtype=float)
        true_offset = np.asarray([point["true_offset_hz"] for point in points], dtype=float)
        model_offset = np.asarray([point["model_offset_hz"] for point in points], dtype=float)
        a_values = np.asarray([point["a_nl"] for point in points], dtype=float)
        nrmse_values = np.asarray([point.get("nrmse", np.nan) for point in points], dtype=float)
        bias_power_values = np.asarray([point.get("bias_power_dbm", np.nan) for point in points], dtype=float)
        def _payload_float(point: dict, key: str) -> float:
            try:
                return float(point.get("payload", {}).get(key, np.nan))
            except Exception:
                return np.nan

        q_internal_values = np.asarray([_payload_float(point, "q_internal") for point in points], dtype=float)
        q_coupling_values = np.asarray([_payload_float(point, "q_coupling") for point in points], dtype=float)
        scan_indices = np.asarray([point["scan_index"] for point in points], dtype=float)

        x_mode = getattr(self, "res_fit_offset_x_var", tk.StringVar(value="reference_frequency")).get()
        y_mode = getattr(self, "res_fit_offset_y_var", tk.StringVar(value="residual_rel")).get()
        color_mode = getattr(self, "res_fit_offset_color_var", tk.StringVar(value="scan")).get()

        if x_mode == "a_nl":
            x = a_values
            x_label = "Nonlinear parameter a"
        elif x_mode == "true_rel":
            x = true_offset / reference_fr0
            x_label = "True offset / reference fr0"
        elif x_mode == "bias_power":
            x = bias_power_values
            x_label = "Bias power (dBm)"
        else:
            x = reference_fr0 / _HZ_PER_GHZ
            x_label = "Reference fr0 from lowest-power scan (GHz)"

        if y_mode == "ratio":
            y = np.divide(
                model_offset,
                true_offset,
                out=np.full_like(model_offset, np.nan, dtype=float),
                where=np.abs(true_offset) > 0.0,
            )
            y_label = "Model offset / true offset"
        elif y_mode == "model_rel":
            y = model_offset / reference_fr0
            y_label = "Model offset / reference fr0"
        elif y_mode == "true_rel":
            y = true_offset / reference_fr0
            y_label = "True offset / reference fr0"
        elif y_mode == "q_internal":
            y = q_internal_values
            y_label = "Internal Q"
        elif y_mode == "q_coupling":
            y = q_coupling_values
            y_label = "Coupling Q"
        else:
            y = (model_offset - true_offset) / reference_fr0
            y_label = "(model offset - true offset) / reference fr0"

        finite = np.isfinite(x) & np.isfinite(y)
        if color_mode == "a_nl":
            finite &= np.isfinite(a_values)
        elif color_mode == "reference_frequency":
            finite &= np.isfinite(reference_fr0)
        elif color_mode == "fit_quality_rank":
            finite &= np.isfinite(nrmse_values)
        else:
            finite &= np.isfinite(scan_indices)

        self.res_fit_offset_pick_points = []
        if not np.any(finite):
            ax.text(0.5, 0.5, "No finite points for the selected axes.", ha="center", va="center")
        else:
            connect_var = getattr(self, "res_fit_offset_connect_resonators_var", None)
            connect_resonators = bool(connect_var.get()) if connect_var is not None else False
            if connect_resonators:
                grouped: dict[str, list[int]] = {}
                for point_index in np.where(finite)[0]:
                    grouped.setdefault(str(points[int(point_index)].get("resonator_number", "")), []).append(
                        int(point_index)
                    )
                for indices in grouped.values():
                    if len(indices) < 2:
                        continue
                    order = sorted(indices, key=lambda point_index: bias_power_values[point_index])
                    ax.plot(
                        x[order],
                        y[order],
                        color="0.65",
                        linewidth=0.9,
                        alpha=0.65,
                        zorder=1,
                    )

            if color_mode == "a_nl":
                scatter = ax.scatter(
                    x[finite],
                    y[finite],
                    c=a_values[finite],
                    cmap="rainbow",
                    s=36,
                    edgecolors="0.2",
                    zorder=2,
                )
                cbar = self.res_fit_offset_figure.colorbar(scatter, ax=ax)
                cbar.set_label("Nonlinear parameter a")
            elif color_mode == "reference_frequency":
                scatter = ax.scatter(
                    x[finite],
                    y[finite],
                    c=reference_fr0[finite] / _HZ_PER_GHZ,
                    cmap="rainbow_r",
                    s=36,
                    edgecolors="0.2",
                    zorder=2,
                )
                cbar = self.res_fit_offset_figure.colorbar(scatter, ax=ax)
                cbar.set_label("Reference fr0 (GHz)")
            elif color_mode == "fit_quality_rank":
                finite_indices_for_rank = np.where(finite)[0]
                rank_values = np.full_like(nrmse_values, np.nan, dtype=float)
                order = finite_indices_for_rank[np.argsort(nrmse_values[finite_indices_for_rank])]
                for rank, point_index in enumerate(order, start=1):
                    rank_values[point_index] = float(rank)
                scatter = ax.scatter(
                    x[finite],
                    y[finite],
                    c=rank_values[finite],
                    cmap="rainbow",
                    s=36,
                    edgecolors="0.2",
                    vmin=0.5,
                    vmax=float(len(order)) + 0.5,
                    zorder=2,
                )
                cbar = self.res_fit_offset_figure.colorbar(scatter, ax=ax)
                cbar.set_label("Fit quality rank by nrmse (1 = best)")
            else:
                unique_indices = sorted({int(value) for value in scan_indices[finite]})
                color_lookup = {}
                scan_cmap = colormaps.get_cmap("rainbow")
                denom = max(1, len(unique_indices) - 1)
                for pos, scan_index in enumerate(unique_indices):
                    color_lookup[scan_index] = scan_cmap(pos / denom)
                for scan_index in unique_indices:
                    mask = finite & (scan_indices == float(scan_index))
                    label = points[int(np.where(scan_indices == float(scan_index))[0][0])]["scan_label"]
                    ax.scatter(
                        x[mask],
                        y[mask],
                        color=color_lookup[scan_index],
                        s=36,
                        edgecolors="0.2",
                        label=label if len(unique_indices) <= 12 else None,
                        zorder=2,
                    )
                if len(unique_indices) <= 12:
                    ax.legend(loc="best", fontsize=8)
        finite_indices = np.where(finite)[0]
        for point_index in finite_indices:
            self.res_fit_offset_pick_points.append(
                {
                    "x": float(x[point_index]),
                    "y": float(y[point_index]),
                    "point": points[int(point_index)],
                }
            )

        ax.axhline(0.0, color="0.4", linewidth=1.0, linestyle=":")
        if y_mode == "ratio":
            ax.axhline(1.0, color="0.25", linewidth=1.0, linestyle="--")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        ref_label = stats.get("reference_scan_label", "")
        ref_power = stats.get("reference_power_dbm", np.nan)
        ax.set_title(f"Accepted fits vs offset theory | reference {ref_label} ({ref_power:g} dBm)", fontsize=10)
        self.res_fit_offset_figure.tight_layout()
        self.res_fit_offset_canvas.draw_idle()

        displayed = int(np.count_nonzero(finite))
        if getattr(self, "res_fit_offset_status_var", None) is not None:
            self.res_fit_offset_status_var.set(
                f"Reference: {ref_label} at {ref_power:g} dBm. "
                f"{displayed}/{len(points)} point(s) displayed; "
                f"{stats.get('reference_resonator_count', 0)} reference resonator(s); "
                f"skipped {stats.get('skipped_not_in_reference', 0)} not in reference, "
                f"{stats.get('skipped_unaccepted', 0)} not accepted, "
                f"{stats.get('skipped_invalid', 0)} invalid."
            )

    def _res_fit_offset_on_click(self, event) -> None:
        if event.inaxes is None or event.x is None or event.y is None:
            return
        pick_points = list(getattr(self, "res_fit_offset_pick_points", []))
        if not pick_points:
            return
        distances = []
        for entry in pick_points:
            try:
                display_x, display_y = event.inaxes.transData.transform((entry["x"], entry["y"]))
            except Exception:
                continue
            distances.append((float(np.hypot(display_x - event.x, display_y - event.y)), entry))
        if not distances:
            return
        distance, nearest = min(distances, key=lambda row: row[0])
        if distance > 12.0:
            return
        self._res_fit_offset_open_fit_detail(nearest["point"])

    def _res_fit_offset_open_fit_detail(self, point: dict) -> None:
        if getattr(self, "res_fit_offset_detail_window", None) is not None and self.res_fit_offset_detail_window.winfo_exists():
            self.res_fit_offset_detail_window.destroy()

        self.res_fit_offset_detail_point = point
        self.res_fit_offset_detail_window = tk.Toplevel(self.root)
        self.res_fit_offset_detail_window.title("Accepted Fit Detail")
        self.res_fit_offset_detail_window.geometry("1180x760")
        self.res_fit_offset_detail_window.protocol("WM_DELETE_WINDOW", self._res_fit_offset_detail_close)

        controls = tk.Frame(self.res_fit_offset_detail_window, padx=8, pady=6)
        controls.pack(side="top", fill="x")
        output_row = tk.Frame(controls)
        output_row.pack(side="top", fill="x")
        self.res_fit_offset_detail_output_vars = {}
        for label, key, width in (
            ("Scan", "scan", 24),
            ("Res #", "resonator_number", 7),
            ("ref fr0 GHz", "reference_fr0_ghz", 11),
            ("powered fr GHz", "true_fr_ghz", 12),
            ("true offset Hz", "true_offset_hz", 12),
            ("model offset Hz", "model_offset_hz", 13),
            ("model/true", "ratio", 10),
            ("Qr", "qr", 10),
            ("Qi", "qi", 10),
            ("Qc", "qc", 10),
            ("a", "a_nl", 9),
            ("nrmse", "nrmse", 10),
        ):
            tk.Label(output_row, text=label).pack(side="left", padx=(0, 2))
            var = tk.StringVar()
            self.res_fit_offset_detail_output_vars[key] = var
            tk.Entry(output_row, textvariable=var, width=width, state="readonly").pack(side="left", padx=(0, 6))

        reference_row = tk.Frame(controls)
        reference_row.pack(side="top", fill="x", pady=(6, 0))
        tk.Label(reference_row, text="Reference").pack(side="left", padx=(0, 6))
        self.res_fit_offset_detail_reference_vars = {}
        for label, key, width in (
            ("Scan", "scan", 24),
            ("fr0 GHz", "fr0_ghz", 11),
            ("powered fr GHz", "true_fr_ghz", 12),
            ("delta fr Hz", "delta_fr_hz", 12),
            ("Qr", "qr", 10),
            ("Qi", "qi", 10),
            ("Qc", "qc", 10),
            ("a", "a_nl", 9),
            ("nrmse", "nrmse", 10),
        ):
            tk.Label(reference_row, text=label).pack(side="left", padx=(0, 2))
            var = tk.StringVar()
            self.res_fit_offset_detail_reference_vars[key] = var
            tk.Entry(reference_row, textvariable=var, width=width, state="readonly").pack(side="left", padx=(0, 6))

        option_row = tk.Frame(controls)
        option_row.pack(side="top", fill="x", pady=(6, 0))
        self.res_fit_offset_detail_show_reference_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            option_row,
            text="Show lowest-power reference fit",
            variable=self.res_fit_offset_detail_show_reference_var,
            command=self._res_fit_offset_render_fit_detail,
        ).pack(side="left")

        self.res_fit_offset_detail_figure = Figure(figsize=(11, 6.8))
        self.res_fit_offset_detail_canvas = FigureCanvasTkAgg(
            self.res_fit_offset_detail_figure, master=self.res_fit_offset_detail_window
        )
        self.res_fit_offset_detail_toolbar = NavigationToolbar2Tk(
            self.res_fit_offset_detail_canvas, self.res_fit_offset_detail_window
        )
        self.res_fit_offset_detail_toolbar.update()
        self.res_fit_offset_detail_toolbar.pack(side="top", fill="x")
        self.res_fit_offset_detail_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._res_fit_offset_render_fit_detail()

    def _res_fit_offset_detail_close(self) -> None:
        if (
            getattr(self, "res_fit_offset_detail_window", None) is not None
            and self.res_fit_offset_detail_window.winfo_exists()
        ):
            self.res_fit_offset_detail_window.destroy()
        self.res_fit_offset_detail_window = None
        self.res_fit_offset_detail_canvas = None
        self.res_fit_offset_detail_toolbar = None
        self.res_fit_offset_detail_figure = None
        self.res_fit_offset_detail_output_vars = {}
        self.res_fit_offset_detail_reference_vars = {}
        self.res_fit_offset_detail_show_reference_var = None
        self.res_fit_offset_detail_point = None

    def _res_fit_offset_render_fit_detail(self) -> None:
        point = getattr(self, "res_fit_offset_detail_point", None)
        if (
            point is None
            or getattr(self, "res_fit_offset_detail_figure", None) is None
            or self.res_fit_offset_detail_canvas is None
        ):
            return
        payload = point["payload"]
        vars_map = getattr(self, "res_fit_offset_detail_output_vars", {})
        for var in vars_map.values():
            var.set("")
        vars_map["scan"].set(str(point.get("scan_label", ""))[:24])
        vars_map["resonator_number"].set(str(point.get("resonator_number", "")))
        vars_map["reference_fr0_ghz"].set(f"{float(point.get('reference_fr0_hz', np.nan)) / _HZ_PER_GHZ:.9g}")
        vars_map["true_fr_ghz"].set(f"{float(point.get('true_fr_hz', np.nan)) / _HZ_PER_GHZ:.9g}")
        vars_map["true_offset_hz"].set(f"{float(point.get('true_offset_hz', np.nan)):.9g}")
        vars_map["model_offset_hz"].set(f"{float(point.get('model_offset_hz', np.nan)):.9g}")
        true_offset = float(point.get("true_offset_hz", np.nan))
        model_offset = float(point.get("model_offset_hz", np.nan))
        ratio = model_offset / true_offset if np.isfinite(true_offset) and true_offset != 0.0 else np.nan
        vars_map["ratio"].set(f"{ratio:.9g}" if np.isfinite(ratio) else "")
        vars_map["qr"].set(f"{float(payload.get('q_loaded', np.nan)):.9g}")
        vars_map["qi"].set(f"{float(payload.get('q_internal', np.nan)):.9g}")
        vars_map["qc"].set(f"{float(payload.get('q_coupling', np.nan)):.9g}")
        vars_map["a_nl"].set(self._res_fit_offset_format_decimal(point.get("a_nl", np.nan)))
        vars_map["nrmse"].set(f"{float(payload.get('nrmse', np.nan)):.9g}")

        reference_vars = getattr(self, "res_fit_offset_detail_reference_vars", {})
        for var in reference_vars.values():
            var.set("")

        self.res_fit_offset_detail_figure.clear()
        ax_amp = self.res_fit_offset_detail_figure.add_subplot(1, 2, 1)
        ax_iq = self.res_fit_offset_detail_figure.add_subplot(1, 2, 2)
        show_reference_var = getattr(self, "res_fit_offset_detail_show_reference_var", None)
        show_reference = bool(show_reference_var.get()) if show_reference_var is not None else False
        reference_scan = point.get("reference_scan")
        reference_payload = point.get("reference_payload")
        if show_reference and reference_scan is not None and isinstance(reference_payload, dict):
            reference_vars["scan"].set(str(Path(getattr(reference_scan, "filename", "")).name)[:24])
            reference_vars["fr0_ghz"].set(f"{float(reference_payload.get('fr0_hz', np.nan)) / _HZ_PER_GHZ:.9g}")
            reference_vars["true_fr_ghz"].set(
                f"{float(reference_payload.get('true_fr_hz', np.nan)) / _HZ_PER_GHZ:.9g}"
            )
            reference_vars["delta_fr_hz"].set(f"{float(reference_payload.get('delta_fr_hz', np.nan)):.9g}")
            reference_vars["qr"].set(f"{float(reference_payload.get('q_loaded', np.nan)):.9g}")
            reference_vars["qi"].set(f"{float(reference_payload.get('q_internal', np.nan)):.9g}")
            reference_vars["qc"].set(f"{float(reference_payload.get('q_coupling', np.nan)):.9g}")
            reference_vars["a_nl"].set(self._res_fit_offset_format_decimal(reference_payload.get("a_nl", np.nan)))
            reference_vars["nrmse"].set(f"{float(reference_payload.get('nrmse', np.nan)):.9g}")
            ref_marker_hz = reference_payload.get("marker_frequency_hz")
            try:
                ref_marker_hz = float(ref_marker_hz)
            except Exception:
                ref_marker_hz = float(reference_payload.get("fr0_hz", np.nan))
            ref_item = {
                "scan": reference_scan,
                "payload": reference_payload,
                "marker_hz": ref_marker_hz,
                "resonator_number": str(point.get("resonator_number", "")),
                "rank": "reference",
                "title": "",
            }
            self._res_fit_quality_draw_fit_axes(
                ax_amp,
                ax_iq,
                ref_item,
                raw_color="0.70",
                fit_color="0.35",
                marker_color="0.55",
                fr0_color="mediumpurple",
                true_color="lightcoral",
                alpha=0.55,
                label_prefix="ref ",
                zorder=1,
                set_title=False,
            )
        marker_hz = payload.get("marker_frequency_hz")
        try:
            marker_hz = float(marker_hz)
        except Exception:
            marker_hz = float(payload.get("fr0_hz", np.nan))
        item = {
            "scan": point["scan"],
            "payload": payload,
            "marker_hz": marker_hz,
            "resonator_number": str(point.get("resonator_number", "")),
            "rank": "offset",
            "title": (
                f"Selected fit | {Path(getattr(point['scan'], 'filename', '')).name} | "
                f"Resonator {point.get('resonator_number', '')}"
            ),
        }
        self._res_fit_quality_draw_fit_axes(ax_amp, ax_iq, item, zorder=3)
        self.res_fit_offset_detail_figure.tight_layout()
        self.res_fit_offset_detail_canvas.draw_idle()

    @staticmethod
    def _res_fit_offset_format_decimal(value: object) -> str:
        try:
            number = float(value)
        except Exception:
            return ""
        if not np.isfinite(number):
            return ""
        if abs(number) < 0.5e-12:
            number = 0.0
        return f"{number:.12f}".rstrip("0").rstrip(".")

    def open_resonance_fit_quality_window(self) -> None:
        scans = self._selected_scans()
        if not scans:
            messagebox.showwarning(
                "No selection",
                "No scans selected for analysis.\nUse 'Select Scans for Analysis' first.",
            )
            return
        ranked = self._res_fit_quality_collect_ranked(scans)
        if not ranked:
            messagebox.showwarning(
                "No batch fits",
                "No successful attached marker fit results were found on the selected scans.",
            )
            return
        if getattr(self, "res_fit_quality_window", None) is not None and self.res_fit_quality_window.winfo_exists():
            self.res_fit_quality_window.lift()
            return

        self.res_fit_quality_ranked = ranked
        self.res_fit_quality_window = tk.Toplevel(self.root)
        self.res_fit_quality_window.title("Rank Resonator Fits by NRMSE")
        self.res_fit_quality_window.geometry("1320x900")
        self.res_fit_quality_window.protocol("WM_DELETE_WINDOW", self._res_fit_quality_close)

        controls = tk.Frame(self.res_fit_quality_window, padx=8, pady=6)
        controls.pack(side="top", fill="x")
        top_row = tk.Frame(controls)
        top_row.pack(side="top", fill="x")
        self.res_fit_quality_status_var = tk.StringVar(
            value=f"Loaded {len(ranked)} successful attached fit(s), ranked by nrmse."
        )
        tk.Label(top_row, textvariable=self.res_fit_quality_status_var, anchor="w").pack(
            side="left", fill="x", expand=True
        )
        tk.Button(top_row, text="Accept Good or Better", width=20, command=self._res_fit_quality_accept_threshold).pack(
            side="right", padx=(8, 0)
        )
        tk.Button(top_row, text="Accept All", width=12, command=self._res_fit_quality_accept_all).pack(
            side="right", padx=(8, 0)
        )

        slider_row = tk.Frame(controls)
        slider_row.pack(side="top", fill="x", pady=(6, 0))
        tk.Label(slider_row, text="Fit rank by nrmse").pack(side="left", padx=(0, 6))
        self.res_fit_quality_rank_var = tk.IntVar(value=0)
        self.res_fit_quality_rank_slider = tk.Scale(
            slider_row,
            from_=0,
            to=max(0, len(ranked) - 1),
            orient="horizontal",
            length=520,
            showvalue=True,
            variable=self.res_fit_quality_rank_var,
            command=lambda _v: self._res_fit_quality_render_selected(),
        )
        self.res_fit_quality_rank_slider.pack(side="left")
        self.res_fit_quality_rank_slider.bind("<ButtonRelease-1>", lambda _e: self._res_fit_quality_render_selected())

        output_row = tk.Frame(controls)
        output_row.pack(side="top", fill="x", pady=(6, 0))
        self.res_fit_quality_output_vars = {}
        for label, key, width in (
            ("Rank", "rank", 6),
            ("Res #", "resonator_number", 7),
            ("Marker GHz", "marker_ghz", 11),
            ("fr0 GHz", "fr0_ghz", 11),
            ("powered fr GHz", "true_fr_ghz", 12),
            ("delta fr Hz", "delta_fr_hz", 12),
            ("Qr", "qr", 10),
            ("Qi", "qi", 10),
            ("Qc", "qc", 10),
            ("a", "a_nl", 9),
            ("nrmse", "nrmse", 10),
            ("Accepted", "accepted", 9),
        ):
            tk.Label(output_row, text=label).pack(side="left", padx=(0, 2))
            var = tk.StringVar()
            self.res_fit_quality_output_vars[key] = var
            tk.Entry(output_row, textvariable=var, width=width, state="readonly").pack(side="left", padx=(0, 6))

        self.res_fit_quality_figure = Figure(figsize=(12, 7.5))
        self.res_fit_quality_canvas = FigureCanvasTkAgg(self.res_fit_quality_figure, master=self.res_fit_quality_window)
        self.res_fit_quality_toolbar = NavigationToolbar2Tk(self.res_fit_quality_canvas, self.res_fit_quality_window)
        self.res_fit_quality_toolbar.update()
        self.res_fit_quality_toolbar.pack(side="top", fill="x")
        self.res_fit_quality_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._res_fit_quality_render_selected()

    def _res_fit_quality_close(self) -> None:
        if getattr(self, "res_fit_quality_window", None) is not None and self.res_fit_quality_window.winfo_exists():
            self.res_fit_quality_window.destroy()
        self.res_fit_quality_window = None
        self.res_fit_quality_canvas = None
        self.res_fit_quality_toolbar = None
        self.res_fit_quality_figure = None
        self.res_fit_quality_status_var = None
        self.res_fit_quality_rank_slider = None
        self.res_fit_quality_output_vars = {}
        self.res_fit_quality_ranked = []

    def _res_fit_quality_collect_ranked(self, scans: list[object]) -> list[dict]:
        items = []
        for scan in scans:
            scan_key = self._scan_key(scan)
            fit_payload = scan.candidate_resonators.get("logan_nonlinear_iq_marker_fits")
            assignments = fit_payload.get("assignments") if isinstance(fit_payload, dict) else {}
            if not isinstance(assignments, dict):
                continue
            for resonator_number, payload in assignments.items():
                if not isinstance(payload, dict) or not payload.get("success"):
                    continue
                try:
                    nrmse = float(payload.get("nrmse"))
                except Exception:
                    continue
                if not np.isfinite(nrmse):
                    continue
                marker_hz = payload.get("marker_frequency_hz")
                try:
                    marker_hz = float(marker_hz)
                except Exception:
                    marker_hz = float(payload.get("fr0_hz", np.nan))
                items.append(
                    {
                        "scan": scan,
                        "scan_key": scan_key,
                        "resonator_number": str(resonator_number),
                        "marker_hz": marker_hz,
                        "payload": payload,
                        "nrmse": nrmse,
                    }
                )
        ranked = sorted(items, key=lambda item: item["nrmse"])
        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank
        return ranked

    def _res_fit_quality_current_item(self) -> Optional[dict]:
        ranked = list(getattr(self, "res_fit_quality_ranked", []))
        if not ranked:
            return None
        idx = int(getattr(self, "res_fit_quality_rank_var", tk.IntVar(value=0)).get())
        idx = min(max(idx, 0), len(ranked) - 1)
        return ranked[idx]

    def _res_fit_quality_update_output_fields(self, item: Optional[dict]) -> None:
        vars_map = getattr(self, "res_fit_quality_output_vars", {})
        for var in vars_map.values():
            var.set("")
        if item is None:
            return
        payload = item["payload"]
        marker_hz = float(item.get("marker_hz", np.nan))
        vars_map["rank"].set(f"{int(item['rank'])}/{len(getattr(self, 'res_fit_quality_ranked', []))}")
        vars_map["resonator_number"].set(str(item.get("resonator_number", "")))
        vars_map["marker_ghz"].set(f"{marker_hz / _HZ_PER_GHZ:.9g}" if np.isfinite(marker_hz) else "")
        vars_map["fr0_ghz"].set(f"{float(payload.get('fr0_hz', np.nan)) / _HZ_PER_GHZ:.9g}")
        vars_map["true_fr_ghz"].set(f"{float(payload.get('true_fr_hz', np.nan)) / _HZ_PER_GHZ:.9g}")
        vars_map["delta_fr_hz"].set(f"{float(payload.get('delta_fr_hz', np.nan)):.9g}")
        vars_map["qr"].set(f"{float(payload.get('q_loaded', np.nan)):.9g}")
        vars_map["qi"].set(f"{float(payload.get('q_internal', np.nan)):.9g}")
        vars_map["qc"].set(f"{float(payload.get('q_coupling', np.nan)):.9g}")
        vars_map["a_nl"].set(f"{float(payload.get('a_nl', np.nan)):.9g}")
        vars_map["nrmse"].set(f"{float(payload.get('nrmse', np.nan)):.9g}")
        accepted = payload.get("accepted")
        vars_map["accepted"].set("" if accepted is None else ("yes" if bool(accepted) else "no"))

    def _res_fit_quality_render_selected(self) -> None:
        if getattr(self, "res_fit_quality_figure", None) is None or self.res_fit_quality_canvas is None:
            return
        item = self._res_fit_quality_current_item()
        self._res_fit_quality_update_output_fields(item)
        self.res_fit_quality_figure.clear()
        ax_amp = self.res_fit_quality_figure.add_subplot(2, 2, 1)
        ax_iq = self.res_fit_quality_figure.add_subplot(2, 2, 2)
        ax_rank = self.res_fit_quality_figure.add_subplot(2, 1, 2)
        ranked = list(getattr(self, "res_fit_quality_ranked", []))
        if ranked:
            ranks = np.asarray([float(entry["rank"]) for entry in ranked], dtype=float)
            nrmse = np.asarray([float(entry["nrmse"]) for entry in ranked], dtype=float)
            positive = np.isfinite(nrmse) & (nrmse > 0.0)
            if np.any(positive):
                ax_rank.plot(ranks[positive], nrmse[positive], color="tab:blue", linewidth=1.1)
                ax_rank.set_yscale("log")
            if item is not None:
                item_nrmse = float(item["nrmse"])
                if np.isfinite(item_nrmse) and item_nrmse > 0.0:
                    ax_rank.plot([float(item["rank"])], [item_nrmse], marker="o", color="crimson", markersize=7)
            ax_rank.set_xlabel("Fit rank, best to worst")
            ax_rank.set_ylabel("nrmse")
            ax_rank.grid(True, alpha=0.3)
        if item is None:
            ax_amp.text(0.5, 0.5, "No fit selected.", ha="center", va="center")
            self.res_fit_quality_canvas.draw_idle()
            return
        self._res_fit_quality_draw_fit_axes(ax_amp, ax_iq, item)
        self.res_fit_quality_figure.tight_layout()
        self.res_fit_quality_canvas.draw_idle()

    def _res_fit_quality_draw_fit_axes(
        self,
        ax_amp,
        ax_iq,
        item: dict,
        *,
        raw_color: str = "tab:blue",
        fit_color: str = "darkorange",
        marker_color: str = "0.35",
        fr0_color: str = "purple",
        true_color: str = "crimson",
        alpha: float = 1.0,
        label_prefix: str = "",
        zorder: int = 2,
        set_title: bool = True,
    ) -> None:
        scan = item["scan"]
        payload = item["payload"]
        freq = np.asarray(scan.freq, dtype=float)
        z = np.asarray(scan.s21_complex_raw, dtype=np.complex128)
        order = np.argsort(freq)
        freq = freq[order]
        z = z[order]
        marker_hz = float(item.get("marker_hz", np.nan))
        lo, hi = payload.get("selection_range_hz", (np.nan, np.nan))
        lo = float(lo)
        hi = float(hi)
        mask = np.isfinite(freq) & (freq >= lo) & (freq <= hi)
        if not np.any(mask):
            mask = np.ones_like(freq, dtype=bool)
        ax_amp.plot(
            freq[mask] / _HZ_PER_GHZ,
            np.abs(z[mask]),
            color=raw_color,
            linewidth=1.1,
            alpha=alpha,
            zorder=zorder,
            label=f"{label_prefix}raw |S21|",
        )
        ax_iq.plot(
            np.real(z[mask]),
            np.imag(z[mask]),
            color=raw_color,
            linewidth=1.0,
            alpha=alpha,
            zorder=zorder,
            label=f"{label_prefix}raw S21",
        )
        fit_freq = np.asarray(payload.get("fit_freq_hz", []), dtype=float)
        fit_z = np.asarray(payload.get("fit_s21_complex", []), dtype=np.complex128)
        if fit_freq.size and fit_z.size == fit_freq.size:
            ax_amp.plot(
                fit_freq / _HZ_PER_GHZ,
                np.abs(fit_z),
                color=fit_color,
                linestyle="--",
                linewidth=1.4,
                alpha=alpha,
                zorder=zorder,
                label=f"{label_prefix}fit",
            )
            fr0_hz = float(payload.get("fr0_hz", np.nan))
            true_fr_hz = float(payload.get("true_fr_hz", np.nan))
            if np.isfinite(marker_hz):
                marker_amp = np.interp(marker_hz, fit_freq, np.abs(fit_z))
                ax_amp.plot(
                    [marker_hz / _HZ_PER_GHZ],
                    [marker_amp],
                    marker="o",
                    markersize=6,
                    markerfacecolor="none",
                    markeredgecolor=marker_color,
                    markeredgewidth=1.4,
                    alpha=alpha,
                    linestyle="none",
                    zorder=zorder + 1,
                    label=f"{label_prefix}marker",
                )
            if np.isfinite(fr0_hz):
                fr0_amp = np.interp(fr0_hz, fit_freq, np.abs(fit_z))
                ax_amp.plot(
                    [fr0_hz / _HZ_PER_GHZ],
                    [fr0_amp],
                    marker="x",
                    markersize=7,
                    markeredgewidth=1.8,
                    color=fr0_color,
                    alpha=alpha,
                    linestyle="none",
                    zorder=zorder,
                    label=f"{label_prefix}fr0",
                )
            if np.isfinite(true_fr_hz):
                true_amp = np.interp(true_fr_hz, fit_freq, np.abs(fit_z))
                ax_amp.plot(
                    [true_fr_hz / _HZ_PER_GHZ],
                    [true_amp],
                    marker="+",
                    markersize=12,
                    markeredgewidth=2.0,
                    color=true_color,
                    alpha=alpha,
                    linestyle="none",
                    zorder=zorder,
                    label=f"{label_prefix}powered fr",
                )
            ax_iq.plot(
                np.real(fit_z),
                np.imag(fit_z),
                color=fit_color,
                linestyle="--",
                alpha=alpha,
                zorder=zorder,
                label=f"{label_prefix}fit",
            )
            if np.isfinite(fr0_hz):
                fr0_iq = np.interp(fr0_hz, fit_freq, np.real(fit_z)) + 1j * np.interp(fr0_hz, fit_freq, np.imag(fit_z))
                ax_iq.plot(
                    [fr0_iq.real],
                    [fr0_iq.imag],
                    marker="x",
                    color=fr0_color,
                    alpha=alpha,
                    linestyle="none",
                    zorder=zorder,
                    label=f"{label_prefix}fr0",
                )
            if np.isfinite(true_fr_hz):
                true_iq = np.interp(true_fr_hz, fit_freq, np.real(fit_z)) + 1j * np.interp(
                    true_fr_hz, fit_freq, np.imag(fit_z)
                )
                ax_iq.plot(
                    [true_iq.real],
                    [true_iq.imag],
                    marker="+",
                    markersize=12,
                    markeredgewidth=2.0,
                    color=true_color,
                    alpha=alpha,
                    linestyle="none",
                    zorder=zorder,
                    label=f"{label_prefix}powered fr",
                )
        title = item.get("title")
        if title is None:
            title = f"Rank {item['rank']} | {Path(scan.filename).name} | Resonator {item['resonator_number']}"
        if set_title:
            ax_amp.set_title(title, fontsize=9)
        ax_amp.set_xlabel("Frequency (GHz)")
        ax_amp.set_ylabel("|S21|")
        ax_amp.grid(True, alpha=0.3)
        ax_amp.legend(loc="best", fontsize=8)
        ax_iq.set_xlabel("Re(raw S21)")
        ax_iq.set_ylabel("Im(raw S21)")
        ax_iq.grid(True, alpha=0.3)
        ax_iq.set_aspect("equal", adjustable="box")
        ax_iq.legend(loc="best", fontsize=8)

    def _res_fit_quality_accept_threshold(self) -> None:
        item = self._res_fit_quality_current_item()
        if item is None:
            return
        threshold_rank = int(item["rank"])
        threshold_nrmse = float(item["nrmse"])
        if not messagebox.askyesno(
            "Attach acceptance flags",
            f"Accept fits ranked {threshold_rank} or better?\n\n"
            f"Threshold nrmse: {threshold_nrmse:.9g}\n"
            "Fits worse than this will be marked rejected.",
            parent=self.res_fit_quality_window,
        ):
            return
        self._res_fit_quality_attach_acceptance(threshold_rank=threshold_rank)

    def _res_fit_quality_accept_all(self) -> None:
        ranked = list(getattr(self, "res_fit_quality_ranked", []))
        if not ranked:
            return
        if not messagebox.askyesno(
            "Accept all fits",
            f"Accept all {len(ranked)} ranked fit result(s)?",
            parent=self.res_fit_quality_window,
        ):
            return
        self._res_fit_quality_attach_acceptance(threshold_rank=len(ranked))

    def _res_fit_quality_attach_acceptance(self, *, threshold_rank: int) -> None:
        ranked = list(getattr(self, "res_fit_quality_ranked", []))
        if not ranked:
            return
        accepted_count = 0
        rejected_count = 0
        scan_keys: set[str] = set()
        threshold_nrmse = float(ranked[threshold_rank - 1]["nrmse"]) if 0 < threshold_rank <= len(ranked) else np.inf
        timestamp_details = {
            "threshold_rank": int(threshold_rank),
            "threshold_nrmse": float(threshold_nrmse),
            "ranked_fit_count": int(len(ranked)),
        }
        for item in ranked:
            accepted = int(item["rank"]) <= int(threshold_rank)
            payload = item["payload"]
            payload["accepted"] = bool(accepted)
            payload["acceptance"] = {
                **timestamp_details,
                "accepted": bool(accepted),
                "rank": int(item["rank"]),
            }
            scan_keys.add(str(item["scan_key"]))
            if accepted:
                accepted_count += 1
            else:
                rejected_count += 1
        for scan in self._selected_scans():
            scan_key = self._scan_key(scan)
            if scan_key not in scan_keys:
                continue
            fit_payload = scan.candidate_resonators.get("logan_nonlinear_iq_marker_fits")
            if isinstance(fit_payload, dict):
                fit_payload["acceptance_threshold_rank"] = int(threshold_rank)
                fit_payload["acceptance_threshold_nrmse"] = float(threshold_nrmse)
                fit_payload["accepted_count"] = int(accepted_count)
                fit_payload["rejected_count"] = int(rejected_count)
            scan.processing_history.append(
                _make_event(
                    "attach_logan_marker_fit_acceptance",
                    {
                        **timestamp_details,
                        "accepted_count": int(accepted_count),
                        "rejected_count": int(rejected_count),
                    },
                )
            )
        self.dataset.processing_history.append(
            _make_event(
                "attach_logan_marker_fit_acceptance",
                {
                    **timestamp_details,
                    "accepted_count": int(accepted_count),
                    "rejected_count": int(rejected_count),
                    "scan_count": int(len(scan_keys)),
                },
            )
        )
        self._mark_dirty()
        self._refresh_status()
        self._autosave_dataset()
        self._res_fit_quality_update_output_fields(self._res_fit_quality_current_item())
        self._res_fit_quality_render_selected()
        self.res_fit_quality_status_var.set(
            f"Attached acceptance flags: {accepted_count} accepted, {rejected_count} rejected."
        )
        self._log(f"Attached marker fit acceptance flags: {accepted_count} accepted, {rejected_count} rejected.")

    def open_batch_resonance_fit_window(self) -> None:
        scans = self._selected_scans()
        if not scans:
            messagebox.showwarning(
                "No selection",
                "No scans selected for analysis.\nUse 'Select Scans for Analysis' first.",
            )
            return
        if fit_nonlinear_iq is None or nonlinear_iq is None:
            messagebox.showerror("citkid unavailable", "Could not import citkid.res fitter/model functions.")
            return

        rows = self._batch_res_fit_collect_rows(scans)
        marker_count = sum(len(row["markers"]) for row in rows)
        if marker_count <= 0:
            messagebox.showwarning(
                "No resonator markers",
                "No attached resonator markers were found on the selected scans.",
            )
            return

        if getattr(self, "batch_res_fit_window", None) is not None and self.batch_res_fit_window.winfo_exists():
            self.batch_res_fit_window.lift()
            return

        self.batch_res_fit_rows = rows
        self.batch_res_fit_results = {}
        self.batch_res_fit_scan_slider = None
        self.batch_res_fit_marker_slider = None
        self.batch_res_fit_window = tk.Toplevel(self.root)
        self.batch_res_fit_window.title("Fit Marked Resonators")
        self.batch_res_fit_window.geometry("1320x880")
        self.batch_res_fit_window.protocol("WM_DELETE_WINDOW", self._batch_res_fit_close)

        controls = tk.Frame(self.batch_res_fit_window, padx=8, pady=6)
        controls.pack(side="top", fill="x")
        top_row = tk.Frame(controls)
        top_row.pack(side="top", fill="x")
        self.batch_res_fit_status_var = tk.StringVar(
            value=f"Ready: {len(rows)} selected scan(s), {marker_count} resonator marker(s)."
        )
        tk.Label(top_row, textvariable=self.batch_res_fit_status_var, anchor="w").pack(
            side="left", fill="x", expand=True
        )
        tk.Button(top_row, text="Fit", width=10, command=self._batch_res_fit_run).pack(side="right", padx=(8, 0))
        tk.Button(top_row, text="Attach Results", width=14, command=self._batch_res_fit_confirm_attach).pack(
            side="right", padx=(8, 0)
        )

        fit_row = tk.Frame(controls)
        fit_row.pack(side="top", fill="x", pady=(6, 0))
        tk.Label(fit_row, text="Fit window df/f").pack(side="left", padx=(0, 6))
        self.batch_res_fit_window_var = tk.DoubleVar(value=1.0e-3)
        self.batch_res_fit_force_amp_only_var = tk.BooleanVar(value=False)
        tk.Scale(
            fit_row,
            from_=0.0,
            to=2.0e-3,
            resolution=1.0e-5,
            orient="horizontal",
            length=260,
            showvalue=True,
            variable=self.batch_res_fit_window_var,
        ).pack(side="left")
        tk.Checkbutton(
            fit_row,
            text="Amplitude-only",
            variable=self.batch_res_fit_force_amp_only_var,
        ).pack(side="left", padx=(10, 8))
        tk.Label(fit_row, text=f"Total markers: {marker_count}").pack(side="left", padx=(14, 8))
        tk.Label(fit_row, text="Scans").pack(side="left", padx=(14, 4))
        self.batch_res_fit_scan_progress = ttk.Progressbar(fit_row, orient="horizontal", length=180, mode="determinate")
        self.batch_res_fit_scan_progress.pack(side="left", padx=(0, 8))
        tk.Label(fit_row, text="Markers in current scan").pack(side="left", padx=(8, 4))
        self.batch_res_fit_marker_progress = ttk.Progressbar(
            fit_row, orient="horizontal", length=180, mode="determinate"
        )
        self.batch_res_fit_marker_progress.pack(side="left")

        nav_row = tk.Frame(controls)
        nav_row.pack(side="top", fill="x", pady=(6, 0))
        tk.Label(nav_row, text="VNA scan").pack(side="left", padx=(0, 6))
        self.batch_res_fit_scan_index_var = tk.IntVar(value=0)
        self.batch_res_fit_scan_slider = tk.Scale(
            nav_row,
            from_=0,
            to=max(0, len(rows) - 1),
            orient="horizontal",
            length=260,
            showvalue=True,
            variable=self.batch_res_fit_scan_index_var,
            command=lambda _v: self._batch_res_fit_on_scan_slider_changed(),
        )
        self.batch_res_fit_scan_slider.pack(side="left")
        tk.Label(nav_row, text="Resonator marker").pack(side="left", padx=(16, 6))
        self.batch_res_fit_marker_index_var = tk.IntVar(value=0)
        self.batch_res_fit_marker_slider = tk.Scale(
            nav_row,
            from_=0,
            to=max(0, len(rows[0]["markers"]) - 1),
            orient="horizontal",
            length=260,
            showvalue=True,
            variable=self.batch_res_fit_marker_index_var,
            command=lambda _v: self._batch_res_fit_render_selected(),
        )
        self.batch_res_fit_marker_slider.pack(side="left")
        self.batch_res_fit_scan_slider.bind("<ButtonRelease-1>", lambda _e: self._batch_res_fit_render_selected())
        self.batch_res_fit_marker_slider.bind("<ButtonRelease-1>", lambda _e: self._batch_res_fit_render_selected())

        output_row = tk.Frame(controls)
        output_row.pack(side="top", fill="x", pady=(6, 0))
        self.batch_res_fit_output_vars = {}
        for label, key, width in (
            ("Res #", "resonator_number", 7),
            ("Marker GHz", "marker_ghz", 11),
            ("fr0 GHz", "fr0_ghz", 11),
            ("powered fr GHz", "true_fr_ghz", 12),
            ("delta fr Hz", "delta_fr_hz", 12),
            ("Qr", "qr", 10),
            ("Qi", "qi", 10),
            ("Qc", "qc", 10),
            ("a", "a_nl", 9),
            ("nrmse", "nrmse", 10),
            ("Fit mode", "fit_data_mode", 14),
            ("Status", "status", 18),
        ):
            tk.Label(output_row, text=label).pack(side="left", padx=(0, 2))
            var = tk.StringVar()
            self.batch_res_fit_output_vars[key] = var
            tk.Entry(output_row, textvariable=var, width=width, state="readonly").pack(side="left", padx=(0, 6))

        self.batch_res_fit_figure = Figure(figsize=(12, 7))
        self.batch_res_fit_canvas = FigureCanvasTkAgg(self.batch_res_fit_figure, master=self.batch_res_fit_window)
        self.batch_res_fit_toolbar = NavigationToolbar2Tk(self.batch_res_fit_canvas, self.batch_res_fit_window)
        self.batch_res_fit_toolbar.update()
        self.batch_res_fit_toolbar.pack(side="top", fill="x")
        self.batch_res_fit_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._batch_res_fit_render_selected()

    def _batch_res_fit_close(self) -> None:
        if getattr(self, "batch_res_fit_window", None) is not None and self.batch_res_fit_window.winfo_exists():
            self.batch_res_fit_window.destroy()
        self.batch_res_fit_window = None
        self.batch_res_fit_canvas = None
        self.batch_res_fit_toolbar = None
        self.batch_res_fit_figure = None
        self.batch_res_fit_status_var = None
        self.batch_res_fit_window_var = None
        self.batch_res_fit_force_amp_only_var = None
        self.batch_res_fit_scan_progress = None
        self.batch_res_fit_marker_progress = None
        self.batch_res_fit_scan_slider = None
        self.batch_res_fit_marker_slider = None
        self.batch_res_fit_output_vars = {}

    def _batch_res_fit_collect_rows(self, scans: list[object]) -> list[dict]:
        rows = []
        for scan in scans:
            payload = scan.candidate_resonators.get("sheet_resonances")
            assignments = payload.get("assignments") if isinstance(payload, dict) else {}
            if not isinstance(assignments, dict):
                assignments = {}
            markers = []
            for resonator_number, record in assignments.items():
                if not isinstance(record, dict):
                    continue
                try:
                    target_hz = float(record.get("frequency_hz"))
                except Exception:
                    continue
                if np.isfinite(target_hz):
                    markers.append(
                        {
                            "resonator_number": str(resonator_number).strip(),
                            "marker_hz": target_hz,
                            "marker_record": dict(record),
                        }
                    )
            if markers:
                rows.append(
                    {
                        "scan": scan,
                        "scan_key": self._scan_key(scan),
                        "markers": sorted(markers, key=lambda item: item["marker_hz"]),
                    }
                )
        return rows

    def _batch_res_fit_current_selection(self) -> tuple[Optional[dict], Optional[dict]]:
        rows = list(getattr(self, "batch_res_fit_rows", []))
        if not rows:
            return None, None
        scan_idx = int(getattr(self, "batch_res_fit_scan_index_var", tk.IntVar(value=0)).get())
        scan_idx = min(max(scan_idx, 0), len(rows) - 1)
        row = rows[scan_idx]
        markers = row["markers"]
        if not markers:
            return row, None
        marker_idx = int(getattr(self, "batch_res_fit_marker_index_var", tk.IntVar(value=0)).get())
        marker_idx = min(max(marker_idx, 0), len(markers) - 1)
        return row, markers[marker_idx]

    def _batch_res_fit_on_scan_slider_changed(self) -> None:
        row, _marker = self._batch_res_fit_current_selection()
        if row is None or self.batch_res_fit_marker_slider is None:
            return
        self.batch_res_fit_marker_slider.configure(to=max(0, len(row["markers"]) - 1))
        self.batch_res_fit_marker_index_var.set(0)
        self._batch_res_fit_render_selected()

    def _batch_res_fit_result_key(self, row: dict, marker: dict) -> tuple[str, str]:
        return str(row["scan_key"]), str(marker["resonator_number"])

    @staticmethod
    def _batch_res_fit_has_no_phase(z: np.ndarray) -> bool:
        z = np.asarray(z, dtype=np.complex128)
        if z.size == 0:
            return False
        finite = np.isfinite(np.real(z)) & np.isfinite(np.imag(z))
        if not np.any(finite):
            return False
        return bool(np.nanmax(np.abs(np.imag(z[finite]))) <= 1.0e-12)

    @staticmethod
    def _batch_res_fit_bounds_include_p0(p0: np.ndarray, bounds: tuple[list[float], list[float]]) -> tuple[list[float], list[float]]:
        lower = list(bounds[0])
        upper = list(bounds[1])
        for index, value in enumerate(np.asarray(p0, dtype=float)):
            if not np.isfinite(value):
                continue
            if lower[index] > upper[index]:
                lower[index], upper[index] = upper[index], lower[index]
            if value < lower[index]:
                lower[index] = value * 0.9 if value > 0 else value * 1.1 if value < 0 else -1.0
            if value > upper[index]:
                upper[index] = value * 1.1 if value > 0 else value * 0.9 if value < 0 else 1.0
            if lower[index] == upper[index]:
                delta = max(abs(value) * 0.1, 1.0e-12)
                lower[index] -= delta
                upper[index] += delta
        return lower, upper

    def _batch_res_fit_amplitude_only(
        self,
        f_fit: np.ndarray,
        z_fit: np.ndarray,
        p0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
        if nonlinear_iq is None:
            raise RuntimeError("citkid nonlinear_iq model is unavailable.")
        amp_fit = np.abs(np.asarray(z_fit, dtype=np.complex128))
        p0 = np.asarray(p0, dtype=float)
        # Same default parameter order and limits as citkid.res.fitter.fit_nonlinear_iq.
        bounds = (
            [float(np.min(f_fit)), 1.0e3, 0.01, -np.pi / 2.0, 0.0, -1.0e2, -1.0e2, -1.0e-6],
            [float(np.max(f_fit)), 1.0e7, 1.0 - 1.0e-6, np.pi / 2.0, 1.0, 1.0e2, 1.0e2, 1.0e-6],
        )
        for index in (1, 5, 6):
            if p0[index] != 0.0 and np.isfinite(p0[index]):
                bounds[0][index] = float(p0[index] / 10.0)
                bounds[1][index] = float(p0[index] * 10.0)
        lower, upper = self._batch_res_fit_bounds_include_p0(p0, bounds)

        def amp_model(freq: np.ndarray, fr: float, qr: float, amp: float, phi: float, a_nl: float, i0: float, q0: float, tau: float) -> np.ndarray:
            return np.abs(nonlinear_iq(freq, fr, qr, amp, phi, a_nl, i0, q0, tau, True))

        popt, pcov = optimize.curve_fit(
            amp_model,
            np.asarray(f_fit, dtype=float),
            amp_fit,
            p0=p0,
            bounds=(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
            maxfev=20000,
        )
        perr = np.sqrt(np.diag(pcov)) if pcov.size else np.full_like(popt, np.nan, dtype=float)
        model = nonlinear_iq(f_fit, *popt, True)
        resid = amp_fit - np.abs(model)
        norm = float(np.vdot(amp_fit, amp_fit).real)
        nrmse = float(np.vdot(resid, resid).real / norm) if norm > 0.0 else np.inf
        return np.asarray(p0, dtype=float), np.asarray(popt, dtype=float), np.asarray(perr, dtype=float), nrmse, model

    def _batch_res_fit_fit_one(self, row: dict, marker: dict, df_over_f: float) -> dict:
        scan = row["scan"]
        freq = np.asarray(scan.freq, dtype=float)
        z = np.asarray(scan.s21_complex_raw, dtype=np.complex128)
        order = np.argsort(freq)
        freq = freq[order]
        z = z[order]
        center = float(marker["marker_hz"])
        half_width = 0.5 * max(float(df_over_f), 0.0) * center
        if half_width <= 0.0:
            raise ValueError("Fit window df/f must be greater than zero.")
        lo = center - half_width
        hi = center + half_width
        mask = np.isfinite(freq) & np.isfinite(np.real(z)) & np.isfinite(np.imag(z)) & (freq >= lo) & (freq <= hi)
        if np.count_nonzero(mask) < 8:
            raise ValueError(f"Only {np.count_nonzero(mask)} point(s) in fit window.")
        f_fit = freq[mask]
        z_fit = z[mask]
        try:
            p0 = np.asarray(guess_p0_nonlinear_iq(f_fit, z_fit), dtype=float)
        except Exception:
            z0 = complex(np.mean(z_fit)) if z_fit.size else 1.0 + 0j
            p0 = np.asarray([center, 1.0e4, 0.5, 0.0, 0.0, z0.real, z0.imag, 0.0], dtype=float)
        p0[0] = center
        force_amp_only = (
            bool(self.batch_res_fit_force_amp_only_var.get())
            if getattr(self, "batch_res_fit_force_amp_only_var", None) is not None
            else False
        )
        if force_amp_only or self._batch_res_fit_has_no_phase(z_fit):
            p0_out, popt, perr, nrmse, model = self._batch_res_fit_amplitude_only(f_fit, z_fit, p0)
            fit_data_mode = "amplitude_only_forced" if force_amp_only else "amplitude_only"
            message = (
                "Batch forced amplitude-only fit complete."
                if force_amp_only
                else "Batch amplitude-only fit complete."
            )
        else:
            p0_out, popt, perr, nrmse, _figax = fit_nonlinear_iq(
                f_fit,
                z_fit,
                p0=np.asarray(p0, dtype=float),
                fit_tau=True,
                downward=True,
                plotq=False,
            )
            popt = np.asarray(popt, dtype=float)
            model = nonlinear_iq(f_fit, *popt, True)
            fit_data_mode = "complex_iq"
            message = "Batch nonlinear IQ fit complete."
        payload = self._res_make_logan_fit_payload(
            scan,
            lo=float(lo),
            hi=float(hi),
            p0=np.asarray(p0_out, dtype=float),
            popt=popt,
            perr=np.asarray(perr, dtype=float),
            nrmse=float(nrmse),
            model=model,
            f_fit=f_fit,
            success=True,
            message=message,
        )
        payload.update(
            {
                "resonator_number": str(marker["resonator_number"]),
                "marker_frequency_hz": float(center),
                "fit_window_df_over_f": float(df_over_f),
                "fit_data_mode": fit_data_mode,
                "source_marker_record": dict(marker.get("marker_record", {})),
            }
        )
        return payload

    def _batch_res_fit_run(self) -> None:
        rows = list(getattr(self, "batch_res_fit_rows", []))
        if not rows:
            return
        df_over_f = float(self.batch_res_fit_window_var.get()) if self.batch_res_fit_window_var is not None else 1.0e-3
        total_markers = sum(len(row["markers"]) for row in rows)
        self.batch_res_fit_results = {}
        self.batch_res_fit_scan_progress.configure(maximum=max(1, len(rows)), value=0)
        self.batch_res_fit_marker_progress.configure(maximum=1, value=0)
        last_render_time = 0.0
        finished = 0
        failures = 0
        for scan_idx, row in enumerate(rows):
            markers = row["markers"]
            self.batch_res_fit_scan_index_var.set(scan_idx)
            self.batch_res_fit_marker_progress.configure(maximum=max(1, len(markers)), value=0)
            for marker_idx, marker in enumerate(markers):
                self.batch_res_fit_marker_index_var.set(marker_idx)
                self.batch_res_fit_status_var.set(
                    f"Fitting scan {scan_idx + 1}/{len(rows)}, marker {marker_idx + 1}/{len(markers)} "
                    f"({finished}/{total_markers} complete)."
                )
                self.batch_res_fit_window.update_idletasks()
                self.batch_res_fit_window.update()
                try:
                    payload = self._batch_res_fit_fit_one(row, marker, df_over_f)
                except Exception as exc:
                    payload = {
                        "success": False,
                        "message": str(exc),
                        "resonator_number": str(marker["resonator_number"]),
                        "marker_frequency_hz": float(marker["marker_hz"]),
                        "fit_window_df_over_f": float(df_over_f),
                    }
                    failures += 1
                self.batch_res_fit_results[self._batch_res_fit_result_key(row, marker)] = payload
                finished += 1
                self.batch_res_fit_marker_progress.configure(value=marker_idx + 1)
                now = time.monotonic()
                if finished == 1 or now - last_render_time >= 10.0:
                    self._batch_res_fit_render_selected()
                    last_render_time = now
            self.batch_res_fit_scan_progress.configure(value=scan_idx + 1)
        self._batch_res_fit_render_selected()
        self.batch_res_fit_status_var.set(
            f"Fit complete: {finished - failures}/{finished} succeeded, {failures} failed. Review results, then Attach Results."
        )

    def _batch_res_fit_selected_payload(self) -> tuple[Optional[dict], Optional[dict], Optional[dict]]:
        row, marker = self._batch_res_fit_current_selection()
        if row is None or marker is None:
            return row, marker, None
        payload = getattr(self, "batch_res_fit_results", {}).get(self._batch_res_fit_result_key(row, marker))
        return row, marker, payload

    def _batch_res_fit_update_output_fields(self, row: Optional[dict], marker: Optional[dict], payload: Optional[dict]) -> None:
        vars_map = getattr(self, "batch_res_fit_output_vars", {})
        for var in vars_map.values():
            var.set("")
        if marker is None:
            return
        vars_map["resonator_number"].set(str(marker.get("resonator_number", "")))
        marker_hz = float(marker.get("marker_hz", np.nan))
        vars_map["marker_ghz"].set(f"{marker_hz / _HZ_PER_GHZ:.9g}" if np.isfinite(marker_hz) else "")
        if not isinstance(payload, dict):
            vars_map["status"].set("not fit")
            return
        vars_map["status"].set("ok" if payload.get("success") else str(payload.get("message", "failed"))[:18])
        if payload.get("success"):
            vars_map["fr0_ghz"].set(f"{float(payload.get('fr0_hz', np.nan)) / _HZ_PER_GHZ:.9g}")
            vars_map["true_fr_ghz"].set(f"{float(payload.get('true_fr_hz', np.nan)) / _HZ_PER_GHZ:.9g}")
            vars_map["delta_fr_hz"].set(f"{float(payload.get('delta_fr_hz', np.nan)):.9g}")
            vars_map["qr"].set(f"{float(payload.get('q_loaded', np.nan)):.9g}")
            vars_map["qi"].set(f"{float(payload.get('q_internal', np.nan)):.9g}")
            vars_map["qc"].set(f"{float(payload.get('q_coupling', np.nan)):.9g}")
            vars_map["a_nl"].set(f"{float(payload.get('a_nl', np.nan)):.9g}")
            vars_map["nrmse"].set(f"{float(payload.get('nrmse', np.nan)):.9g}")
            vars_map["fit_data_mode"].set(str(payload.get("fit_data_mode", ""))[:14])

    def _batch_res_fit_render_selected(self) -> None:
        if getattr(self, "batch_res_fit_figure", None) is None or self.batch_res_fit_canvas is None:
            return
        row, marker, payload = self._batch_res_fit_selected_payload()
        self._batch_res_fit_update_output_fields(row, marker, payload)
        self.batch_res_fit_figure.clear()
        ax_amp = self.batch_res_fit_figure.add_subplot(1, 2, 1)
        ax_iq = self.batch_res_fit_figure.add_subplot(1, 2, 2)
        if row is None or marker is None:
            ax_amp.text(0.5, 0.5, "No marker selected.", ha="center", va="center")
            ax_iq.axis("off")
            self.batch_res_fit_canvas.draw_idle()
            return
        scan = row["scan"]
        freq = np.asarray(scan.freq, dtype=float)
        z = np.asarray(scan.s21_complex_raw, dtype=np.complex128)
        order = np.argsort(freq)
        freq = freq[order]
        z = z[order]
        marker_hz = float(marker["marker_hz"])
        df_over_f = (
            float(payload.get("fit_window_df_over_f"))
            if isinstance(payload, dict) and payload.get("fit_window_df_over_f") is not None
            else float(self.batch_res_fit_window_var.get())
        )
        half_width = 0.5 * max(df_over_f, 1.0e-12) * marker_hz
        lo = marker_hz - half_width
        hi = marker_hz + half_width
        mask = (freq >= lo) & (freq <= hi)
        if not np.any(mask):
            mask = np.ones_like(freq, dtype=bool)
        ax_amp.plot(freq[mask] / _HZ_PER_GHZ, np.abs(z[mask]), color="tab:blue", linewidth=1.1, label="raw |S21|")
        ax_amp.axvline(marker_hz / _HZ_PER_GHZ, color="0.35", linestyle=":", linewidth=1.2, label="marker")
        ax_iq.plot(np.real(z[mask]), np.imag(z[mask]), color="tab:blue", linewidth=1.0, label="raw S21")
        if isinstance(payload, dict) and payload.get("success"):
            fit_freq = np.asarray(payload.get("fit_freq_hz", []), dtype=float)
            fit_z = np.asarray(payload.get("fit_s21_complex", []), dtype=np.complex128)
            if fit_freq.size and fit_z.size == fit_freq.size:
                ax_amp.plot(
                    fit_freq / _HZ_PER_GHZ,
                    np.abs(fit_z),
                    color="darkorange",
                    linestyle="--",
                    linewidth=1.4,
                    label="fit",
                )
                fr0_hz = float(payload.get("fr0_hz", np.nan))
                true_fr_hz = float(payload.get("true_fr_hz", np.nan))
                if np.isfinite(fr0_hz):
                    ax_amp.axvline(fr0_hz / _HZ_PER_GHZ, color="purple", linestyle=":", linewidth=1.4, label="fr0")
                if np.isfinite(true_fr_hz):
                    ax_amp.axvline(
                        true_fr_hz / _HZ_PER_GHZ,
                        color="crimson",
                        linestyle="-.",
                        linewidth=1.4,
                        label="powered fr",
                    )
                ax_iq.plot(np.real(fit_z), np.imag(fit_z), color="darkorange", linestyle="--", label="fit")
                if np.isfinite(fr0_hz):
                    fr0_iq = np.interp(fr0_hz, fit_freq, np.real(fit_z)) + 1j * np.interp(
                        fr0_hz, fit_freq, np.imag(fit_z)
                    )
                    ax_iq.plot([fr0_iq.real], [fr0_iq.imag], marker="x", color="purple", linestyle="none", label="fr0")
                if np.isfinite(true_fr_hz):
                    true_iq = np.interp(true_fr_hz, fit_freq, np.real(fit_z)) + 1j * np.interp(
                        true_fr_hz, fit_freq, np.imag(fit_z)
                    )
                    ax_iq.plot(
                        [true_iq.real],
                        [true_iq.imag],
                        marker="+",
                        markersize=12,
                        markeredgewidth=2.0,
                        color="crimson",
                        linestyle="none",
                        label="powered fr",
                    )
        title = f"{Path(scan.filename).name} | Resonator {marker['resonator_number']}"
        ax_amp.set_title(title, fontsize=10)
        ax_amp.set_xlabel("Frequency (GHz)")
        ax_amp.set_ylabel("|S21|")
        ax_amp.grid(True, alpha=0.3)
        ax_amp.legend(loc="best", fontsize=8)
        ax_iq.set_xlabel("Re(raw S21)")
        ax_iq.set_ylabel("Im(raw S21)")
        ax_iq.grid(True, alpha=0.3)
        ax_iq.set_aspect("equal", adjustable="box")
        ax_iq.legend(loc="best", fontsize=8)
        self.batch_res_fit_figure.tight_layout()
        self.batch_res_fit_canvas.draw_idle()

    def _batch_res_fit_confirm_attach(self) -> None:
        results = getattr(self, "batch_res_fit_results", {})
        if not results:
            messagebox.showwarning("No fit results", "Run Fit before attaching results.", parent=self.batch_res_fit_window)
            return
        success_count = sum(1 for payload in results.values() if isinstance(payload, dict) and payload.get("success"))
        if success_count <= 0:
            messagebox.showwarning("No successful fits", "No successful fits are available to attach.", parent=self.batch_res_fit_window)
            return
        if not messagebox.askyesno(
            "Attach batch fits",
            f"Attach {success_count} successful batch fit result(s)?\n\nPreviously attached batch fit parameters on these scans will be overwritten.",
            parent=self.batch_res_fit_window,
        ):
            return
        by_scan: dict[str, dict] = {}
        for (scan_key, resonator_number), payload in results.items():
            if isinstance(payload, dict) and payload.get("success"):
                by_scan.setdefault(str(scan_key), {})[str(resonator_number)] = payload
        attached_scans = 0
        for row in getattr(self, "batch_res_fit_rows", []):
            scan_key = str(row["scan_key"])
            scan = row["scan"]
            assignments = by_scan.get(scan_key, {})
            if not assignments:
                continue
            scan.candidate_resonators["logan_nonlinear_iq_marker_fits"] = {
                "fit_window_df_over_f": float(self.batch_res_fit_window_var.get()),
                "created_from": "sheet_resonances",
                "assignments": assignments,
            }
            scan.processing_history.append(
                _make_event(
                    "attach_logan_marker_batch_fits",
                    {
                        "fit_count": int(len(assignments)),
                        "fit_window_df_over_f": float(self.batch_res_fit_window_var.get()),
                    },
                )
            )
            attached_scans += 1
        self.dataset.processing_history.append(
            _make_event(
                "attach_logan_marker_batch_fits",
                {"scan_count": int(attached_scans), "fit_count": int(success_count)},
            )
        )
        self._mark_dirty()
        self._refresh_status()
        self._autosave_dataset()
        self._log(f"Attached {success_count} marker fit result(s) across {attached_scans} scan(s).")
        self.batch_res_fit_status_var.set(f"Attached {success_count} fit result(s) across {attached_scans} scan(s).")
