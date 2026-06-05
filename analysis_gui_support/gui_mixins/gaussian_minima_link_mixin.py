from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter import messagebox

from analysis_gui_support.analysis_models import _current_user, _make_event, _read_polar_series


MAX_LINK_REL_FREQ_CHANGE = 0.01


class GaussianMinimaLinkMixin:
    @staticmethod
    def _gaussian_minima_indices_for_scan(scan) -> np.ndarray:
        gaussian = scan.candidate_resonators.get("gaussian_convolution", {})
        indices = gaussian.get("candidate_indices") if isinstance(gaussian, dict) else None
        if indices is None:
            norm = scan.baseline_filter.get("normalized", {})
            conv = norm.get("gaussian_conv") if isinstance(norm, dict) else None
            indices = conv.get("minima_indices") if isinstance(conv, dict) else None
        idx = np.asarray(indices if indices is not None else np.array([]), dtype=int).ravel()
        idx = idx[idx >= 0]
        if idx.size == 0:
            return idx
        return np.unique(idx)

    def _next_unused_resonator_numbers(self, count: int) -> list[str]:
        used: set[str] = set()
        for scan in self.dataset.vna_scans:
            payload = scan.candidate_resonators.get("sheet_resonances")
            assignments = payload.get("assignments") if isinstance(payload, dict) else None
            if isinstance(assignments, dict):
                used.update(str(key).strip() for key in assignments if str(key).strip())

        numbers: list[str] = []
        candidate = 1
        while len(numbers) < int(count):
            label = str(candidate)
            if label not in used:
                numbers.append(label)
                used.add(label)
            candidate += 1
        return numbers

    @staticmethod
    def _match_previous_to_current(
        previous: list[dict],
        current: list[dict],
        max_rel_change: float = MAX_LINK_REL_FREQ_CHANGE,
    ) -> tuple[list[tuple[int, int]], list[dict]]:
        if not previous or not current:
            return [], []
        pairs: list[tuple[float, int, int]] = []
        for prev_idx, prev_item in enumerate(previous):
            prev_freq = float(prev_item["freq_hz"])
            for cur_idx, cur_item in enumerate(current):
                pairs.append((abs(prev_freq - float(cur_item["freq_hz"])), prev_idx, cur_idx))
        pairs.sort(key=lambda item: item[0])

        used_prev: set[int] = set()
        used_cur: set[int] = set()
        matches: list[tuple[int, int]] = []
        rejected: list[dict] = []
        for distance, prev_idx, cur_idx in pairs:
            if prev_idx in used_prev or cur_idx in used_cur:
                continue
            prev_freq = float(previous[prev_idx]["freq_hz"])
            cur_freq = float(current[cur_idx]["freq_hz"])
            denom = abs(prev_freq) if abs(prev_freq) > 0.0 else 1.0
            rel_change = abs(cur_freq - prev_freq) / denom
            if rel_change > float(max_rel_change):
                rejected.append(
                    {
                        "resonator_number": previous[prev_idx].get("resonator_number"),
                        "previous_freq_hz": prev_freq,
                        "current_freq_hz": cur_freq,
                        "distance_hz": float(distance),
                        "relative_change": float(rel_change),
                    }
                )
                used_prev.add(prev_idx)
                continue
            used_prev.add(prev_idx)
            used_cur.add(cur_idx)
            matches.append((prev_idx, cur_idx))
        return matches, rejected

    def _build_gaussian_minima_tracks(self, scans: list) -> tuple[list[dict], list[str]]:
        rows: list[dict] = []
        warnings: list[str] = []
        needed_numbers = 0

        for scan_index, scan in enumerate(scans):
            freq = np.asarray(scan.freq, dtype=float)
            norm = scan.baseline_filter.get("normalized", {})
            amp, _phase = _read_polar_series(
                norm if isinstance(norm, dict) else {},
                amplitude_key="norm_amp",
                phase_key="norm_phase_deg_unwrapped",
            )
            amp = np.asarray(amp, dtype=float)
            minima_idx = self._gaussian_minima_indices_for_scan(scan)
            minima_idx = minima_idx[minima_idx < freq.size]
            if freq.shape != amp.shape or freq.size == 0:
                warnings.append(f"{Path(scan.filename).name}: missing normalized amplitude")
                continue
            if minima_idx.size == 0:
                warnings.append(f"{Path(scan.filename).name}: no attached Gaussian minima indices")
            needed_numbers += int(minima_idx.size)
            rows.append(
                {
                    "scan": scan,
                    "scan_key": self._scan_key(scan),
                    "scan_index": scan_index,
                    "freq": freq,
                    "amp": amp,
                    "items": [
                        {"index": int(idx), "freq_hz": float(freq[int(idx)]), "resonator_number": None}
                        for idx in minima_idx
                    ],
                }
            )

        number_pool = self._next_unused_resonator_numbers(needed_numbers)
        next_number_idx = 0
        previous_items: list[dict] = []
        for row_index, row in enumerate(rows):
            current_items = list(row["items"])
            if row_index == 0:
                for item in current_items:
                    item["resonator_number"] = number_pool[next_number_idx]
                    next_number_idx += 1
                previous_items = current_items
                continue

            matches, rejected = self._match_previous_to_current(previous_items, current_items)
            for rejected_item in rejected:
                label = str(rejected_item.get("resonator_number", "")).strip() or "unknown"
                warnings.append(
                    f"{Path(row['scan'].filename).name}: stopped resonator {label}; "
                    f"nearest minimum changed by {100.0 * float(rejected_item['relative_change']):.3g}% "
                    f"(limit {100.0 * MAX_LINK_REL_FREQ_CHANGE:.3g}%)."
                )
            assigned_current: set[int] = set()
            for prev_idx, cur_idx in matches:
                current_items[cur_idx]["resonator_number"] = previous_items[prev_idx]["resonator_number"]
                assigned_current.add(cur_idx)
            for cur_idx, item in enumerate(current_items):
                if cur_idx in assigned_current:
                    continue
                item["resonator_number"] = number_pool[next_number_idx]
                next_number_idx += 1
            previous_items = current_items

        return rows, warnings

    def open_link_gaussian_minima_window(self) -> None:
        scans = self._selected_scans()
        if len(scans) < 2:
            messagebox.showwarning(
                "Need selected scans",
                "Select at least two VNA scans before linking Gaussian minima.",
            )
            return

        rows, warnings = self._build_gaussian_minima_tracks(scans)
        if not rows or not any(row["items"] for row in rows):
            detail = "No selected scans had attached Gaussian minima indices."
            if warnings:
                detail += "\n\n" + "\n".join(warnings[:10])
            messagebox.showwarning("No Gaussian minima", detail)
            return

        window = tk.Toplevel(self.root)
        window.title("Link Gaussian Minima to Resonator Numbers")
        window.geometry("1320x900")
        window.transient(self.root)

        controls = tk.Frame(window, padx=8, pady=8)
        controls.pack(side="top", fill="x")
        marker_count = sum(len(row["items"]) for row in rows)
        track_count = len({str(item["resonator_number"]) for row in rows for item in row["items"]})
        status_var = tk.StringVar(
            value=(
                f"Previewing {marker_count} minima in {track_count} linked resonator track(s). "
                f"Links require < {100.0 * MAX_LINK_REL_FREQ_CHANGE:g}% frequency change from the previous scan. "
                "Click Attach to write these numbers to the selected scans."
            )
        )
        tk.Label(controls, textvariable=status_var, anchor="w", justify="left").pack(side="left", fill="x", expand=True)

        fig = Figure(figsize=(12, 7))
        canvas = FigureCanvasTkAgg(fig, master=window)
        toolbar = NavigationToolbar2Tk(canvas, window)
        toolbar.update()
        toolbar.pack(side="top", fill="x")
        canvas.get_tk_widget().pack(fill="both", expand=True)

        self._draw_gaussian_minima_link_preview(fig, rows)
        canvas.draw_idle()

        def attach() -> None:
            self._attach_gaussian_minima_tracks(rows)
            status_var.set(f"Attached {marker_count} linked Gaussian minima to {len(rows)} selected scan(s).")
            messagebox.showinfo(
                "Linked minima attached",
                f"Attached {marker_count} linked Gaussian minima to {len(rows)} selected scan(s).",
                parent=window,
            )
            window.destroy()

        button_row = tk.Frame(window, padx=8, pady=8)
        button_row.pack(side="bottom", fill="x")
        tk.Button(button_row, text="Cancel", width=12, command=window.destroy).pack(side="right")
        tk.Button(button_row, text="Attach", width=12, command=attach).pack(side="right", padx=(0, 8))

        if warnings:
            self._log("Gaussian minima link warnings: " + " | ".join(warnings[:8]))

    def _draw_gaussian_minima_link_preview(self, fig: Figure, rows: list[dict]) -> None:
        fig.clear()
        ax = fig.add_subplot(111)
        spacing = 1.5
        offset_by_scan_key, tick_info = self._attached_resonance_editor_offset_map(rows, spacing)
        trace_colors = self._attached_resonance_editor_trace_colors()
        track_points: dict[str, list[tuple[float, float]]] = {}

        y_low = None
        y_high = None
        for row in rows:
            scan_key = str(row["scan_key"])
            offset = float(offset_by_scan_key.get(scan_key, 0.0))
            freq = np.asarray(row["freq"], dtype=float)
            amp_display = self._attached_resonance_editor_display_amp(row["amp"])
            freq_ghz = freq / 1.0e9
            y = amp_display + offset
            color = trace_colors[int(row.get("scan_index", 0)) % len(trace_colors)]
            ax.plot(freq_ghz, y, linewidth=1.0, color=color, alpha=0.9, zorder=1)
            y_low = float(np.min(y)) if y_low is None else min(y_low, float(np.min(y)))
            y_high = float(np.max(y)) if y_high is None else max(y_high, float(np.max(y)))

            for item in row["items"]:
                idx = int(item["index"])
                if idx < 0 or idx >= freq.size:
                    continue
                label = str(item["resonator_number"])
                x_ghz = float(freq[idx]) / 1.0e9
                y_val = float(amp_display[idx]) + offset
                ax.plot([x_ghz], [y_val], linestyle="none", marker="o", markersize=5, color="tab:red", zorder=4)
                ax.text(x_ghz, y_val - 0.18, label, ha="center", va="top", fontsize=7, color="tab:red", zorder=5)
                track_points.setdefault(label, []).append((x_ghz, y_val))

        for label, points in track_points.items():
            if len(points) < 2:
                continue
            points = sorted(points, key=lambda item: item[1])
            ax.plot(
                [pt[0] for pt in points],
                [pt[1] for pt in points],
                color="0.25",
                linewidth=0.8,
                alpha=0.5,
                zorder=2,
            )

        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Normalized |S21| + vertical offset")
        ax.set_title("Proposed Gaussian-Minima Resonator Links")
        ax.grid(True, alpha=0.3)
        ax.set_yticks([item[0] for item in tick_info])
        ax.set_yticklabels([item[1] for item in tick_info], fontsize=8)
        if y_low is not None and y_high is not None:
            ax.set_ylim(y_low - 0.25, y_high + 0.25)
        fig.subplots_adjust(left=0.12, right=0.985, bottom=0.09, top=0.95)

    def _attach_gaussian_minima_tracks(self, rows: list[dict]) -> None:
        attached_at = datetime.now().isoformat(timespec="seconds")
        marker_count = 0
        for row in rows:
            scan = row["scan"]
            payload = self._sheet_resonance_attachment(scan)
            assignments = payload["assignments"]
            for item in row["items"]:
                resonator_number = str(item["resonator_number"])
                frequency_hz = float(item["freq_hz"])
                assignments[resonator_number] = {
                    "frequency_hz": frequency_hz,
                    "sheet_path": "",
                    "sheet_name": "",
                    "row": 0,
                    "column": 0,
                    "identifier": self._sheet_identifier_for_scan(scan),
                    "source": "linked_gaussian_minima",
                    "gaussian_minimum_index": int(item["index"]),
                    "max_link_relative_frequency_change": float(MAX_LINK_REL_FREQ_CHANGE),
                    "attached_at": attached_at,
                    "attached_by": _current_user(),
                }
                marker_count += 1
            scan.processing_history.append(
                _make_event(
                    "attach_linked_gaussian_minima",
                    {
                        "marker_count": int(len(row["items"])),
                        "filename": scan.filename,
                        "max_link_relative_frequency_change": float(MAX_LINK_REL_FREQ_CHANGE),
                    },
                )
            )

        self.dataset.processing_history.append(
            _make_event(
                "attach_linked_gaussian_minima_selected",
                {
                    "scan_count": int(len(rows)),
                    "marker_count": int(marker_count),
                    "max_link_relative_frequency_change": float(MAX_LINK_REL_FREQ_CHANGE),
                },
            )
        )
        self._mark_dirty()
        self._refresh_status()
        self._autosave_dataset()
        self._log(f"Attached {marker_count} linked Gaussian minima resonator marker(s).")
