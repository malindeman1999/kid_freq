from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter import messagebox

from analysis_gui_support.analysis_models import _make_event, _read_polar_series

try:
    import winsound
except Exception:  # pragma: no cover - non-Windows fallback
    winsound = None

class AttachedResonanceEditorMixin:
    def open_attached_resonance_editor(self) -> None:
        scans = self._selected_scans()
        if not scans:
            messagebox.showwarning(
                "No selection",
                "No scans selected for analysis.\nUse 'Select Scans for Analysis' first.",
            )
            return

        if self.attached_res_edit_window is not None and self.attached_res_edit_window.winfo_exists():
            self.attached_res_edit_window.lift()
            self._render_attached_resonance_editor()
            return

        self.attached_res_edit_window = tk.Toplevel(self.root)
        self.attached_res_edit_window.title("Edit Resonator Markers")
        self.attached_res_edit_window.geometry("1320x900")
        self.attached_res_edit_window.protocol("WM_DELETE_WINDOW", self._attached_resonance_editor_exit)

        controls = tk.Frame(self.attached_res_edit_window, padx=8, pady=8)
        controls.pack(side="top", fill="x")
        controls_row = tk.Frame(controls)
        controls_row.pack(side="top", fill="x")
        action_controls = tk.Frame(controls_row)
        action_controls.pack(side="left", anchor="w")
        number_controls = tk.Frame(controls_row)
        number_controls.pack(side="left", anchor="w", padx=(12, 0))
        self.attached_res_edit_status_var = tk.StringVar(value="Normalized selected scans will be plotted.")
        tk.Label(controls, textvariable=self.attached_res_edit_status_var, anchor="w").pack(
            side="top", fill="x", pady=(8, 0)
        )
        self.attached_res_edit_add_button = tk.Button(
            number_controls, text="Add Resonator", width=14, command=self._attached_resonance_editor_toggle_add
        )
        self.attached_res_edit_add_button.pack(side="left")
        tk.Label(number_controls, text="Spacing").pack(side="left", padx=(10, 4))
        self.attached_res_edit_spacing_var = tk.DoubleVar(value=1.5)
        self.attached_res_edit_spacing_scale = tk.Scale(
            number_controls,
            from_=0.0,
            to=3.0,
            resolution=0.05,
            orient="horizontal",
            length=120,
            showvalue=True,
            variable=self.attached_res_edit_spacing_var,
        )
        self.attached_res_edit_spacing_scale.pack(side="left")
        self.attached_res_edit_spacing_scale.bind(
            "<ButtonRelease-1>",
            self._attached_resonance_editor_on_spacing_release,
        )
        self.attached_res_edit_spacing_scale.bind(
            "<KeyRelease>",
            self._attached_resonance_editor_on_spacing_release,
        )
        tk.Label(number_controls, text="Search width (kHz)").pack(side="left", padx=(10, 4))
        self.attached_res_edit_search_window_khz_var = tk.DoubleVar(value=300.0)
        self.attached_res_edit_search_window_scale = tk.Scale(
            number_controls,
            from_=25.0,
            to=2000.0,
            resolution=25.0,
            orient="horizontal",
            length=140,
            showvalue=True,
            variable=self.attached_res_edit_search_window_khz_var,
        )
        self.attached_res_edit_search_window_scale.pack(side="left")
        self.attached_res_edit_truncate_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            number_controls,
            text="Truncate |S21|",
            variable=self.attached_res_edit_truncate_var,
            command=self._attached_resonance_editor_on_truncate_toggle,
        ).pack(side="left", padx=(10, 4))
        self.attached_res_edit_truncate_threshold_var = tk.DoubleVar(value=1.5)
        self.attached_res_edit_truncate_threshold_scale = tk.Scale(
            number_controls,
            from_=1.0,
            to=2.0,
            resolution=0.05,
            orient="horizontal",
            length=110,
            showvalue=True,
            variable=self.attached_res_edit_truncate_threshold_var,
        )
        self.attached_res_edit_truncate_threshold_scale.pack(side="left")
        self.attached_res_edit_truncate_threshold_scale.bind(
            "<ButtonRelease-1>",
            self._attached_resonance_editor_on_truncate_release,
        )
        self.attached_res_edit_truncate_threshold_scale.bind(
            "<KeyRelease>",
            self._attached_resonance_editor_on_truncate_release,
        )
        tk.Label(number_controls, text="Working #").pack(side="left", padx=(10, 4))
        self.attached_res_edit_working_number_var = tk.StringVar(value="1")
        self.attached_res_edit_working_number_spinbox = tk.Spinbox(
            number_controls,
            from_=1,
            to=9999,
            increment=1,
            width=6,
            textvariable=self.attached_res_edit_working_number_var,
        )
        self.attached_res_edit_working_number_spinbox.pack(side="left")
        tk.Button(
            number_controls,
            text="Next Unused #",
            width=13,
            command=self._attached_resonance_editor_set_next_unused_number,
        ).pack(side="left", padx=(6, 0))
        self.attached_res_edit_renumber_button = tk.Button(
            action_controls,
            text="Renumber All Low->High",
            width=18,
            command=self._attached_resonance_editor_renumber_low_to_high,
        )
        self.attached_res_edit_renumber_button.pack(side="left", padx=(0, 8))
        self.attached_res_edit_undo_button = tk.Button(
            action_controls, text="Undo", width=10, command=self._attached_resonance_editor_undo
        )
        self.attached_res_edit_undo_button.pack(side="left", padx=(0, 8))
        tk.Button(
            action_controls, text="Delete Selected", width=14, command=self._attached_resonance_editor_delete_selected
        ).pack(side="left", padx=(0, 8))
        tk.Button(
            action_controls,
            text="Clear Sel. Scan Markers",
            width=20,
            command=self._attached_resonance_editor_clear_selected_scan_markers,
        ).pack(side="left", padx=(0, 8))
        tk.Button(
            action_controls, text="Reset View", width=12, command=self._attached_resonance_editor_reset_view
        ).pack(side="left", padx=(0, 8))
        self.attached_res_edit_save_button = tk.Button(
            action_controls, text="Save", width=10, command=self._attached_resonance_editor_save
        )
        self.attached_res_edit_save_button.pack(side="left", padx=(0, 8))
        self.attached_res_edit_exit_button = tk.Button(
            action_controls, text="Exit", width=12, command=self._attached_resonance_editor_exit
        )
        self.attached_res_edit_exit_button.pack(side="left")

        self.attached_res_edit_figure = Figure(figsize=(12, 7))
        self.attached_res_edit_canvas = FigureCanvasTkAgg(
            self.attached_res_edit_figure, master=self.attached_res_edit_window
        )
        self.attached_res_edit_toolbar = NavigationToolbar2Tk(
            self.attached_res_edit_canvas, self.attached_res_edit_window
        )
        self.attached_res_edit_toolbar.update()
        self.attached_res_edit_toolbar.pack(side="top", fill="x")
        self.attached_res_edit_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.attached_res_edit_canvas.mpl_connect(
            "button_press_event", self._attached_resonance_editor_on_click
        )

        self._attached_res_edit_selected = None
        self._attached_res_edit_pending_add = False
        self._attached_res_edit_default_xlim = None
        self._attached_res_edit_missing_normalized_warned = None
        self._attached_res_edit_warnings_cache = []
        self._attached_res_edit_overlay_artists = []
        self._attached_res_edit_marker_artists = {}
        self._attached_res_edit_track_artists = {}
        self._attached_res_edit_snapshot = self._attached_resonance_editor_capture_snapshot()
        self._attached_res_edit_undo_stack = []
        self._attached_res_edit_changed = False
        self._attached_resonance_editor_reset_working_number()
        self._render_attached_resonance_editor()


    def _close_attached_resonance_editor(self) -> None:
        if self.attached_res_edit_window is not None and self.attached_res_edit_window.winfo_exists():
            self.attached_res_edit_window.destroy()
        self.attached_res_edit_window = None
        self.attached_res_edit_canvas = None
        self.attached_res_edit_toolbar = None
        self.attached_res_edit_figure = None
        self.attached_res_edit_status_var = None
        self.attached_res_edit_working_number_var = None
        self.attached_res_edit_working_number_spinbox = None
        self.attached_res_edit_spacing_var = None
        self.attached_res_edit_spacing_scale = None
        self.attached_res_edit_search_window_khz_var = None
        self.attached_res_edit_search_window_scale = None
        self.attached_res_edit_truncate_var = None
        self.attached_res_edit_truncate_threshold_var = None
        self.attached_res_edit_truncate_threshold_scale = None
        self.attached_res_edit_add_button = None
        self.attached_res_edit_renumber_button = None
        self.attached_res_edit_undo_button = None
        self.attached_res_edit_save_button = None
        self.attached_res_edit_exit_button = None
        self.attached_res_edit_ax = None
        self._attached_res_edit_points = []
        self._attached_res_edit_rows_cache = []
        self._attached_res_edit_offset_by_scan_key = {}
        self._attached_res_edit_selected = None
        self._attached_res_edit_pending_add = False
        self._attached_res_edit_default_xlim = None
        self._attached_res_edit_missing_normalized_warned = None
        self._attached_res_edit_warnings_cache = []
        self._attached_res_edit_overlay_artists = []
        self._attached_res_edit_marker_artists = {}
        self._attached_res_edit_track_artists = {}
        self._attached_res_edit_snapshot = None
        self._attached_res_edit_undo_stack = []
        self._attached_res_edit_changed = False


    def _attached_resonance_editor_capture_snapshot(self, scan_keys: set[str] | None = None) -> dict:
        scan_payloads: dict[str, object] = {}
        scan_history_lengths: dict[str, int] = {}
        for scan in self.dataset.vna_scans:
            key = self._scan_key(scan)
            if scan_keys is not None and key not in scan_keys:
                continue
            payload = scan.candidate_resonators.get("sheet_resonances")
            scan_payloads[key] = copy.deepcopy(payload) if isinstance(payload, dict) else None
            scan_history_lengths[key] = len(scan.processing_history)
        return {
            "scan_payloads": scan_payloads,
            "scan_history_lengths": scan_history_lengths,
            "dataset_processing_history_len": len(self.dataset.processing_history),
            "dataset_transcript_len": len(self.dataset.transcript),
            "was_dirty": self._dirty,
            "selected": copy.deepcopy(self._attached_res_edit_selected),
            "working_number": self._attached_resonance_editor_working_number(),
        }


    def _attached_resonance_editor_apply_snapshot(self, snapshot: Optional[dict]) -> None:
        if not isinstance(snapshot, dict):
            return
        payloads = snapshot.get("scan_payloads", {})
        history_lengths = snapshot.get("scan_history_lengths", {})
        for scan in self.dataset.vna_scans:
            key = self._scan_key(scan)
            if key not in payloads:
                continue
            payload = payloads[key]
            if isinstance(payload, dict):
                scan.candidate_resonators["sheet_resonances"] = copy.deepcopy(payload)
            else:
                scan.candidate_resonators.pop("sheet_resonances", None)
            prior_len = int(history_lengths.get(key, len(scan.processing_history)))
            if len(scan.processing_history) > prior_len:
                del scan.processing_history[prior_len:]
        dataset_processing_len = int(snapshot.get("dataset_processing_history_len", len(self.dataset.processing_history)))
        if len(self.dataset.processing_history) > dataset_processing_len:
            del self.dataset.processing_history[dataset_processing_len:]
        dataset_transcript_len = int(snapshot.get("dataset_transcript_len", len(self.dataset.transcript)))
        if len(self.dataset.transcript) > dataset_transcript_len:
            del self.dataset.transcript[dataset_transcript_len:]
        self._dirty = bool(snapshot.get("was_dirty", self._dirty))
        selected = snapshot.get("selected")
        if isinstance(selected, tuple) and len(selected) == 2:
            self._attached_res_edit_selected = (str(selected[0]), str(selected[1]))
        else:
            self._attached_res_edit_selected = None
        if self.attached_res_edit_working_number_var is not None:
            working_number = str(snapshot.get("working_number", self._attached_resonance_editor_next_number()))
            self.attached_res_edit_working_number_var.set(working_number)
        self._refresh_status()
        self._reload_transcript_ui()


    def _attached_resonance_editor_restore_snapshot(self) -> None:
        self._attached_resonance_editor_apply_snapshot(self._attached_res_edit_snapshot)


    def _attached_resonance_editor_push_undo_snapshot(self, scan_keys: set[str] | None = None) -> None:
        self._attached_res_edit_undo_stack.append(
            self._attached_resonance_editor_capture_snapshot(scan_keys=scan_keys)
        )
        self._attached_resonance_editor_update_undo_button()


    def _attached_resonance_editor_save(self) -> bool:
        if not self._attached_res_edit_changed:
            if self.attached_res_edit_status_var is not None:
                self.attached_res_edit_status_var.set("Attached resonator edits are already saved.")
            self._attached_resonance_editor_update_save_button()
            return True
        self._mark_dirty()
        if not self._autosave_dataset():
            self._attached_resonance_editor_update_save_button()
            return False
        self._attached_res_edit_changed = False
        self._attached_res_edit_snapshot = self._attached_resonance_editor_capture_snapshot()
        self._attached_res_edit_undo_stack = []
        if self.attached_res_edit_status_var is not None:
            self.attached_res_edit_status_var.set("Attached resonator edits saved.")
        self._attached_resonance_editor_update_undo_button()
        self._attached_resonance_editor_update_save_button()
        self._log("Saved resonator marker edits.")
        return True


    def _attached_resonance_editor_exit(self) -> None:
        if not self._attached_res_edit_changed:
            self._close_attached_resonance_editor()
            return
        dialog = messagebox.Message(
            parent=self.attached_res_edit_window,
            title="Save resonator marker edits?",
            message="Save resonator marker edits before exiting?",
            icon=messagebox.WARNING,
            type=messagebox.YESNOCANCEL,
            default=messagebox.YES,
        )
        response = str(dialog.show()).lower()
        if response == "cancel":
            return
        if response == "yes":
            if not self._attached_resonance_editor_save():
                return
        else:
            self._attached_resonance_editor_restore_snapshot()
        self._close_attached_resonance_editor()


    def _selected_scans_for_attached_resonance_editor(
        self,
    ) -> tuple[list[dict], list[str]]:
        rows: list[dict] = []
        warnings: list[str] = []
        selected_scans = self._selected_scans()
        for idx, scan in enumerate(selected_scans):
            if not self._has_valid_normalized_output(scan):
                warnings.append(Path(scan.filename).name)
                continue
            payload = scan.candidate_resonators.get("sheet_resonances")
            assignments = payload.get("assignments") if isinstance(payload, dict) else {}
            if not isinstance(assignments, dict):
                assignments = {}
            freq = np.asarray(scan.freq, dtype=float)
            norm = scan.baseline_filter.get("normalized", {})
            amp, _phase = _read_polar_series(
                norm,
                amplitude_key="norm_amp",
                phase_key="norm_phase_deg_unwrapped",
            )
            amp = np.asarray(amp, dtype=float)
            if freq.shape != amp.shape or freq.size == 0:
                warnings.append(Path(scan.filename).name)
                continue
            order = np.argsort(freq)
            freq = freq[order]
            amp = amp[order]
            resonators: list[dict] = []
            for resonator_number, record in assignments.items():
                if not isinstance(record, dict):
                    continue
                try:
                    target_hz = float(record.get("frequency_hz"))
                except Exception:
                    continue
                if not (float(freq[0]) <= target_hz <= float(freq[-1])):
                    continue
                resonators.append(
                    {
                        "resonator_number": str(resonator_number).strip(),
                        "target_hz": target_hz,
                    }
                )
            rows.append(
                {
                    "scan": scan,
                    "scan_key": self._scan_key(scan),
                    "plot_group": scan.plot_group,
                    "freq": freq,
                    "amp": amp,
                    "scan_index": idx,
                    "resonators": sorted(resonators, key=lambda item: item["target_hz"]),
                }
            )
        return rows, warnings


    @staticmethod
    def _attached_resonance_editor_offset_map(
        rows: list[dict],
        spacing: float,
    ) -> tuple[dict[str, float], list[tuple[float, str]]]:
        level_keys: list[tuple[str, int]] = []
        labels_by_level: dict[tuple[str, int], list[str]] = {}
        for row in rows:
            plot_group = row.get("plot_group")
            if plot_group is None:
                level_key = ("scan", int(row.get("scan_index", 0)))
            else:
                level_key = ("group", int(plot_group))
            if level_key not in labels_by_level:
                level_keys.append(level_key)
                labels_by_level[level_key] = []
            file_timestamp = str(getattr(row["scan"], "file_timestamp", "")).strip()
            date_label = file_timestamp.split("T", 1)[0] if file_timestamp else "unknown date"
            labels_by_level[level_key].append(date_label)

        offset_by_scan_key: dict[str, float] = {}
        tick_info: list[tuple[float, str]] = []
        nlevels = len(level_keys)
        for level_pos, level_key in enumerate(level_keys):
            offset = float((nlevels - 1 - level_pos) * spacing)
            tick_y = offset + 1.0
            label_names = labels_by_level[level_key]
            label = label_names[0] if label_names else "unknown date"
            tick_info.append((tick_y, label))
            for row in rows:
                plot_group = row.get("plot_group")
                row_level_key = ("group", int(plot_group)) if plot_group is not None else ("scan", int(row.get("scan_index", 0)))
                if row_level_key == level_key:
                    offset_by_scan_key[str(row["scan_key"])] = offset
        return offset_by_scan_key, tick_info


    @staticmethod
    def _attached_resonance_editor_trace_colors() -> list[str]:
        return [
            "#8c564b",
            "#9467bd",
            "#ff7f0e",
            "#17becf",
            "#bcbd22",
            "#7f7f7f",
            "#e377c2",
            "#1f77b4",
            "#d62728",
            "#2ca02c",
            "#aec7e8",
            "#ffbb78",
        ]

    def _attached_resonance_editor_row_resonators(self, row: dict) -> list[dict]:
        scan = row["scan"]
        payload = scan.candidate_resonators.get("sheet_resonances")
        assignments = payload.get("assignments") if isinstance(payload, dict) else {}
        if not isinstance(assignments, dict):
            return []
        freq = np.asarray(row["freq"], dtype=float)
        if freq.size == 0:
            return []
        resonators: list[dict] = []
        for resonator_number, record in assignments.items():
            if not isinstance(record, dict):
                continue
            try:
                target_hz = float(record.get("frequency_hz"))
            except Exception:
                continue
            if not (float(freq[0]) <= target_hz <= float(freq[-1])):
                continue
            resonators.append(
                {
                    "resonator_number": str(resonator_number).strip(),
                    "target_hz": target_hz,
                }
            )
        return sorted(resonators, key=lambda item: item["target_hz"])

    def _attached_resonance_editor_clear_overlay_artists(self) -> None:
        for artist in list(getattr(self, "_attached_res_edit_overlay_artists", [])):
            try:
                artist.remove()
            except Exception:
                pass
        self._attached_res_edit_overlay_artists = []
        self._attached_res_edit_marker_artists = {}
        self._attached_res_edit_track_artists = {}

    @staticmethod
    def _attached_resonance_editor_marker_key(scan_key: str, resonator_number: str) -> tuple[str, str]:
        return (str(scan_key), str(resonator_number))

    @staticmethod
    def _attached_resonance_editor_apply_marker_style(marker_record: dict, is_selected: bool) -> None:
        line = marker_record.get("line")
        text = marker_record.get("text")
        if line is not None:
            line.set_markersize(9 if is_selected else 6)
            line.set_markeredgecolor("black" if is_selected else "tab:red")
        if text is not None:
            text.set_color("black" if is_selected else "tab:red")

    def _attached_resonance_editor_refresh_track_for_resonator(self, resonator_number: str) -> None:
        if self.attached_res_edit_ax is None:
            return
        label = str(resonator_number)
        points = [p for p in self._attached_res_edit_points if str(p.get("resonator_number")) == label]
        points = sorted(points, key=lambda item: float(item["y"]), reverse=True)
        line = self._attached_res_edit_track_artists.get(label)
        if len(points) < 2:
            if line is not None:
                try:
                    line.remove()
                except Exception:
                    pass
                self._attached_res_edit_track_artists.pop(label, None)
                if line in self._attached_res_edit_overlay_artists:
                    self._attached_res_edit_overlay_artists.remove(line)
            return
        x_data = [float(pt["x_ghz"]) for pt in points]
        y_data = [float(pt["y"]) for pt in points]
        if line is None:
            line = self.attached_res_edit_ax.plot(
                x_data,
                y_data,
                linestyle=":",
                linewidth=1.0,
                color="tab:red",
                alpha=0.9,
                zorder=2,
            )[0]
            self._attached_res_edit_track_artists[label] = line
            self._attached_res_edit_overlay_artists.append(line)
        else:
            line.set_data(x_data, y_data)

    def _attached_resonance_editor_draw_overlay(self) -> None:
        if self.attached_res_edit_ax is None:
            return
        rows = self._attached_res_edit_rows_cache
        offset_by_scan_key = self._attached_res_edit_offset_by_scan_key
        if not rows or not offset_by_scan_key:
            return
        ax = self.attached_res_edit_ax
        self._attached_resonance_editor_clear_overlay_artists()
        self._attached_res_edit_points = []
        resonator_tracks: dict[str, list[tuple[float, float]]] = {}
        resonator_markers: list[dict] = []
        overlay_artists: list[object] = []
        y_text_offset = 0.18

        for row in rows:
            scan = row["scan"]
            scan_key = str(row["scan_key"])
            offset = float(offset_by_scan_key.get(scan_key, 0.0))
            amp_display = self._attached_resonance_editor_display_amp(row["amp"])
            for resonator in self._attached_resonance_editor_row_resonators(row):
                target_hz = float(resonator["target_hz"])
                target_ghz = target_hz / 1.0e9
                y_pt = self._interpolate_y(row["freq"], amp_display, target_hz) + offset
                resonator_number = str(resonator["resonator_number"])
                point = {
                    "scan": scan,
                    "scan_key": scan_key,
                    "resonator_number": resonator_number,
                    "x_ghz": target_ghz,
                    "y": y_pt,
                    "freq_hz": target_hz,
                }
                self._attached_res_edit_points.append(point)
                resonator_markers.append(point)
                resonator_tracks.setdefault(resonator_number, []).append((target_ghz, y_pt))

        for _resonator_number, points in resonator_tracks.items():
            if len(points) < 2:
                continue
            points = sorted(points, key=lambda item: item[1], reverse=True)
            line = ax.plot(
                [pt[0] for pt in points],
                [pt[1] for pt in points],
                linestyle=":",
                linewidth=1.0,
                color="tab:red",
                alpha=0.9,
                zorder=2,
            )[0]
            overlay_artists.append(line)
            self._attached_res_edit_track_artists[str(_resonator_number)] = line

        for point in resonator_markers:
            is_selected = self._attached_res_edit_selected == (point["scan_key"], point["resonator_number"])
            marker_line = ax.plot(
                [point["x_ghz"]],
                [point["y"]],
                linestyle="none",
                marker="o",
                markersize=(9 if is_selected else 6),
                markerfacecolor="none",
                markeredgecolor=("black" if is_selected else "tab:red"),
                markeredgewidth=1.5,
                zorder=4,
            )[0]
            marker_text = ax.text(
                point["x_ghz"],
                point["y"] - y_text_offset,
                point["resonator_number"],
                ha="center",
                va="top",
                fontsize=8,
                color=("black" if is_selected else "tab:red"),
                zorder=5,
            )
            overlay_artists.append(marker_line)
            overlay_artists.append(marker_text)
            self._attached_res_edit_marker_artists[
                self._attached_resonance_editor_marker_key(point["scan_key"], point["resonator_number"])
            ] = {"line": marker_line, "text": marker_text}
        self._attached_res_edit_overlay_artists = overlay_artists

    def _attached_resonance_editor_fast_add_overlay_update(
        self,
        *,
        scan,
        resonator_number: str,
        target_hz: float,
        previous_selected: Optional[tuple[str, str]],
    ) -> None:
        if self.attached_res_edit_ax is None:
            self._attached_resonance_editor_redraw_overlay()
            return
        scan_key = self._scan_key(scan)
        rows = self._attached_res_edit_rows_cache
        row = None
        for candidate in rows:
            if str(candidate.get("scan_key")) == str(scan_key):
                row = candidate
                break
        if row is None:
            self._attached_resonance_editor_redraw_overlay()
            return
        offset = float(self._attached_res_edit_offset_by_scan_key.get(str(scan_key), 0.0))
        amp_display = self._attached_resonance_editor_display_amp(row["amp"])
        x_ghz = float(target_hz) / 1.0e9
        y_val = self._interpolate_y(row["freq"], amp_display, float(target_hz)) + offset
        marker_key = self._attached_resonance_editor_marker_key(str(scan_key), str(resonator_number))
        existing_marker = self._attached_res_edit_marker_artists.get(marker_key)
        if existing_marker is not None:
            for artist in (existing_marker.get("line"), existing_marker.get("text")):
                if artist is None:
                    continue
                try:
                    artist.remove()
                except Exception:
                    pass
                if artist in self._attached_res_edit_overlay_artists:
                    self._attached_res_edit_overlay_artists.remove(artist)
            self._attached_res_edit_marker_artists.pop(marker_key, None)
            self._attached_res_edit_points = [
                pt for pt in self._attached_res_edit_points
                if not (str(pt.get("scan_key")) == str(scan_key) and str(pt.get("resonator_number")) == str(resonator_number))
            ]

        point = {
            "scan": scan,
            "scan_key": str(scan_key),
            "resonator_number": str(resonator_number),
            "x_ghz": x_ghz,
            "y": y_val,
            "freq_hz": float(target_hz),
        }
        self._attached_res_edit_points.append(point)
        marker_line = self.attached_res_edit_ax.plot(
            [x_ghz],
            [y_val],
            linestyle="none",
            marker="o",
            markersize=6,
            markerfacecolor="none",
            markeredgecolor="tab:red",
            markeredgewidth=1.5,
            zorder=4,
        )[0]
        marker_text = self.attached_res_edit_ax.text(
            x_ghz,
            y_val - 0.18,
            str(resonator_number),
            ha="center",
            va="top",
            fontsize=8,
            color="tab:red",
            zorder=5,
        )
        self._attached_res_edit_overlay_artists.append(marker_line)
        self._attached_res_edit_overlay_artists.append(marker_text)
        marker_record = {"line": marker_line, "text": marker_text}
        self._attached_res_edit_marker_artists[marker_key] = marker_record

        if previous_selected is not None:
            prev_key = self._attached_resonance_editor_marker_key(previous_selected[0], previous_selected[1])
            prev_marker = self._attached_res_edit_marker_artists.get(prev_key)
            if prev_marker is not None and prev_key != marker_key:
                self._attached_resonance_editor_apply_marker_style(prev_marker, False)
        self._attached_resonance_editor_apply_marker_style(marker_record, True)
        self._attached_resonance_editor_refresh_track_for_resonator(str(resonator_number))

    def _attached_resonance_editor_update_status_message(self) -> None:
        if self.attached_res_edit_status_var is None:
            return
        warnings = list(getattr(self, "_attached_res_edit_warnings_cache", []))
        status = (
            "Left-click a resonator to select. Double-click on the same scan to move it. "
            "Use 'Add Resonator' then click a scan to add one."
        )
        if warnings:
            status += " Missing normalized data for: " + ", ".join(warnings[:6])
            if len(warnings) > 6:
                status += f", ... (+{len(warnings) - 6} more)"
        if self._attached_res_edit_pending_add:
            resonator_number = self._attached_resonance_editor_working_number()
            status = f"Add mode: click near a scan trace to add resonator {resonator_number}."
            if warnings:
                status += " Missing normalized data for: " + ", ".join(warnings[:4])
                if len(warnings) > 4:
                    status += f", ... (+{len(warnings) - 4} more)"
        self.attached_res_edit_status_var.set(status)

    def _attached_resonance_editor_redraw_overlay(self) -> None:
        if self.attached_res_edit_ax is None or self.attached_res_edit_canvas is None:
            self._render_attached_resonance_editor()
            return
        if not self._attached_res_edit_rows_cache or not self._attached_res_edit_offset_by_scan_key:
            self._render_attached_resonance_editor()
            return
        self._attached_resonance_editor_draw_overlay()
        self._attached_resonance_editor_update_status_message()
        self._attached_resonance_editor_update_add_button()
        self._attached_resonance_editor_update_undo_button()
        self._attached_resonance_editor_update_save_button()
        self.attached_res_edit_canvas.draw_idle()


    def _render_attached_resonance_editor(self) -> None:
        if self.attached_res_edit_figure is None or self.attached_res_edit_canvas is None:
            return
        rows, warnings = self._selected_scans_for_attached_resonance_editor()
        warning_tuple = tuple(sorted(set(warnings)))
        if warning_tuple:
            if self._attached_res_edit_missing_normalized_warned != warning_tuple:
                self._attached_res_edit_missing_normalized_warned = warning_tuple
                detail_lines = list(warning_tuple[:12])
                if len(warning_tuple) > 12:
                    detail_lines.append(f"... and {len(warning_tuple) - 12} more")
                messagebox.showwarning(
                    "Missing normalized data",
                    "The following selected scans do not have normalized data and will not be shown:\n\n"
                    + "\n".join(detail_lines),
                    parent=self.attached_res_edit_window,
                )
        else:
            self._attached_res_edit_missing_normalized_warned = None
        if not rows:
            message = "No selected scans with normalized data."
            if warnings:
                message += " Missing normalized data for: " + ", ".join(warnings[:6])
                if len(warnings) > 6:
                    message += f", ... (+{len(warnings) - 6} more)"
            self._attached_res_edit_rows_cache = []
            self._attached_res_edit_offset_by_scan_key = {}
            self._attached_res_edit_warnings_cache = list(warnings)
            self._attached_res_edit_points = []
            self._attached_res_edit_overlay_artists = []
            if self.attached_res_edit_status_var is not None:
                self.attached_res_edit_status_var.set(message)
            self.attached_res_edit_figure.clear()
            ax = self.attached_res_edit_figure.add_subplot(111)
            ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            self._attached_resonance_editor_update_save_button()
            self.attached_res_edit_canvas.draw_idle()
            return

        prior_xlim = None
        if self.attached_res_edit_ax is not None:
            try:
                prior_xlim = self.attached_res_edit_ax.get_xlim()
            except Exception:
                prior_xlim = None

        self.attached_res_edit_figure.clear()
        ax = self.attached_res_edit_figure.add_subplot(111)
        self.attached_res_edit_ax = ax
        self._attached_res_edit_points = []
        self._attached_resonance_editor_clear_overlay_artists()
        self._attached_res_edit_rows_cache = rows
        offset_by_scan_key, tick_info = self._attached_resonance_editor_offset_map(
            rows,
            self._attached_resonance_editor_curve_spacing(),
        )
        self._attached_res_edit_offset_by_scan_key = dict(offset_by_scan_key)

        freq_min = min(float(row["freq"][0]) for row in rows) / 1.0e9
        freq_max = max(float(row["freq"][-1]) for row in rows) / 1.0e9
        x_pad = max((freq_max - freq_min) * 0.01, 1e-6)

        y_by_scan_key: dict[str, np.ndarray] = {}
        trace_colors = self._attached_resonance_editor_trace_colors()
        for row in rows:
            scan_key = str(row["scan_key"])
            offset = float(offset_by_scan_key[scan_key])
            freq_ghz = np.asarray(row["freq"], dtype=float) / 1.0e9
            amp_display = self._attached_resonance_editor_display_amp(row["amp"])
            y = amp_display + offset
            y_by_scan_key[scan_key] = y
            trace_color = trace_colors[int(row.get("scan_index", 0)) % len(trace_colors)]
            ax.plot(freq_ghz, y, linewidth=1.0, color=trace_color, alpha=0.9, zorder=1)

        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Normalized |S21| + vertical offset")
        ax.grid(True, alpha=0.3)
        ax.set_yticks([item[0] for item in tick_info])
        ax.set_yticklabels([item[1] for item in tick_info], fontsize=8)

        y_low = min(float(np.min(y_by_scan_key[str(row["scan_key"])])) for row in rows)
        y_high = max(float(np.max(y_by_scan_key[str(row["scan_key"])])) for row in rows)
        ax.set_ylim(y_low - 0.2, y_high + 0.2)
        self._attached_res_edit_default_xlim = (freq_min - 0.5 * x_pad, freq_max + 2.0 * x_pad)
        if prior_xlim is not None:
            ax.set_xlim(prior_xlim)
        else:
            ax.set_xlim(self._attached_res_edit_default_xlim)

        self._attached_res_edit_warnings_cache = list(warnings)
        self._attached_resonance_editor_draw_overlay()
        self._attached_resonance_editor_update_status_message()
        self._attached_resonance_editor_update_add_button()
        self._attached_resonance_editor_update_undo_button()
        self._attached_resonance_editor_update_save_button()

        self.attached_res_edit_figure.subplots_adjust(left=0.12, right=0.985, bottom=0.09, top=0.96)
        self.attached_res_edit_canvas.draw_idle()


    def _attached_resonance_editor_reset_view(self) -> None:
        if self.attached_res_edit_ax is None or self._attached_res_edit_default_xlim is None:
            return
        self.attached_res_edit_ax.set_xlim(self._attached_res_edit_default_xlim)
        if self.attached_res_edit_canvas is not None:
            self.attached_res_edit_canvas.draw_idle()


    def _attached_resonance_editor_curve_spacing(self) -> float:
        if self.attached_res_edit_spacing_var is None:
            return 1.5
        try:
            value = float(self.attached_res_edit_spacing_var.get())
        except Exception:
            value = 1.5
        value = min(max(value, 0.0), 3.0)
        if abs(value - float(self.attached_res_edit_spacing_var.get())) > 1e-12:
            self.attached_res_edit_spacing_var.set(value)
        return value


    def _attached_resonance_editor_truncate_enabled(self) -> bool:
        return bool(self.attached_res_edit_truncate_var.get()) if self.attached_res_edit_truncate_var is not None else True


    def _attached_resonance_editor_search_window_hz(self) -> float:
        if self.attached_res_edit_search_window_khz_var is None:
            return 3.0e5
        try:
            value_khz = float(self.attached_res_edit_search_window_khz_var.get())
        except Exception:
            value_khz = 300.0
        value_khz = min(max(value_khz, 25.0), 2000.0)
        if abs(value_khz - float(self.attached_res_edit_search_window_khz_var.get())) > 1e-12:
            self.attached_res_edit_search_window_khz_var.set(value_khz)
        return value_khz * 1.0e3


    def _attached_resonance_editor_truncate_threshold(self) -> float:
        if self.attached_res_edit_truncate_threshold_var is None:
            return 1.5
        try:
            value = float(self.attached_res_edit_truncate_threshold_var.get())
        except Exception:
            value = 1.5
        value = min(max(value, 1.0), 2.0)
        if abs(value - float(self.attached_res_edit_truncate_threshold_var.get())) > 1e-12:
            self.attached_res_edit_truncate_threshold_var.set(value)
        return value


    def _attached_resonance_editor_display_amp(self, amp: Sequence[float]) -> np.ndarray:
        amp_arr = np.asarray(amp, dtype=float)
        if not self._attached_resonance_editor_truncate_enabled():
            return amp_arr
        return np.minimum(amp_arr, self._attached_resonance_editor_truncate_threshold())


    def _attached_resonance_editor_on_spacing_release(self, _event) -> None:
        if self.attached_res_edit_window is None or not self.attached_res_edit_window.winfo_exists():
            return
        prior_xlim = None
        if self.attached_res_edit_ax is not None:
            try:
                prior_xlim = self.attached_res_edit_ax.get_xlim()
            except Exception:
                prior_xlim = None
        self._render_attached_resonance_editor()
        if prior_xlim is not None and self.attached_res_edit_ax is not None:
            self.attached_res_edit_ax.set_xlim(prior_xlim)
            if self.attached_res_edit_canvas is not None:
                self.attached_res_edit_canvas.draw_idle()


    def _attached_resonance_editor_on_truncate_toggle(self) -> None:
        self._render_attached_resonance_editor()


    def _attached_resonance_editor_on_truncate_release(self, _event) -> None:
        self._render_attached_resonance_editor()


    def _attached_resonance_editor_toggle_add(self) -> None:
        self._attached_res_edit_pending_add = not self._attached_res_edit_pending_add
        self._attached_res_edit_selected = None
        if self.attached_res_edit_status_var is not None:
            if self._attached_res_edit_pending_add:
                resonator_number = self._attached_resonance_editor_working_number()
                self.attached_res_edit_status_var.set(
                    f"Add mode: click near a selected normalized scan to add resonator {resonator_number}."
                )
            else:
                self.attached_res_edit_status_var.set("Add mode deactivated.")
        self._attached_resonance_editor_redraw_overlay()


    def _attached_resonance_editor_update_add_button(self) -> None:
        if self.attached_res_edit_add_button is None:
            return
        if self._attached_res_edit_pending_add:
            self.attached_res_edit_add_button.configure(
                relief="sunken",
                bg="light green",
                activebackground="light green",
            )
            return
        self.attached_res_edit_add_button.configure(
            relief="raised",
            bg=self._default_button_bg,
            activebackground=self._default_button_activebg,
        )


    def _attached_resonance_editor_update_save_button(self) -> None:
        if self.attached_res_edit_save_button is None:
            return
        if self._attached_res_edit_changed:
            self.attached_res_edit_save_button.configure(
                bg="pink",
                activebackground="pink",
            )
            return
        self.attached_res_edit_save_button.configure(
            bg=self._default_button_bg,
            activebackground=self._default_button_activebg,
        )


    def _attached_resonance_editor_update_undo_button(self) -> None:
        if self.attached_res_edit_undo_button is None:
            return
        if self._attached_res_edit_undo_stack:
            self.attached_res_edit_undo_button.configure(state="normal")
        else:
            self.attached_res_edit_undo_button.configure(state="disabled")


    def _attached_resonance_editor_undo(self) -> None:
        if not self._attached_res_edit_undo_stack:
            if self.attached_res_edit_status_var is not None:
                self.attached_res_edit_status_var.set("Nothing to undo.")
            self._attached_resonance_editor_update_undo_button()
            return
        snapshot = self._attached_res_edit_undo_stack.pop()
        self._attached_resonance_editor_apply_snapshot(snapshot)
        self._attached_res_edit_changed = bool(self._attached_res_edit_undo_stack)
        self._attached_resonance_editor_update_undo_button()
        self._attached_resonance_editor_update_save_button()
        if self.attached_res_edit_status_var is not None:
            self.attached_res_edit_status_var.set("Undid last resonator marker edit.")
        self._attached_resonance_editor_redraw_overlay()


    def _attached_resonance_editor_renumber_low_to_high(self) -> None:
        all_scans = list(self.dataset.vna_scans)
        resonator_values: dict[str, list[float]] = {}
        for scan in all_scans:
            payload = scan.candidate_resonators.get("sheet_resonances")
            assignments = payload.get("assignments") if isinstance(payload, dict) else {}
            if not isinstance(assignments, dict):
                continue
            for resonator_number, record in assignments.items():
                if not isinstance(record, dict):
                    continue
                try:
                    freq_hz = float(record.get("frequency_hz"))
                except Exception:
                    continue
                if not np.isfinite(freq_hz):
                    continue
                label = str(resonator_number).strip()
                if not label:
                    continue
                resonator_values.setdefault(label, []).append(freq_hz)

        ordered_labels = sorted(
            (label for label, values in resonator_values.items() if values),
            key=lambda label: (
                float(np.mean(np.asarray(resonator_values[label], dtype=float))),
                self._resonator_sort_key(label),
            ),
        )
        if not ordered_labels:
            if self.attached_res_edit_status_var is not None:
                self.attached_res_edit_status_var.set("No resonator markers are available to renumber.")
            return

        renumber_map = {
            old_label: str(new_idx)
            for new_idx, old_label in enumerate(ordered_labels, start=1)
        }
        if all(old_label == new_label for old_label, new_label in renumber_map.items()):
            if self.attached_res_edit_status_var is not None:
                self.attached_res_edit_status_var.set(
                    "Resonator markers are already numbered low to high across the dataset."
                )
            return

        self._attached_resonance_editor_push_undo_snapshot()
        changed_scans = 0
        for scan in all_scans:
            payload = scan.candidate_resonators.get("sheet_resonances")
            assignments = payload.get("assignments") if isinstance(payload, dict) else {}
            if not isinstance(assignments, dict) or not assignments:
                continue
            new_assignments: dict[str, dict] = {}
            scan_changed = False
            for resonator_number, record in assignments.items():
                if not isinstance(record, dict):
                    continue
                old_label = str(resonator_number).strip()
                new_label = renumber_map.get(old_label, old_label)
                if new_label != old_label:
                    scan_changed = True
                new_assignments[new_label] = record
            payload["assignments"] = new_assignments
            if scan_changed:
                scan.processing_history.append(
                    _make_event(
                        "renumber_attached_resonators_low_to_high",
                        {
                            "renumbered_labels": int(len(ordered_labels)),
                        },
                    )
                )
                changed_scans += 1

        self.dataset.processing_history.append(
            _make_event(
                "renumber_attached_resonators_low_to_high_dataset",
                {
                    "renumbered_labels": int(len(ordered_labels)),
                    "changed_scans": int(changed_scans),
                },
            )
        )

        if self._attached_res_edit_selected is not None:
            selected_scan_key, selected_number = self._attached_res_edit_selected
            mapped = renumber_map.get(str(selected_number).strip(), str(selected_number).strip())
            self._attached_res_edit_selected = (selected_scan_key, mapped)
        self._attached_res_edit_changed = True
        self._attached_resonance_editor_reset_working_number()
        self._attached_resonance_editor_update_save_button()
        self._attached_resonance_editor_update_undo_button()
        if self.attached_res_edit_status_var is not None:
            self.attached_res_edit_status_var.set(
                f"Renumbered {len(ordered_labels)} resonator marker number(s) from low to high mean frequency across {changed_scans} scan(s)."
            )
        self._attached_resonance_editor_redraw_overlay()


    def _attached_resonance_editor_working_number(self) -> str:
        if self.attached_res_edit_working_number_var is None:
            return "1"
        raw_value = str(self.attached_res_edit_working_number_var.get()).strip()
        try:
            number = max(1, int(raw_value))
        except Exception:
            number = 1
        if raw_value != str(number):
            self.attached_res_edit_working_number_var.set(str(number))
        return str(number)


    def _attached_resonance_editor_next_number(self) -> str:
        max_number = 0
        for scan in self._selected_scans():
            payload = scan.candidate_resonators.get("sheet_resonances")
            assignments = payload.get("assignments") if isinstance(payload, dict) else {}
            if not isinstance(assignments, dict):
                continue
            for resonator_number in assignments.keys():
                try:
                    parsed = int(str(resonator_number).strip())
                except Exception:
                    continue
                if parsed >= 1:
                    max_number = max(max_number, parsed)
        return str(max_number + 1)


    def _attached_resonance_editor_reset_working_number(self) -> None:
        if self.attached_res_edit_working_number_var is None:
            return
        self.attached_res_edit_working_number_var.set(self._attached_resonance_editor_next_number())

    def _attached_resonance_editor_set_next_unused_number(self) -> None:
        next_number = self._attached_resonance_editor_next_number()
        if self.attached_res_edit_working_number_var is not None:
            self.attached_res_edit_working_number_var.set(next_number)
        if self.attached_res_edit_status_var is not None:
            self.attached_res_edit_status_var.set(f"Working resonator number set to next unused value: {next_number}.")


    def _attached_resonance_editor_delete_selected(self) -> None:
        if self._attached_res_edit_selected is None:
            if self.attached_res_edit_status_var is not None:
                self.attached_res_edit_status_var.set("No resonator selected to delete.")
            return
        scan_key, resonator_number = self._attached_res_edit_selected
        for scan in self.dataset.vna_scans:
            if self._scan_key(scan) != scan_key:
                continue
            payload = scan.candidate_resonators.get("sheet_resonances")
            if not isinstance(payload, dict):
                break
            assignments = payload.get("assignments")
            if not isinstance(assignments, dict):
                break
            self._attached_resonance_editor_push_undo_snapshot(scan_keys={str(scan_key)})
            assignments.pop(resonator_number, None)
            if not assignments:
                scan.candidate_resonators.pop("sheet_resonances", None)
            self.dataset.processing_history.append(
                _make_event(
                    "delete_attached_resonator",
                    {"scan": scan.filename, "resonator_number": resonator_number},
                )
            )
            self._attached_res_edit_changed = True
            break
        self._attached_res_edit_selected = None
        self._attached_resonance_editor_reset_working_number()
        self._attached_resonance_editor_redraw_overlay()


    def _attached_resonance_editor_clear_selected_scan_markers(self) -> None:
        scans = self._selected_scans()
        if not scans:
            if self.attached_res_edit_status_var is not None:
                self.attached_res_edit_status_var.set("No selected scans are available.")
            return

        scans_with_markers = []
        for scan in scans:
            payload = scan.candidate_resonators.get("sheet_resonances")
            assignments = payload.get("assignments") if isinstance(payload, dict) else {}
            if isinstance(assignments, dict) and assignments:
                scans_with_markers.append(scan)

        if not scans_with_markers:
            if self.attached_res_edit_status_var is not None:
                self.attached_res_edit_status_var.set("Selected scans do not have attached resonator markers.")
            return

        names = [Path(str(scan.filename)).name for scan in scans_with_markers]
        ok = messagebox.askyesno(
            "Clear Resonator Markers",
            f"Clear attached resonator markers from {len(scans_with_markers)} selected scan(s)?\n\n"
            + "\n".join(names[:10])
            + ("\n..." if len(names) > 10 else ""),
            parent=self.attached_res_edit_window,
        )
        if not ok:
            return

        self._attached_resonance_editor_push_undo_snapshot(
            scan_keys={self._scan_key(scan) for scan in scans_with_markers}
        )
        cleared = 0
        for scan in scans_with_markers:
            scan.candidate_resonators.pop("sheet_resonances", None)
            scan.processing_history.append(
                _make_event(
                    "clear_attached_resonator_markers",
                    {"scan": scan.filename},
                )
            )
            cleared += 1
        self.dataset.processing_history.append(
            _make_event(
                "clear_attached_resonator_markers_selected_scans",
                {
                    "selected_count": len(scans),
                    "cleared_count": cleared,
                    "filenames": [scan.filename for scan in scans_with_markers],
                },
            )
        )
        self._attached_res_edit_selected = None
        self._attached_res_edit_changed = True
        self._attached_resonance_editor_reset_working_number()
        if self.attached_res_edit_status_var is not None:
            self.attached_res_edit_status_var.set(
                f"Cleared attached resonator markers from {cleared} selected scan(s)."
            )
        self._attached_resonance_editor_redraw_overlay()


    def _attached_resonance_editor_on_click(self, event) -> None:
        if self.attached_res_edit_ax is None or event.inaxes != self.attached_res_edit_ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if not self._attached_resonance_editor_click_is_within_plot(float(event.xdata), float(event.ydata)):
            if self.attached_res_edit_status_var is not None:
                self.attached_res_edit_status_var.set("Click inside the visible plot window to edit resonators.")
            return
        if self.attached_res_edit_toolbar is not None:
            mode = getattr(self.attached_res_edit_toolbar, "mode", "")
            if str(mode).strip():
                if self.attached_res_edit_status_var is not None:
                    self.attached_res_edit_status_var.set(
                        f"Navigation mode active ({str(mode).strip()}). Finish pan/zoom to resume editing."
                    )
                return

        if self._attached_res_edit_pending_add:
            self._attached_resonance_editor_add_at_click(float(event.xdata), float(event.ydata))
            return
        nearest = self._attached_resonance_editor_find_nearest_point(float(event.xdata), float(event.ydata))

        if event.dblclick and self._attached_res_edit_selected is not None:
            self._attached_resonance_editor_move_selected(float(event.xdata), float(event.ydata))
            return

        if nearest is None:
            self._attached_res_edit_selected = None
            if self.attached_res_edit_status_var is not None:
                self.attached_res_edit_status_var.set("No resonator selected.")
            self._attached_resonance_editor_redraw_overlay()
            return

        self._attached_res_edit_selected = (nearest["scan_key"], nearest["resonator_number"])
        if self.attached_res_edit_status_var is not None:
            self.attached_res_edit_status_var.set(
                f"Selected resonator {nearest['resonator_number']} on {Path(nearest['scan'].filename).name}."
            )
        self._attached_resonance_editor_redraw_overlay()


    def _attached_resonance_editor_find_nearest_point(self, x_ghz: float, y_val: float) -> Optional[dict]:
        if not self._attached_res_edit_points or self.attached_res_edit_ax is None:
            return None
        x0, x1 = self.attached_res_edit_ax.get_xlim()
        y0, y1 = self.attached_res_edit_ax.get_ylim()
        x_span = max(abs(x1 - x0), 1e-9)
        y_span = max(abs(y1 - y0), 1e-9)
        best = None
        best_metric = None
        for point in self._attached_res_edit_points:
            metric = ((point["x_ghz"] - x_ghz) / x_span) ** 2 + ((point["y"] - y_val) / y_span) ** 2
            if best_metric is None or metric < best_metric:
                best_metric = metric
                best = point
        if best_metric is None or best_metric > 0.0025:
            return None
        return best


    def _attached_resonance_editor_click_is_within_plot(self, x_ghz: float, y_val: float) -> bool:
        if self.attached_res_edit_ax is None:
            return False
        try:
            x0, x1 = self.attached_res_edit_ax.get_xlim()
            y0, y1 = self.attached_res_edit_ax.get_ylim()
        except Exception:
            return False
        lo_x, hi_x = (float(x0), float(x1)) if float(x0) <= float(x1) else (float(x1), float(x0))
        lo_y, hi_y = (float(y0), float(y1)) if float(y0) <= float(y1) else (float(y1), float(y0))
        return lo_x <= float(x_ghz) <= hi_x and lo_y <= float(y_val) <= hi_y


    def _attached_resonance_editor_signal_success(self) -> None:
        widget = self.attached_res_edit_window if self.attached_res_edit_window is not None else self.root
        if winsound is not None:
            try:
                winsound.PlaySound(
                    "SystemDefault",
                    winsound.SND_ALIAS | winsound.SND_ASYNC | winsound.SND_NODEFAULT,
                )
                return
            except Exception:
                pass
        try:
            widget.bell()
        except Exception:
            pass


    def _attached_resonance_editor_row_for_add_click(
        self,
        rows: list[dict],
        offset_by_scan_key: dict[str, float],
        click_hz: float,
        y_val: float,
        *,
        window_hz: float,
        visible_range_hz: Optional[tuple[float, float]],
    ) -> Optional[tuple[dict, float]]:
        group_offsets_in_order: list[float] = []
        seen_offsets: set[float] = set()
        for row in rows:
            offset = float(offset_by_scan_key.get(str(row["scan_key"]), 0.0))
            if offset in seen_offsets:
                continue
            seen_offsets.add(offset)
            group_offsets_in_order.append(offset)

        chosen_offset = None
        for offset in group_offsets_in_order:
            if offset <= float(y_val) <= offset + 1.0:
                chosen_offset = offset
                break
        if chosen_offset is None:
            return None

        best_row = None
        best_target_hz = None
        best_metric = None
        for row in rows:
            offset = float(offset_by_scan_key.get(str(row["scan_key"]), 0.0))
            if abs(offset - chosen_offset) > 1e-12:
                continue
            freq_arr = np.asarray(row["freq"], dtype=float)
            if freq_arr.size == 0:
                continue
            if not (float(freq_arr[0]) <= float(click_hz) <= float(freq_arr[-1])):
                continue
            target_hz = self._attached_resonance_editor_minimum_near_click(
                row["freq"],
                row["amp"],
                click_hz,
                window_hz=window_hz,
                visible_range_hz=visible_range_hz,
            )
            if target_hz is None:
                continue
            metric = abs(float(target_hz) - float(click_hz))
            if best_metric is None or metric < best_metric:
                best_metric = metric
                best_row = row
                best_target_hz = float(target_hz)
        if best_row is None or best_target_hz is None:
            return None
        return best_row, best_target_hz


    def _show_attached_resonance_minimum_search_diagnostic(
        self,
        *,
        row: dict,
        click_hz: float,
        window_hz: float,
        visible_range_hz: Optional[tuple[float, float]],
        detail: str,
    ) -> None:
        if self.attached_res_edit_window is None or not self.attached_res_edit_window.winfo_exists():
            return
        freq_arr = np.asarray(row["freq"], dtype=float)
        amp_display = self._attached_resonance_editor_display_amp(row["amp"])
        if freq_arr.size == 0 or amp_display.size == 0:
            messagebox.showwarning("Add resonator failed", detail, parent=self.attached_res_edit_window)
            return

        if visible_range_hz is not None:
            lo_visible, hi_visible = visible_range_hz
        else:
            lo_visible, hi_visible = float(freq_arr[0]), float(freq_arr[-1])
        display_mask = (freq_arr >= float(lo_visible)) & (freq_arr <= float(hi_visible))
        if not np.any(display_mask):
            display_mask = np.ones(freq_arr.shape, dtype=bool)
            lo_visible, hi_visible = float(freq_arr[0]), float(freq_arr[-1])

        lo_search = max(float(lo_visible), float(click_hz) - float(window_hz))
        hi_search = min(float(hi_visible), float(click_hz) + float(window_hz))
        search_mask = (freq_arr >= lo_search) & (freq_arr <= hi_search)

        window = tk.Toplevel(self.attached_res_edit_window)
        window.title("Add Resonator Failed")
        window.geometry("920x560")
        window.transient(self.attached_res_edit_window)

        controls = tk.Frame(window, padx=8, pady=8)
        controls.pack(side="top", fill="x")
        tk.Label(
            controls,
            text=detail,
            anchor="w",
            justify="left",
            wraplength=880,
        ).pack(side="left", fill="x", expand=True)
        tk.Button(controls, text="Close", width=10, command=window.destroy).pack(side="right", padx=(8, 0))

        fig = Figure(figsize=(9.2, 4.8))
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        ax.plot(freq_arr[display_mask] / 1.0e9, amp_display[display_mask], color="tab:blue", linewidth=1.2)
        ax.axvspan(lo_search / 1.0e9, hi_search / 1.0e9, color="gold", alpha=0.25, label="searched window")
        ax.axvline(click_hz / 1.0e9, color="crimson", linestyle="--", linewidth=1.5, label="mouse click")
        if np.any(search_mask):
            ax.plot(
                freq_arr[search_mask] / 1.0e9,
                amp_display[search_mask],
                color="darkorange",
                linewidth=2.0,
            )
        ax.set_xlim(float(lo_visible) / 1.0e9, float(hi_visible) / 1.0e9)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Displayed |S21|")
        ax.set_title(Path(row["scan"].filename).name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        canvas.draw_idle()


    def _attached_resonance_editor_add_at_click(self, x_ghz: float, y_val: float) -> None:
        rows = self._attached_res_edit_rows_cache
        offset_by_scan_key = self._attached_res_edit_offset_by_scan_key
        if not rows or not offset_by_scan_key:
            rows, _warnings = self._selected_scans_for_attached_resonance_editor()
            offset_by_scan_key, _tick_info = self._attached_resonance_editor_offset_map(
                rows,
                self._attached_resonance_editor_curve_spacing(),
            )
        if not rows:
            return
        visible_range_hz = self._attached_resonance_editor_visible_range_hz()
        click_hz = x_ghz * 1.0e9
        target = self._attached_resonance_editor_row_for_add_click(
            rows,
            offset_by_scan_key,
            click_hz,
            y_val,
            window_hz=self._attached_resonance_editor_search_window_hz(),
            visible_range_hz=visible_range_hz,
        )
        if target is None:
            if self.attached_res_edit_status_var is not None:
                self.attached_res_edit_status_var.set(
                    "Click inside a group's 0-1 band at a frequency where a visible local minimum can be found."
                )
            return
        best_row, target_hz = target
        if visible_range_hz is not None:
            lo_hz, hi_hz = visible_range_hz
            if not (lo_hz <= click_hz <= hi_hz):
                if self.attached_res_edit_status_var is not None:
                    self.attached_res_edit_status_var.set("Click inside the visible plot window to add a resonator.")
                return
        resonator_number = self._attached_resonance_editor_working_number()
        scan = best_row["scan"]
        if visible_range_hz is not None:
            lo_hz, hi_hz = visible_range_hz
            if not (lo_hz <= target_hz <= hi_hz):
                detail = (
                    f"Attached-resonator add aborted because snapped target "
                    f"{target_hz / 1.0e9:.9g} GHz is outside the visible range "
                    f"{lo_hz / 1.0e9:.9g} to {hi_hz / 1.0e9:.9g} GHz."
                )
                if self.attached_res_edit_status_var is not None:
                    self.attached_res_edit_status_var.set(detail)
                messagebox.showwarning("Add resonator failed", detail, parent=self.attached_res_edit_window)
                self._log(
                    f"Attach resonator warning: click at {x_ghz:.9g} GHz on {Path(scan.filename).name} "
                    f"snapped to off-screen target {target_hz / 1.0e9:.9g} GHz."
                )
                return
        self._attached_resonance_editor_signal_success()
        payload = self._sheet_resonance_attachment(scan)
        assignments = payload["assignments"]
        self._attached_resonance_editor_push_undo_snapshot(scan_keys={self._scan_key(scan)})
        assignments[resonator_number] = {
            "frequency_hz": target_hz,
            "sheet_path": "",
            "sheet_name": "",
            "row": 0,
            "column": 0,
            "identifier": self._sheet_identifier_for_scan(scan),
        }
        self.dataset.processing_history.append(
            _make_event(
                "add_attached_resonator",
                {"scan": scan.filename, "resonator_number": resonator_number, "frequency_hz": target_hz},
            )
        )
        previous_selected = self._attached_res_edit_selected
        self._attached_res_edit_changed = True
        self._attached_res_edit_selected = (self._scan_key(scan), resonator_number)
        self._attached_resonance_editor_fast_add_overlay_update(
            scan=scan,
            resonator_number=str(resonator_number),
            target_hz=float(target_hz),
            previous_selected=previous_selected,
        )
        self._attached_resonance_editor_update_status_message()
        self._attached_resonance_editor_update_add_button()
        self._attached_resonance_editor_update_undo_button()
        self._attached_resonance_editor_update_save_button()
        if self.attached_res_edit_canvas is not None:
            self.attached_res_edit_canvas.draw_idle()


    def _attached_resonance_editor_visible_range_hz(self) -> Optional[tuple[float, float]]:
        if self.attached_res_edit_ax is None:
            return None
        try:
            x0, x1 = self.attached_res_edit_ax.get_xlim()
        except Exception:
            return None
        lo_ghz, hi_ghz = (float(x0), float(x1)) if float(x0) <= float(x1) else (float(x1), float(x0))
        return (lo_ghz * 1.0e9, hi_ghz * 1.0e9)


    def _attached_resonance_editor_minimum_near_click(
        self,
        freq_hz: Sequence[float],
        amp: Sequence[float],
        click_hz: float,
        window_hz: float = 3.0e5,
        visible_range_hz: Optional[tuple[float, float]] = None,
    ) -> Optional[float]:
        freq_arr = np.asarray(freq_hz, dtype=float)
        amp_arr = np.asarray(amp, dtype=float)
        if freq_arr.size == 0 or amp_arr.size == 0:
            return None
        window_mask = np.abs(freq_arr - float(click_hz)) <= float(window_hz)
        if visible_range_hz is not None:
            lo_hz, hi_hz = visible_range_hz
            visible_mask = (freq_arr >= float(lo_hz)) & (freq_arr <= float(hi_hz))
            window_mask = window_mask & visible_mask
        if np.any(window_mask):
            candidate_indices = np.flatnonzero(window_mask)
            best_local_idx = int(candidate_indices[int(np.argmin(amp_arr[window_mask]))])
            return float(freq_arr[best_local_idx])
        return None



    def _attached_resonance_editor_move_selected(self, x_ghz: float, y_val: float) -> None:
        if self._attached_res_edit_selected is None:
            return
        scan_key, resonator_number = self._attached_res_edit_selected
        rows = self._attached_res_edit_rows_cache
        offset_by_scan_key = self._attached_res_edit_offset_by_scan_key
        if not rows or not offset_by_scan_key:
            rows, _warnings = self._selected_scans_for_attached_resonance_editor()
            offset_by_scan_key, _tick_info = self._attached_resonance_editor_offset_map(
                rows,
                self._attached_resonance_editor_curve_spacing(),
            )
        target_row = None
        for row in rows:
            if row["scan_key"] != scan_key:
                continue
            offset = float(offset_by_scan_key[str(row["scan_key"])])
            amp_display = self._attached_resonance_editor_display_amp(row["amp"])
            y_trace = self._interpolate_y(row["freq"], amp_display, x_ghz * 1.0e9) + offset
            if abs(y_trace - y_val) <= 0.35:
                target_row = row
                break
        if target_row is None:
            if self.attached_res_edit_status_var is not None:
                self.attached_res_edit_status_var.set("Double-click on the same scan trace to move the selected resonator.")
            return
        scan = target_row["scan"]
        payload = scan.candidate_resonators.get("sheet_resonances")
        if not isinstance(payload, dict):
            return
        assignments = payload.get("assignments")
        if not isinstance(assignments, dict) or resonator_number not in assignments:
            return
        self._attached_resonance_editor_push_undo_snapshot(scan_keys={str(scan_key)})
        nearest_idx = int(np.argmin(np.abs(np.asarray(target_row["freq"], dtype=float) - x_ghz * 1.0e9)))
        target_hz = float(target_row["freq"][nearest_idx])
        assignments[resonator_number]["frequency_hz"] = target_hz
        self.dataset.processing_history.append(
            _make_event(
                "move_attached_resonator",
                {"scan": scan.filename, "resonator_number": resonator_number, "frequency_hz": target_hz},
            )
        )
        self._attached_res_edit_changed = True
        self._attached_resonance_editor_redraw_overlay()

def main() -> None:
    root = tk.Tk()
    DataAnalysisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
