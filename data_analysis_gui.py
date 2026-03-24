from __future__ import annotations

import copy
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from openpyxl import load_workbook
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter import filedialog, messagebox, simpledialog, scrolledtext, ttk

from analysis_gui_support.analysis_io import (
    DATASETS_DIR,
    DEFAULT_DATASET_FILE,
    _dataset_dir,
    _dataset_pickle_path,
    _load_dataset,
    _load_vna_file,
    _read_app_state,
    _safe_name,
    _save_dataset,
    _write_app_state,
)
from analysis_gui_support.analysis_models import (
    Dataset,
    VNAScan,
    _current_user,
    _make_event,
    _read_polar_series,
)
from analysis_gui_support.gui_mixins.analysis_selector_plot_mixin import AnalysisSelectorPlotMixin
from analysis_gui_support.gui_mixins.baseline_filter_mixin import BaselineFilterMixin
from analysis_gui_support.gui_mixins.dsdf_convolution_mixin import DSDFConvolutionMixin
from analysis_gui_support.gui_mixins.gaussian_convolution_mixin import GaussianConvolutionMixin
from analysis_gui_support.gui_mixins.interpolation_smooth_mixin import InterpolationSmoothMixin
from analysis_gui_support.gui_mixins.normalization_mixin import NormalizationMixin
from analysis_gui_support.gui_mixins.phase_correction2_mixin import PhaseCorrection2Mixin
from analysis_gui_support.gui_mixins.resonance_selection_mixin import ResonanceSelectionMixin
from analysis_gui_support.gui_mixins.third_phase_correction_mixin import ThirdPhaseCorrectionMixin
from analysis_gui_support.gui_mixins.synthetic_generator_mixin import SyntheticGeneratorMixin
from analysis_gui_support.gui_mixins.unwrap_phase_mixin import UnwrapPhaseMixin

class DataAnalysisGUI(
    AnalysisSelectorPlotMixin,
    BaselineFilterMixin,
    DSDFConvolutionMixin,
    GaussianConvolutionMixin,
    InterpolationSmoothMixin,
    NormalizationMixin,
    PhaseCorrection2Mixin,
    ResonanceSelectionMixin,
    ThirdPhaseCorrectionMixin,
    SyntheticGeneratorMixin,
    UnwrapPhaseMixin,
):
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("VNA Data Analysis")
        self.dataset_path = _read_app_state().resolve()
        self.dataset = _load_dataset(self.dataset_path)

        self.dataset_meta_var = tk.StringVar()
        self.dataset_label_var = tk.StringVar()
        self.scan_count_var = tk.StringVar()
        self.selection_var = tk.StringVar()
        self.saved_var = tk.StringVar()
        self.synth_button: Optional[tk.Button] = None
        self.select_scans_button: Optional[tk.Button] = None
        self.unwrap_button: Optional[tk.Button] = None
        self.phase2_button: Optional[tk.Button] = None
        self.phase3_button: Optional[tk.Button] = None
        self.baseline_button: Optional[tk.Button] = None
        self.interp_button: Optional[tk.Button] = None
        self._dirty = False
        self.baseline_window: Optional[tk.Toplevel] = None
        self.baseline_canvas: Optional[FigureCanvasTkAgg] = None
        self.baseline_toolbar: Optional[NavigationToolbar2Tk] = None
        self.baseline_figure: Optional[Figure] = None
        self.width_slider: Optional[tk.Scale] = None
        self.step_slider: Optional[tk.Scale] = None
        self.low_slope_slider: Optional[tk.Scale] = None
        self.retain_slider: Optional[tk.Scale] = None
        self.center_slider: Optional[tk.Scale] = None
        self._baseline_after_id: Optional[str] = None
        self._baseline_preview_results: Dict[str, Dict[str, np.ndarray]] = {}
        self.baseline_status_var: Optional[tk.StringVar] = None
        self.baseline_progress: Optional[ttk.Progressbar] = None
        self.baseline_attach_button: Optional[tk.Button] = None
        self._baseline_worker_thread: Optional[threading.Thread] = None
        self._baseline_worker_queue: "queue.Queue[dict]" = queue.Queue()
        self._baseline_compute_running: bool = False
        self._baseline_recompute_pending: bool = False
        self._baseline_compute_context: Dict[str, object] = {}
        self.interp_window: Optional[tk.Toplevel] = None
        self.interp_canvas: Optional[FigureCanvasTkAgg] = None
        self.interp_toolbar: Optional[NavigationToolbar2Tk] = None
        self.interp_figure: Optional[Figure] = None
        self.interp_status_var: Optional[tk.StringVar] = None
        self.interp_smooth_slider: Optional[tk.Scale] = None
        self.interp_attach_button: Optional[tk.Button] = None
        self.interp_preview: Dict[str, Dict[str, np.ndarray]] = {}
        self.norm_button: Optional[tk.Button] = None
        self.norm_window: Optional[tk.Toplevel] = None
        self.norm_canvas: Optional[FigureCanvasTkAgg] = None
        self.norm_toolbar: Optional[NavigationToolbar2Tk] = None
        self.norm_figure: Optional[Figure] = None
        self.norm_status_var: Optional[tk.StringVar] = None
        self.norm_attach_button: Optional[tk.Button] = None
        self.norm_preview: Dict[str, Dict[str, np.ndarray]] = {}
        self.gauss_button: Optional[tk.Button] = None
        self.gauss_window: Optional[tk.Toplevel] = None
        self.gauss_canvas: Optional[FigureCanvasTkAgg] = None
        self.gauss_toolbar: Optional[NavigationToolbar2Tk] = None
        self.gauss_figure: Optional[Figure] = None
        self.gauss_slider: Optional[tk.Scale] = None
        self.gauss_threshold_slider: Optional[tk.Scale] = None
        self.gauss_min_region_slider: Optional[tk.Scale] = None
        self.gauss_auto_y_var: Optional[tk.BooleanVar] = None
        self.gauss_status_var: Optional[tk.StringVar] = None
        self.gauss_attach_button: Optional[tk.Button] = None
        self.gauss_preview: Dict[str, Dict[str, np.ndarray]] = {}
        self.dsdf_button: Optional[tk.Button] = None
        self.dsdf_window: Optional[tk.Toplevel] = None
        self.dsdf_canvas: Optional[FigureCanvasTkAgg] = None
        self.dsdf_toolbar: Optional[NavigationToolbar2Tk] = None
        self.dsdf_figure: Optional[Figure] = None
        self.dsdf_fwhm_slider: Optional[tk.Scale] = None
        self.dsdf_threshold_slider: Optional[tk.Scale] = None
        self.dsdf_min_region_slider: Optional[tk.Scale] = None
        self.dsdf_auto_y_var: Optional[tk.BooleanVar] = None
        self.dsdf_show_phase_context_var: Optional[tk.BooleanVar] = None
        self.dsdf_use_corrected_context_var: Optional[tk.BooleanVar] = None
        self.dsdf_status_var: Optional[tk.StringVar] = None
        self.dsdf_attach_button: Optional[tk.Button] = None
        self.dsdf_preview: Dict[str, Dict[str, np.ndarray]] = {}
        self.synth_window: Optional[tk.Toplevel] = None
        self.synth_canvas: Optional[FigureCanvasTkAgg] = None
        self.synth_toolbar: Optional[NavigationToolbar2Tk] = None
        self.synth_figure: Optional[Figure] = None
        self.synth_source_var: Optional[tk.StringVar] = None
        self.synth_status_var: Optional[tk.StringVar] = None
        self.synth_auto_y_var: Optional[tk.BooleanVar] = None
        self.synth_generate_button: Optional[tk.Button] = None
        self.synth_num_res_slider: Optional[tk.Scale] = None
        self.synth_freq_offset_slider: Optional[tk.Scale] = None
        self.synth_num_files_slider: Optional[tk.Scale] = None
        self.synth_qc_slider: Optional[tk.Scale] = None
        self.synth_qi_slider: Optional[tk.Scale] = None
        self.synth_source_path = None
        self.synth_freq = None
        self.synth_preview_files: List[np.ndarray] = []
        self.synth_amp_ax = None
        self.synth_iq_ax = None
        self.unwrap_window: Optional[tk.Toplevel] = None
        self.unwrap_canvas: Optional[FigureCanvasTkAgg] = None
        self.unwrap_toolbar: Optional[NavigationToolbar2Tk] = None
        self.unwrap_figure: Optional[Figure] = None
        self.unwrap_threshold_slider: Optional[tk.Scale] = None
        self.unwrap_max_passes_slider: Optional[tk.Scale] = None
        self.unwrap_min_sep_slider: Optional[tk.Scale] = None
        self.unwrap_apply_exact_360_var: Optional[tk.BooleanVar] = None
        self.unwrap_correct_congruent_var: Optional[tk.BooleanVar] = None
        self.unwrap_p_random_var: Optional[tk.StringVar] = None
        self.unwrap_p_random_entry: Optional[tk.Entry] = None
        self.unwrap_auto_y_var: Optional[tk.BooleanVar] = None
        self.unwrap_mod360_var: Optional[tk.BooleanVar] = None
        self.unwrap_status_var: Optional[tk.StringVar] = None
        self.unwrap_progress: Optional[ttk.Progressbar] = None
        self.unwrap_update_button: Optional[tk.Button] = None
        self.unwrap_attach_button: Optional[tk.Button] = None
        self.unwrap_preview: Dict[str, Dict[str, np.ndarray]] = {}
        self.phase2_window: Optional[tk.Toplevel] = None
        self.phase2_canvas: Optional[FigureCanvasTkAgg] = None
        self.phase2_toolbar: Optional[NavigationToolbar2Tk] = None
        self.phase2_figure: Optional[Figure] = None
        self.phase2_auto_y_var: Optional[tk.BooleanVar] = None
        self.phase2_mod360_var: Optional[tk.BooleanVar] = None
        self.phase2_status_var: Optional[tk.StringVar] = None
        self.phase2_progress: Optional[ttk.Progressbar] = None
        self.phase2_update_button: Optional[tk.Button] = None
        self.phase3_window: Optional[tk.Toplevel] = None
        self.phase3_canvas: Optional[FigureCanvasTkAgg] = None
        self.phase3_toolbar: Optional[NavigationToolbar2Tk] = None
        self.phase3_figure: Optional[Figure] = None
        self.phase3_auto_y_var: Optional[tk.BooleanVar] = None
        self.phase3_mod360_var: Optional[tk.BooleanVar] = None
        self.phase3_status_var: Optional[tk.StringVar] = None
        self.res_button: Optional[tk.Button] = None
        self.res_window: Optional[tk.Toplevel] = None
        self.res_canvas: Optional[FigureCanvasTkAgg] = None
        self.res_toolbar: Optional[NavigationToolbar2Tk] = None
        self.res_figure: Optional[Figure] = None
        self.res_fit_button: Optional[tk.Button] = None
        self.res_status_var: Optional[tk.StringVar] = None
        self.res_fr_var: Optional[tk.StringVar] = None
        self.res_qi_var: Optional[tk.StringVar] = None
        self.res_qc_var: Optional[tk.StringVar] = None
        self.res_qc_phase_var: Optional[tk.StringVar] = None
        self.res_a_mag_var: Optional[tk.StringVar] = None
        self.res_a_phase_var: Optional[tk.StringVar] = None
        self.res_tau_var: Optional[tk.StringVar] = None
        self.res_fix_fr_var: Optional[tk.BooleanVar] = None
        self.res_fix_qi_var: Optional[tk.BooleanVar] = None
        self.res_fix_qc_var: Optional[tk.BooleanVar] = None
        self.res_fix_qc_phase_var: Optional[tk.BooleanVar] = None
        self.res_fix_a_mag_var: Optional[tk.BooleanVar] = None
        self.res_fix_a_phase_var: Optional[tk.BooleanVar] = None
        self.res_fix_tau_var: Optional[tk.BooleanVar] = None
        self.res_auto_y_var: Optional[tk.BooleanVar] = None
        self.res_display_mode_var: Optional[tk.StringVar] = None
        self.res_fit_mode_var: Optional[tk.StringVar] = None
        self.res_span_selector = None
        self.res_amp_ax = None
        self.res_iq_ax = None
        self.res_model_preview: Optional[dict] = None
        self.plot_scans_window: Optional[tk.Toplevel] = None
        self.plot_scans_canvas: Optional[FigureCanvasTkAgg] = None
        self.plot_scans_toolbar: Optional[NavigationToolbar2Tk] = None
        self.plot_scans_figure: Optional[Figure] = None
        self.plot_scans_show_amp_var: Optional[tk.BooleanVar] = None
        self.plot_scans_show_phase_var: Optional[tk.BooleanVar] = None
        self.plot_scans_group_var: Optional[tk.BooleanVar] = None
        self.plot_scans_data_mode_var: Optional[tk.StringVar] = None
        self.plot_scans_raw_radio: Optional[tk.Radiobutton] = None
        self.plot_scans_normalized_radio: Optional[tk.Radiobutton] = None
        self.plot_scans_show_gaussian_var: Optional[tk.BooleanVar] = None
        self.plot_scans_show_dsdf_var: Optional[tk.BooleanVar] = None
        self.plot_scans_show_2pi_var: Optional[tk.BooleanVar] = None
        self.plot_scans_show_vna_phase_var: Optional[tk.BooleanVar] = None
        self.plot_scans_show_other_phase_var: Optional[tk.BooleanVar] = None
        self.plot_scans_show_attached_res_var: Optional[tk.BooleanVar] = None
        self.plot_scans_auto_y_var: Optional[tk.BooleanVar] = None
        self.plot_scans_status_var: Optional[tk.StringVar] = None
        self.attached_res_edit_window: Optional[tk.Toplevel] = None
        self.attached_res_edit_canvas: Optional[FigureCanvasTkAgg] = None
        self.attached_res_edit_toolbar: Optional[NavigationToolbar2Tk] = None
        self.attached_res_edit_figure: Optional[Figure] = None
        self.attached_res_edit_status_var: Optional[tk.StringVar] = None
        self.attached_res_edit_working_number_var: Optional[tk.StringVar] = None
        self.attached_res_edit_working_number_spinbox: Optional[tk.Spinbox] = None
        self.attached_res_edit_spacing_var: Optional[tk.DoubleVar] = None
        self.attached_res_edit_spacing_scale: Optional[tk.Scale] = None
        self.attached_res_edit_truncate_var: Optional[tk.BooleanVar] = None
        self.attached_res_edit_truncate_threshold_var: Optional[tk.DoubleVar] = None
        self.attached_res_edit_truncate_threshold_scale: Optional[tk.Scale] = None
        self.attached_res_edit_add_button: Optional[tk.Button] = None
        self.attached_res_edit_exit_button: Optional[tk.Button] = None
        self.attached_res_edit_ax = None
        self._attached_res_edit_points: List[dict] = []
        self._attached_res_edit_selected: Optional[tuple[str, str]] = None
        self._attached_res_edit_pending_add: bool = False
        self._attached_res_edit_default_xlim: Optional[tuple[float, float]] = None
        self._attached_res_edit_missing_normalized_warned: Optional[tuple[str, ...]] = None
        self._attached_res_edit_snapshot: Optional[dict] = None
        self._attached_res_edit_changed: bool = False
        self._res_scan_key: Optional[str] = None
        self._res_selected_range: Optional[tuple[float, float]] = None
        self._res_manual_ylim: Optional[tuple[float, float]] = None
        self._last_resonance_scan_key: Optional[str] = None
        self._default_button_bg = "SystemButtonFace"
        self._default_button_activebg = "SystemButtonFace"
        self._build_layout()
        if self.synth_button is not None:
            self._default_button_bg = str(self.synth_button.cget("bg"))
            self._default_button_activebg = str(self.synth_button.cget("activebackground"))
        self._reload_transcript_ui()
        self._refresh_status()
        self._log("Application started.")

    @staticmethod
    def _scan_key(scan: VNAScan) -> str:
        return f"{scan.filename}|{scan.loaded_at}"

    def _prune_selected_scan_keys(self) -> None:
        valid_keys = {self._scan_key(scan) for scan in self.dataset.vna_scans}
        self.dataset.selected_scan_keys = [
            key for key in self.dataset.selected_scan_keys if key in valid_keys
        ]

    def _selected_scans(self) -> List[VNAScan]:
        selected_keys = set(self.dataset.selected_scan_keys)
        return [scan for scan in self.dataset.vna_scans if self._scan_key(scan) in selected_keys]

    def _build_layout(self) -> None:
        frame = tk.Frame(self.root, padx=10, pady=10)
        frame.pack(fill="both", expand=True)

        top = tk.Frame(frame)
        top.pack(side="top", fill="x")
        tk.Label(top, textvariable=self.dataset_meta_var, anchor="w", justify="left").pack(
            anchor="w"
        )
        tk.Label(top, text="Active dataset file:", anchor="w", justify="left").pack(
            anchor="w", pady=(6, 0)
        )
        tk.Label(
            top, textvariable=self.dataset_label_var, wraplength=900, justify="left", anchor="w"
        ).pack(anchor="w")
        tk.Label(top, textvariable=self.scan_count_var, anchor="w", justify="left").pack(
            anchor="w", pady=(8, 0)
        )
        tk.Label(top, textvariable=self.selection_var, anchor="w", justify="left").pack(
            anchor="w", pady=(2, 0)
        )
        tk.Label(top, textvariable=self.saved_var, anchor="w", justify="left").pack(
            anchor="w", pady=(2, 10)
        )

        bottom = tk.Frame(frame)
        bottom.pack(side="top", fill="both", expand=True)
        left = tk.Frame(bottom)
        left.pack(side="left", fill="y")
        right = tk.Frame(bottom, padx=12)
        right.pack(side="left", fill="both", expand=True)

        tk.Button(left, text="New Dataset", width=24, command=self.start_new_dataset).pack(
            anchor="w", pady=2
        )
        self.synth_button = tk.Button(
            left, text="Generate Synthetic Data", width=24, command=self.open_synthetic_generator_window
        )
        self.synth_button.pack(anchor="w", pady=2)
        tk.Button(
            left, text="Load Different Dataset", width=24, command=self.load_different_dataset
        ).pack(anchor="w", pady=2)
        tk.Button(left, text="Load VNA Scan(s)", width=24, command=self.load_vna_scan).pack(
            anchor="w", pady=2
        )
        tk.Button(
            left, text="Remove VNA Scan(s)", width=24, command=self.remove_vna_scans
        ).pack(anchor="w", pady=2)
        self.select_scans_button = tk.Button(
            left, text="Select Scans for Analysis", width=24, command=self.open_analysis_selector
        )
        self.select_scans_button.pack(anchor="w", pady=2)
        tk.Button(
            left, text="Group Selected Scans", width=24, command=self.group_selected_scans_for_plotting
        ).pack(anchor="w", pady=2)
        tk.Button(
            left, text="Plot Selected VNA Scans", width=24, command=self.plot_selected_vna_scans
        ).pack(anchor="w", pady=2)
        self.unwrap_button = tk.Button(
            left, text="Phase Correction 1", width=24, command=self.open_unwrap_phase_window
        )
        self.unwrap_button.pack(anchor="w", pady=2)
        self.phase2_button = tk.Button(
            left, text="Phase Correction 2", width=24, command=self.open_second_phase_correction_window
        )
        self.phase2_button.pack(anchor="w", pady=2)
        self.phase3_button = tk.Button(
            left, text="Phase Correction 3", width=24, command=self.open_third_phase_correction_window
        )
        self.phase3_button.pack(anchor="w", pady=2)
        self.baseline_button = tk.Button(
            left, text="Baseline Filtering", width=24, command=self.open_baseline_filter_window
        )
        self.baseline_button.pack(anchor="w", pady=2)
        self.interp_button = tk.Button(
            left, text="Interp + Smooth", width=24, command=self.open_interp_smooth_window
        )
        self.interp_button.pack(anchor="w", pady=2)
        self.norm_button = tk.Button(
            left, text="Normalize Baseline", width=24, command=self.open_normalization_window
        )
        self.norm_button.pack(anchor="w", pady=2)
        self.gauss_button = tk.Button(
            left, text="Gaussian Convolve |S21|", width=24, command=self.open_gaussian_convolution_window
        )
        self.gauss_button.pack(anchor="w", pady=2)
        self.dsdf_button = tk.Button(
            left, text="Gaussian Convolve |dS21/df|", width=24, command=self.open_dsdf_convolution_window
        )
        self.dsdf_button.pack(anchor="w", pady=2)
        self.res_button = tk.Button(
            left, text="Resonance Selection", width=24, command=self.open_resonance_selection_window
        )
        self.res_button.pack(anchor="w", pady=2)
        tk.Button(
            left, text="Load Resonators From Sheet", width=24, command=self.open_resonance_sheet_loader
        ).pack(anchor="w", pady=2)
        tk.Button(
            left, text="Plot Attached Resonators", width=24, command=self.open_attached_resonance_plotter
        ).pack(anchor="w", pady=2)
        tk.Button(
            left, text="Attach Res. to Sel. Scans", width=24, command=self.open_attached_resonance_editor
        ).pack(anchor="w", pady=2)
        tk.Button(
            left, text="Clear Selected Attachments", width=24, command=self.clear_selected_scan_attachments
        ).pack(anchor="w", pady=(12, 2))

        tk.Label(right, text="Transcript:", anchor="w", justify="left").pack(anchor="w", pady=(0, 2))
        self.log_text = scrolledtext.ScrolledText(right, width=110, height=10, state="disabled")
        self.log_text.pack(fill="both", expand=True)

    def _append_transcript_line(self, timestamp: str, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _reload_transcript_ui(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        for entry in self.dataset.transcript:
            self.log_text.insert("end", f"[{entry['timestamp']}] {entry['message']}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _log(self, message: str) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        self.dataset.transcript.append({"timestamp": timestamp, "message": message})
        self._append_transcript_line(timestamp, message)

    def _select_setting_option(
        self, title: str, prompt: str, options: List[str], default_index: int = 0
    ) -> Optional[int]:
        if not options:
            return None
        if len(options) == 1:
            return 0

        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("760x360")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text=prompt, anchor="w", justify="left", wraplength=720).pack(
            fill="x", padx=10, pady=(10, 4)
        )
        listbox = tk.Listbox(dialog, width=110, height=12)
        listbox.pack(fill="both", expand=True, padx=10, pady=4)
        for item in options:
            listbox.insert(tk.END, item)
        if 0 <= default_index < len(options):
            listbox.selection_set(default_index)
            listbox.see(default_index)
        else:
            listbox.selection_set(0)

        selected_index: Dict[str, Optional[int]] = {"value": None}

        def choose() -> None:
            sel = listbox.curselection()
            if not sel:
                return
            selected_index["value"] = int(sel[0])
            dialog.destroy()

        def cancel() -> None:
            dialog.destroy()

        btns = tk.Frame(dialog)
        btns.pack(fill="x", padx=10, pady=(4, 10))
        tk.Button(btns, text="Use Selected", command=choose).pack(side="right")
        tk.Button(btns, text="Cancel", command=cancel).pack(side="right", padx=(0, 8))

        dialog.wait_window()
        return selected_index["value"]

    def _refresh_status(self) -> None:
        created = self.dataset.created_at if self.dataset.created_at else "Unassigned"
        name = self.dataset.dataset_name if self.dataset.dataset_name else "Unassigned"
        self.dataset_meta_var.set(f"Dataset: {name} | Created: {created}")
        self._prune_selected_scan_keys()
        self.dataset_label_var.set(str(self.dataset_path))
        self.scan_count_var.set(f"Loaded VNA scans in dataset: {len(self.dataset.vna_scans)}")
        selected_names = [Path(scan.filename).name for scan in self._selected_scans()]
        if len(selected_names) > 3:
            selected_text = ", ".join(selected_names[:3]) + f", ... (+{len(selected_names) - 3} more)"
        elif selected_names:
            selected_text = ", ".join(selected_names)
        else:
            selected_text = "None"
        self.selection_var.set(
            f"Selected scans for analysis ({len(selected_names)}): {selected_text}"
        )
        last_saved = self.dataset.last_saved_at if self.dataset.last_saved_at else "Never"
        size_text = "0.00 MB"
        try:
            if self.dataset_path.exists():
                size_mb = self.dataset_path.stat().st_size / (1024.0 * 1024.0)
                size_text = f"{size_mb:.2f} MB"
        except Exception:
            size_text = "Unknown"
        self.saved_var.set(f"Last saved: {last_saved} | Dataset file size: {size_text}")
        self._update_save_button_state()
        self._update_interp_button_state()
        self._update_norm_button_state()
        self._update_gauss_button_state()
        self._update_dsdf_button_state()
        self._update_unwrap_button_state()
        self._update_phase2_button_state()
        self._update_phase3_button_state()
        self._update_baseline_button_state()
        self._update_select_scans_button_state()
        self._update_res_button_state()

    def _has_data_to_save(self) -> bool:
        return len(self.dataset.vna_scans) > 0

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._update_save_button_state()

    def _mark_clean(self) -> None:
        self._dirty = False
        self._update_save_button_state()

    def _update_save_button_state(self) -> None:
        return

    def _attach_save_and_close_baseline(self) -> None:
        self.attach_baseline_filter()
        self._close_baseline_window()

    def _attach_save_and_close_interp(self) -> None:
        self._interp_attach()
        self._interp_close()

    def _attach_save_and_close_norm(self) -> None:
        self._norm_attach()
        self._norm_close()

    def _attach_save_and_close_gauss(self) -> None:
        self._gauss_attach()
        self._gauss_close()

    def _attach_save_and_close_dsdf(self) -> None:
        self._dsdf_attach()
        self._dsdf_close()

    def _attach_save_and_close_unwrap(self) -> None:
        self._unwrap_attach()
        self._unwrap_close()

    def _attach_save_and_close_phase2(self) -> None:
        self._phase2_attach()
        self._phase2_close()

    def _attach_save_and_close_phase3(self) -> None:
        self._phase3_attach()
        self._phase3_close()

    def _persist_dataset(self) -> bool:
        try:
            if not self.dataset.created_at:
                self.dataset.created_at = datetime.now().isoformat(timespec="seconds")
            if self.dataset.dataset_name:
                self.dataset_path = _dataset_pickle_path(self.dataset).resolve()
            else:
                self.dataset_path = self.dataset_path.resolve()

            self.dataset.processing_history.append(
                _make_event("save_dataset", {"dataset_path": str(self.dataset_path)})
            )
            _save_dataset(self.dataset, self.dataset_path)
            _write_app_state(self.dataset_path)
            self._mark_clean()
            self._refresh_status()
            return True
        except Exception as exc:
            self._mark_dirty()
            self._refresh_status()
            self._log(f"Save failed: {exc}")
            messagebox.showerror("Save failed", str(exc))
            return False

    def _autosave_dataset(self) -> bool:
        return self._persist_dataset()

    def start_new_dataset(self) -> None:
        if self.dataset.vna_scans or self._dirty:
            ok = messagebox.askyesno(
                "Start New Dataset",
                "Start a new empty dataset?\nUnsaved changes in the current dataset will be lost.",
            )
            if not ok:
                return

        proposed_name = simpledialog.askstring(
            "New Dataset Prefix",
            "Enter the prefix to use for the new dataset folder and pickle filename:",
            parent=self.root,
        )
        if proposed_name is None:
            return
        cleaned_name = _safe_name(proposed_name)
        if not cleaned_name:
            messagebox.showwarning("Invalid prefix", "Please enter a non-empty dataset prefix.")
            return

        for closer in (
            getattr(self, "_synth_close", None),
            getattr(self, "_close_baseline_window", None),
            getattr(self, "_interp_close", None),
            getattr(self, "_norm_close", None),
            getattr(self, "_gauss_close", None),
            getattr(self, "_dsdf_close", None),
            getattr(self, "_unwrap_close", None),
            getattr(self, "_phase2_close", None),
            getattr(self, "_phase3_close", None),
            getattr(self, "_res_close", None),
        ):
            if callable(closer):
                closer()

        created_at = datetime.now().isoformat(timespec="seconds")
        self.dataset = Dataset(
            source_file=str(DEFAULT_DATASET_FILE.resolve()),
            dataset_name=cleaned_name,
            created_at=created_at,
        )
        self.dataset_path = _dataset_pickle_path(self.dataset).resolve()
        self.dataset.source_file = str(self.dataset_path)
        _write_app_state(self.dataset_path)
        self._reload_transcript_ui()
        self._mark_clean()
        self._refresh_status()
        self._log(f"Started new empty dataset: {cleaned_name}")

    def _selected_scans_have_attached_filter(self) -> bool:
        scans = self._selected_scans()
        if not scans:
            return False
        for scan in scans:
            bf = scan.baseline_filter
            if not isinstance(bf, dict):
                return False
            amp, phase = _read_polar_series(
                bf,
                amplitude_key="filtered_amp",
                phase_key="filtered_phase_deg",
            )
            if amp.ndim != 1 or phase.ndim != 1 or amp.shape != phase.shape:
                return False
            if amp.size == 0:
                return False
        return True

    def _configure_action_button(
        self,
        button: Optional[tk.Button],
        *,
        available: bool,
        done_count: int = 0,
        total_count: int = 0,
    ) -> None:
        if button is None:
            return
        if not available:
            button.configure(state="disabled", bg="light grey", activebackground="light grey")
            return
        if total_count > 0 and done_count >= total_count:
            bg = "medium sea green"
        elif done_count > 0:
            bg = "light green"
        else:
            bg = self._default_button_bg
        button.configure(state="normal", bg=bg, activebackground=bg)

    def _selected_progress_counts(self, done_check) -> tuple[list[VNAScan], int]:
        scans = self._selected_scans()
        return scans, sum(1 for scan in scans if done_check(scan))

    def _baseline_target_scans(self) -> list[VNAScan]:
        scans = self._selected_scans()
        return scans if scans else list(self.dataset.vna_scans)

    def _has_valid_phase2_output(self, scan: VNAScan) -> bool:
        phase2 = scan.candidate_resonators.get("phase_correction_2")
        if not isinstance(phase2, dict):
            return False
        amp, phase = _read_polar_series(
            phase2,
            amplitude_key="corrected_amp",
            phase_key="corrected_phase_deg",
        )
        return amp.shape == scan.freq.shape and phase.shape == scan.freq.shape

    def _has_valid_phase3_output(self, scan: VNAScan) -> bool:
        phase3 = scan.candidate_resonators.get("phase_correction_3")
        if not isinstance(phase3, dict):
            return False
        amp, phase = _read_polar_series(
            phase3,
            amplitude_key="corrected_amp",
            phase_key="corrected_phase_deg",
        )
        return amp.shape == scan.freq.shape and phase.shape == scan.freq.shape

    def _has_valid_baseline_filter_output(self, scan: VNAScan) -> bool:
        bf = scan.baseline_filter
        if not isinstance(bf, dict):
            return False
        amp, phase = _read_polar_series(
            bf,
            amplitude_key="filtered_amp",
            phase_key="filtered_phase_deg",
        )
        if amp.ndim != 1 or phase.ndim != 1 or amp.shape != phase.shape or amp.size == 0:
            return False
        keep = np.asarray(bf.get("retained_mask", np.array([])), dtype=bool)
        if keep.ndim != 1 or keep.size != scan.freq.size:
            return False
        return int(np.count_nonzero(keep)) == int(amp.size)

    def _has_valid_interp_output(self, scan: VNAScan) -> bool:
        bf = scan.baseline_filter
        if not isinstance(bf, dict):
            return False
        interp = bf.get("interp_smooth")
        if not isinstance(interp, dict):
            return False
        amp, phase = _read_polar_series(
            interp,
            amplitude_key="interp_amp",
            phase_key="interp_phase",
        )
        return amp.shape == scan.freq.shape and phase.shape == scan.freq.shape

    def _has_valid_normalized_output(self, scan: VNAScan) -> bool:
        bf = scan.baseline_filter
        if not isinstance(bf, dict):
            return False
        norm = bf.get("normalized")
        if not isinstance(norm, dict):
            return False
        amp, phase = _read_polar_series(
            norm,
            amplitude_key="norm_amp",
            phase_key="norm_phase_deg_unwrapped",
        )
        return amp.shape == scan.freq.shape and phase.shape == scan.freq.shape

    @staticmethod
    def _has_valid_candidate_attachment(scan: VNAScan, key: str) -> bool:
        payload = scan.candidate_resonators.get(key)
        return isinstance(payload, dict) and len(payload) > 0

    def _update_interp_button_state(self) -> None:
        scans, done_count = self._selected_progress_counts(self._has_valid_interp_output)
        self._configure_action_button(
            self.interp_button,
            available=bool(scans) and all(self._has_valid_baseline_filter_output(scan) for scan in scans),
            done_count=done_count,
            total_count=len(scans),
        )

    def _selected_scans_have_attached_interp_data(self) -> bool:
        scans = self._selected_scans()
        if not scans:
            return False
        for scan in scans:
            bf = scan.baseline_filter
            if not isinstance(bf, dict):
                return False
            interp = bf.get("interp_smooth")
            if not isinstance(interp, dict):
                return False
            amp, phase = _read_polar_series(
                interp,
                amplitude_key="interp_amp",
                phase_key="interp_phase",
            )
            if amp.ndim != 1 or phase.ndim != 1 or amp.shape != phase.shape:
                return False
            if amp.size == 0:
                return False
            if amp.size != scan.freq.size:
                return False
        return True

    def _update_norm_button_state(self) -> None:
        scans, done_count = self._selected_progress_counts(self._has_valid_normalized_output)
        self._configure_action_button(
            self.norm_button,
            available=bool(scans) and all(self._has_valid_interp_output(scan) for scan in scans),
            done_count=done_count,
            total_count=len(scans),
        )

    def _selected_scans_have_attached_normalized(self) -> bool:
        scans = self._selected_scans()
        if not scans:
            return False
        for scan in scans:
            bf = scan.baseline_filter
            if not isinstance(bf, dict):
                return False
            norm = bf.get("normalized")
            if not isinstance(norm, dict):
                return False
            amp, phase = _read_polar_series(
                norm,
                amplitude_key="norm_amp",
                phase_key="norm_phase_deg_unwrapped",
            )
            if amp.ndim != 1 or phase.ndim != 1 or amp.shape != phase.shape:
                return False
            if amp.size != scan.freq.size:
                return False
        return True

    def _update_res_button_state(self) -> None:
        scans, done_count = self._selected_progress_counts(
            lambda scan: self._has_valid_candidate_attachment(scan, "resonance_selection_view")
        )
        self._configure_action_button(
            self.res_button,
            available=bool(scans) and all(self._has_valid_normalized_output(scan) for scan in scans),
            done_count=done_count,
            total_count=len(scans),
        )

    def _update_gauss_button_state(self) -> None:
        scans, done_count = self._selected_progress_counts(
            lambda scan: self._has_valid_candidate_attachment(scan, "gaussian_convolution")
        )
        self._configure_action_button(
            self.gauss_button,
            available=bool(scans) and all(self._has_valid_normalized_output(scan) for scan in scans),
            done_count=done_count,
            total_count=len(scans),
        )

    def _update_dsdf_button_state(self) -> None:
        scans, done_count = self._selected_progress_counts(
            lambda scan: self._has_valid_candidate_attachment(scan, "dsdf_gaussian_convolution")
        )
        self._configure_action_button(
            self.dsdf_button,
            available=bool(scans) and all(self._has_valid_normalized_output(scan) for scan in scans),
            done_count=done_count,
            total_count=len(scans),
        )

    def _update_unwrap_button_state(self) -> None:
        scans, done_count = self._selected_progress_counts(lambda scan: scan.has_dewrapped_phase())
        self._configure_action_button(
            self.unwrap_button,
            available=bool(scans),
            done_count=done_count,
            total_count=len(scans),
        )

    def _update_phase2_button_state(self) -> None:
        scans, done_count = self._selected_progress_counts(self._has_valid_phase2_output)
        self._configure_action_button(
            self.phase2_button,
            available=bool(scans) and all(scan.has_dewrapped_phase() for scan in scans),
            done_count=done_count,
            total_count=len(scans),
        )

    def _update_phase3_button_state(self) -> None:
        scans, done_count = self._selected_progress_counts(self._has_valid_phase3_output)
        self._configure_action_button(
            self.phase3_button,
            available=bool(scans) and all(self._has_valid_phase2_output(scan) for scan in scans),
            done_count=done_count,
            total_count=len(scans),
        )

    def _update_baseline_button_state(self) -> None:
        scans = self._baseline_target_scans()
        done_count = sum(1 for scan in scans if self._has_valid_baseline_filter_output(scan))
        self._configure_action_button(
            self.baseline_button,
            available=bool(scans) and all(self._has_valid_phase3_output(scan) for scan in scans),
            done_count=done_count,
            total_count=len(scans),
        )

    def _update_select_scans_button_state(self) -> None:
        total_count = len(self.dataset.vna_scans)
        done_count = len(self._selected_scans())
        self._configure_action_button(
            self.select_scans_button,
            available=total_count > 0,
            done_count=done_count,
            total_count=total_count,
        )

    def save_dataset(self) -> None:
        if not self._has_data_to_save():
            self._log("Save skipped: no data to save.")
            self._update_save_button_state()
            return
        if not self.dataset.dataset_name:
            proposed_name = simpledialog.askstring(
                "Name dataset",
                "Enter a dataset name:",
                parent=self.root,
            )
            if proposed_name is None:
                return
            cleaned_name = _safe_name(proposed_name)
            if not cleaned_name:
                messagebox.showwarning("Invalid name", "Please enter a non-empty dataset name.")
                return
            self.dataset.dataset_name = cleaned_name
            if not self.dataset.created_at:
                self.dataset.created_at = datetime.now().isoformat(timespec="seconds")

        self._persist_dataset()

    def load_different_dataset(self) -> None:
        path_text = filedialog.askopenfilename(
            title="Select dataset file",
            initialdir=str(DATASETS_DIR.resolve()),
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
        )
        if not path_text:
            return

        new_path = Path(path_text)
        try:
            self.dataset = _load_dataset(new_path)
            self.dataset_path = new_path.resolve()
            _write_app_state(self.dataset_path)
            self._reload_transcript_ui()
            self._mark_clean()
            self._refresh_status()
            self._log(f"Dataset loaded: {self.dataset_path}")
            messagebox.showinfo("Dataset loaded", f"Loaded dataset:\n{self.dataset_path}")
        except Exception as exc:
            self._log(f"Load failed: {exc}")
            messagebox.showerror("Load failed", str(exc))

    def load_vna_scan(self) -> None:
        path_texts = filedialog.askopenfilenames(
            title="Select VNA scan file(s)",
            filetypes=[
                ("Supported VNA files", "*.npy *.txt *.dat *.csv"),
                ("NumPy files", "*.npy"),
                ("Text files", "*.txt *.dat *.csv"),
                ("All files", "*.*"),
            ],
        )
        if not path_texts:
            return

        added_count = 0
        failed: List[str] = []
        warnings: List[str] = []
        for path_text in path_texts:
            path = Path(path_text)
            try:
                scan, warning = _load_vna_file(path)
                self.dataset.vna_scans.append(scan)
                self.dataset.selected_scan_keys.append(self._scan_key(scan))
                self.dataset.processing_history.append(
                    _make_event(
                        "add_vna_scan_to_dataset",
                        {"filename": scan.filename, "loaded_at": scan.loaded_at},
                    )
                )
                self._mark_dirty()
                added_count += 1
                if warning:
                    warnings.append(warning)
            except Exception as exc:
                failed.append(f"{path.name}: {exc}")

        self._refresh_status()
        self._log(f"VNA load result: added={added_count}, failed={len(failed)}")
        for warning in warnings:
            self._log(f"VNA load warning: {warning}")
        if added_count > 0:
            self._autosave_dataset()

        if added_count > 0 and not failed:
            if warnings:
                messagebox.showwarning(
                    "VNA scans loaded with assumptions",
                    f"Added {added_count} scan(s). Dataset saved automatically.\n\n"
                    + "\n".join(warnings[:8]),
                )
                return
            messagebox.showinfo(
                "VNA scans loaded",
                f"Added {added_count} scan(s). Dataset saved automatically.",
            )
            return

        if added_count > 0 and failed:
            detail_lines = [f"Added {added_count} scan(s), {len(failed)} failed."]
            if warnings:
                detail_lines.extend(["", "Assumption warnings:"])
                detail_lines.extend(warnings[:8])
            detail_lines.extend(["", "Failures:"])
            detail_lines.extend(failed[:8])
            messagebox.showwarning("VNA scan load partial", "\n".join(detail_lines))
            return

        messagebox.showerror(
            "VNA scan load failed",
            "No files were loaded.\n\n" + "\n".join(failed[:8]),
        )

    def remove_vna_scans(self) -> None:
        if not self.dataset.vna_scans:
            messagebox.showwarning("No data", "No VNA scans are loaded in this dataset.")
            return

        selector = tk.Toplevel(self.root)
        selector.title("Remove VNA Scan(s)")
        selector.geometry("820x420")
        selector.transient(self.root)
        selector.grab_set()

        tk.Label(selector, text="Select scan(s) to remove from this dataset:").pack(
            anchor="w", padx=10, pady=(10, 4)
        )
        listbox = tk.Listbox(selector, width=130, height=16, selectmode=tk.MULTIPLE)
        listbox.pack(fill="both", expand=True, padx=10, pady=4)

        for idx, scan in enumerate(self.dataset.vna_scans):
            label = f"{idx:03d} | {Path(scan.filename).name} | loaded {scan.loaded_at}"
            listbox.insert(tk.END, label)

        def do_remove() -> None:
            indices = sorted(listbox.curselection())
            if not indices:
                selector.destroy()
                return
            names = [Path(self.dataset.vna_scans[i].filename).name for i in indices]
            ok = messagebox.askyesno(
                "Confirm Remove",
                f"Remove {len(indices)} scan(s) from dataset?\n\n"
                + "\n".join(names[:8])
                + ("\n..." if len(names) > 8 else ""),
                parent=selector,
            )
            if not ok:
                return

            removed = []
            for i in reversed(indices):
                scan = self.dataset.vna_scans.pop(i)
                removed.append(scan)

            self.dataset.selected_scan_keys = [
                self._scan_key(scan)
                for scan in self.dataset.vna_scans
                if self._scan_key(scan) in set(self.dataset.selected_scan_keys)
            ]
            self.dataset.processing_history.append(
                _make_event(
                    "remove_vna_scans_from_dataset",
                    {
                        "removed_count": len(removed),
                        "removed_files": [s.filename for s in removed],
                    },
                )
            )
            self._mark_dirty()
            self._refresh_status()
            self._log(f"Removed {len(removed)} VNA scan(s) from dataset.")
            self._autosave_dataset()
            selector.destroy()

        btns = tk.Frame(selector)
        btns.pack(fill="x", padx=10, pady=(4, 10))
        tk.Button(btns, text="Cancel", command=selector.destroy).pack(side="right")
        tk.Button(btns, text="Remove Selected", command=do_remove).pack(side="right", padx=(0, 8))

    def group_selected_scans_for_plotting(self) -> None:
        scans = self._selected_scans()
        if not scans:
            messagebox.showwarning(
                "No selection",
                "No scans selected for analysis.\nUse 'Select Scans for Analysis' first.",
            )
            return

        selector = tk.Toplevel(self.root)
        selector.title("Group Selected Scans")
        selector.geometry("900x460")
        selector.transient(self.root)
        selector.grab_set()

        existing_groups = sorted(
            {int(scan.plot_group) for scan in self.dataset.vna_scans if scan.plot_group is not None}
        )
        prompt = (
            "Choose the subset of currently selected analysis scans to join into one plot group.\n"
            "Each scan can belong to only one group. Enter a positive integer to assign, or leave blank to clear."
        )
        if existing_groups:
            prompt += "\nExisting groups: " + ", ".join(str(g) for g in existing_groups)
        tk.Label(selector, text=prompt, anchor="w", justify="left", wraplength=860).pack(
            anchor="w", padx=10, pady=(10, 6)
        )

        group_frame = tk.Frame(selector)
        group_frame.pack(fill="x", padx=10, pady=(0, 6))
        tk.Label(group_frame, text="Group number:").pack(side="left")
        group_entry = tk.Entry(group_frame, width=12)
        group_entry.pack(side="left", padx=(6, 12))
        tk.Label(group_frame, text="Blank clears group for the chosen subset.").pack(side="left")

        listbox = tk.Listbox(selector, width=160, height=16, selectmode=tk.MULTIPLE)
        listbox.pack(fill="both", expand=True, padx=10, pady=4)
        for idx, scan in enumerate(scans):
            file_timestamp = str(getattr(scan, "file_timestamp", "")).strip() or "unknown"
            group_text = f"group {int(scan.plot_group)}" if scan.plot_group is not None else "no group"
            label = (
                f"{idx:03d} | {Path(scan.filename).name} | "
                f"file {file_timestamp} | loaded {scan.loaded_at} | {group_text}"
            )
            listbox.insert(tk.END, label)
            listbox.selection_set(idx)

        def do_apply() -> None:
            indices = sorted(listbox.curselection())
            if not indices:
                messagebox.showwarning("No subset selected", "Select at least one scan in the grouping window.", parent=selector)
                return

            text = group_entry.get().strip()
            new_group: Optional[int] = None
            if text:
                try:
                    new_group = int(text)
                except ValueError:
                    messagebox.showwarning("Invalid group", "Enter a positive integer group number.", parent=selector)
                    return
                if new_group <= 0:
                    messagebox.showwarning("Invalid group", "Enter a positive integer group number.", parent=selector)
                    return

            chosen_scans = [scans[i] for i in indices]
            changed = 0
            for scan in chosen_scans:
                if scan.plot_group != new_group:
                    scan.plot_group = new_group
                    changed += 1

            if changed == 0:
                self._log("Grouping unchanged for selected scan subset.")
                selector.destroy()
                return

            self.dataset.processing_history.append(
                _make_event(
                    "group_selected_scans_for_plotting",
                    {
                        "selected_count": len(chosen_scans),
                        "plot_group": new_group,
                        "filenames": [scan.filename for scan in chosen_scans],
                    },
                )
            )
            self._mark_dirty()
            self._refresh_status()
            self._autosave_dataset()
            if new_group is None:
                self._log(f"Cleared plot groups for {changed} selected scan(s).")
                messagebox.showinfo("Plot groups updated", f"Cleared plot groups for {changed} selected scan(s).", parent=selector)
            else:
                self._log(f"Assigned plot group {new_group} to {changed} selected scan(s).")
                messagebox.showinfo(
                    "Plot groups updated",
                    f"Assigned plot group {new_group} to {changed} selected scan(s).",
                    parent=selector,
                )
            selector.destroy()

        btns = tk.Frame(selector)
        btns.pack(fill="x", padx=10, pady=(4, 10))
        tk.Button(btns, text="Select All", command=lambda: listbox.select_set(0, tk.END)).pack(side="left")
        tk.Button(btns, text="Clear Selection", command=lambda: listbox.selection_clear(0, tk.END)).pack(
            side="left", padx=(6, 0)
        )
        tk.Button(btns, text="Cancel", command=selector.destroy).pack(side="right")
        tk.Button(btns, text="Apply Group", command=do_apply).pack(side="right", padx=(0, 8))

    def clear_selected_scan_attachments(self) -> None:
        scans = self._selected_scans()
        if not scans:
            messagebox.showwarning(
                "No selection",
                "No scans selected for analysis.\nUse 'Select Scans for Analysis' first.",
            )
            return

        selector = tk.Toplevel(self.root)
        selector.title("Clear Attachments")
        selector.geometry("920x480")
        selector.transient(self.root)
        selector.grab_set()

        prompt = (
            "Choose the subset of selected analysis scans to reset.\n"
            "This returns each chosen scan to its original loaded state, while keeping plot group membership."
        )
        tk.Label(selector, text=prompt, anchor="w", justify="left", wraplength=880).pack(
            anchor="w", padx=10, pady=(10, 6)
        )

        listbox = tk.Listbox(selector, width=145, height=18, selectmode=tk.MULTIPLE)
        listbox.pack(fill="both", expand=True, padx=10, pady=4)
        for idx, scan in enumerate(scans):
            group_text = f"group {int(scan.plot_group)}" if scan.plot_group is not None else "no group"
            label = f"{idx:03d} | {Path(scan.filename).name} | loaded {scan.loaded_at} | {group_text}"
            listbox.insert(tk.END, label)

        def do_clear() -> None:
            indices = sorted(listbox.curselection())
            if not indices:
                messagebox.showwarning(
                    "No subset selected",
                    "Select at least one scan in the clear-attachments window.",
                    parent=selector,
                )
                return

            chosen_scans = [scans[i] for i in indices]
            names = [Path(scan.filename).name for scan in chosen_scans]
            ok = messagebox.askyesno(
                "Confirm Clear",
                f"Clear attachments for {len(chosen_scans)} selected scan(s)?\n\n"
                + "\n".join(names[:8])
                + ("\n..." if len(names) > 8 else ""),
                parent=selector,
            )
            if not ok:
                return

            cleared = 0
            for scan in chosen_scans:
                load_events = [
                    event
                    for event in scan.processing_history
                    if str(event.get("action", "")).startswith("load_vna_")
                ]
                scan.s21_phase_deg_unwrapped = None
                scan.baseline_filter = {}
                scan.candidate_resonators = {}
                scan.processing_history = load_events[:1]
                cleared += 1

            self.dataset.processing_history.append(
                _make_event(
                    "clear_selected_scan_attachments",
                    {
                        "selected_count": len(chosen_scans),
                        "filenames": [scan.filename for scan in chosen_scans],
                    },
                )
            )
            self._mark_dirty()
            self._refresh_status()
            self._autosave_dataset()
            self._log(f"Cleared attachments for {cleared} selected scan(s).")
            messagebox.showinfo(
                "Attachments cleared",
                f"Cleared attachments for {cleared} selected scan(s).\nPlot groups were preserved.",
                parent=selector,
            )
            selector.destroy()

        btns = tk.Frame(selector)
        btns.pack(fill="x", padx=10, pady=(4, 10))
        tk.Button(btns, text="Select All", command=lambda: listbox.select_set(0, tk.END)).pack(side="left")
        tk.Button(btns, text="Clear Selection", command=lambda: listbox.selection_clear(0, tk.END)).pack(
            side="left", padx=(6, 0)
        )
        tk.Button(btns, text="Cancel", command=selector.destroy).pack(side="right")
        tk.Button(btns, text="Clear Attachments", command=do_clear).pack(side="right", padx=(0, 8))

    def open_resonance_sheet_loader(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Load Resonators From Sheet")
        dialog.geometry("860x190")
        dialog.transient(self.root)
        dialog.grab_set()

        path_var = tk.StringVar()

        def browse() -> None:
            chosen = filedialog.askopenfilename(
                title="Select resonance spreadsheet",
                filetypes=[("Excel files", "*.xlsx *.xlsm"), ("All files", "*.*")],
            )
            if chosen:
                path_var.set(chosen)

        def run() -> None:
            sheet_path = Path(path_var.get().strip())
            if not sheet_path.exists():
                messagebox.showwarning("Missing file", "Select a valid spreadsheet file first.", parent=dialog)
                return
            try:
                loaded_count = self._load_resonances_from_sheet(sheet_path)
            except Exception as exc:
                self._log(f"Load resonators from sheet failed: {exc}")
                messagebox.showerror("Load failed", str(exc), parent=dialog)
                return
            messagebox.showinfo(
                "Resonators loaded",
                f"Loaded {loaded_count} resonator assignment(s) from:\n{sheet_path}",
                parent=dialog,
            )
            dialog.destroy()

        top = tk.Frame(dialog, padx=10, pady=10)
        top.pack(fill="both", expand=True)
        tk.Label(
            top,
            text="Select an Excel sheet where row 1 contains resonator numbers, row 2 contains headings, column A from row 3 down contains a VNA filename or group name, and the remaining cells contain resonance frequencies. The resonator assignments will be attached to the matching VNA scans and saved into the dataset.",
            anchor="w",
            justify="left",
            wraplength=820,
        ).pack(anchor="w", pady=(0, 8))

        row = tk.Frame(top)
        row.pack(fill="x", pady=(0, 8))
        tk.Entry(row, textvariable=path_var, width=95).pack(side="left", fill="x", expand=True)
        tk.Button(row, text="Browse", width=10, command=browse).pack(side="left", padx=(8, 0))

        btns = tk.Frame(top)
        btns.pack(fill="x", pady=(12, 0))
        tk.Button(btns, text="Cancel", width=12, command=dialog.destroy).pack(side="right")
        tk.Button(btns, text="Load Resonators", width=14, command=run).pack(side="right", padx=(0, 8))

    def open_attached_resonance_plotter(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Plot Attached Resonators")
        dialog.geometry("720x170")
        dialog.transient(self.root)
        dialog.grab_set()

        data_mode_var = tk.StringVar(value="normalized")

        def run() -> None:
            try:
                saved = self._plot_attached_resonances(data_mode=data_mode_var.get())
            except Exception as exc:
                self._log(f"Plot attached resonators failed: {exc}")
                messagebox.showerror("Plot failed", str(exc), parent=dialog)
                return
            if not saved:
                messagebox.showwarning("No plots saved", "No plot files were generated.", parent=dialog)
                return
            out_dir = saved[0].parent
            self._log(f"Plotted attached resonators into {out_dir}")
            messagebox.showinfo(
                "Plots saved",
                f"Saved {len(saved)} plot file(s) to:\n{out_dir}",
                parent=dialog,
            )
            dialog.destroy()

        top = tk.Frame(dialog, padx=10, pady=10)
        top.pack(fill="both", expand=True)
        tk.Label(
            top,
            text="Plot the resonator assignments currently attached to the loaded VNA scans.",
            anchor="w",
            justify="left",
            wraplength=680,
        ).pack(anchor="w", pady=(0, 8))

        mode_row = tk.Frame(top)
        mode_row.pack(fill="x", pady=(0, 8))
        tk.Label(mode_row, text="Plot data:").pack(side="left")
        tk.Radiobutton(
            mode_row,
            text="Baseline normalized (default)",
            variable=data_mode_var,
            value="normalized",
        ).pack(side="left", padx=(8, 12))
        tk.Radiobutton(
            mode_row,
            text="Raw VNA data",
            variable=data_mode_var,
            value="raw",
        ).pack(side="left")

        btns = tk.Frame(top)
        btns.pack(fill="x", pady=(12, 0))
        tk.Button(btns, text="Cancel", width=12, command=dialog.destroy).pack(side="right")
        tk.Button(btns, text="Generate Plots", width=14, command=run).pack(side="right", padx=(0, 8))

    @staticmethod
    def _coerce_frequency_to_scan_hz(value: float, scan: VNAScan) -> Optional[float]:
        freq = np.asarray(scan.freq, dtype=float)
        if freq.size == 0 or not np.isfinite(value):
            return None
        lo = float(np.min(freq))
        hi = float(np.max(freq))
        candidates = [float(value), float(value) * 1e9, float(value) * 1e6, float(value) * 1e3]
        for cand in candidates:
            if lo <= cand <= hi:
                return cand
        return None

    def _find_scan_for_sheet_identifier(self, identifier: str, target_value: float) -> tuple[VNAScan, float]:
        text = identifier.strip()
        lower = text.lower()
        scans = list(self.dataset.vna_scans)

        if lower.startswith("group "):
            try:
                group_num = int(lower.split()[1])
            except Exception as exc:
                raise ValueError(f"Invalid group name: {identifier}") from exc
            grouped = [scan for scan in scans if scan.plot_group == group_num]
            if not grouped:
                raise ValueError(f"No scans found for {identifier}.")
            for scan in grouped:
                target_hz = self._coerce_frequency_to_scan_hz(target_value, scan)
                if target_hz is not None:
                    return scan, target_hz
            raise ValueError(f"No scan in {identifier} contains frequency value {target_value}.")

        for scan in scans:
            name = Path(scan.filename).name.lower()
            stem = Path(scan.filename).stem.lower()
            if lower == name or lower == stem:
                target_hz = self._coerce_frequency_to_scan_hz(target_value, scan)
                if target_hz is None:
                    raise ValueError(f"Frequency value {target_value} is outside {Path(scan.filename).name}.")
                return scan, target_hz
        raise ValueError(f"No scan found matching filename {identifier}.")

    @staticmethod
    def _resonance_plot_window_hz(scan: VNAScan, target_hz: float) -> tuple[float, float]:
        freq = np.sort(np.asarray(scan.freq, dtype=float))
        if freq.size == 0:
            return target_hz, target_hz
        span = float(freq[-1] - freq[0])
        diffs = np.diff(freq)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        step = float(np.median(diffs)) if diffs.size else max(span / max(freq.size - 1, 1), 1.0)
        half_width = max(250.0 * step, 0.002 * span)
        half_width = min(half_width, 0.05 * span if span > 0 else half_width)
        lo = max(float(freq[0]), target_hz - half_width)
        hi = min(float(freq[-1]), target_hz + half_width)
        if hi <= lo:
            half_width = max(1000.0 * step, 1.0)
            lo = max(float(freq[0]), target_hz - half_width)
            hi = min(float(freq[-1]), target_hz + half_width)
        return lo, hi

    @staticmethod
    def _interpolate_y(freq_hz: np.ndarray, y: np.ndarray, target_hz: float) -> float:
        x = np.asarray(freq_hz, dtype=float)
        vals = np.asarray(y, dtype=float)
        order = np.argsort(x)
        x = x[order]
        vals = vals[order]
        return float(np.interp(target_hz, x, vals))

    @staticmethod
    def _sheet_resonator_label(value: object) -> str:
        if value is None:
            return ""
        try:
            numeric = float(value)
            if np.isfinite(numeric) and numeric.is_integer():
                return str(int(numeric))
        except Exception:
            pass
        return str(value).strip()

    @staticmethod
    def _sheet_resonance_attachment(scan: VNAScan) -> dict[str, object]:
        payload = scan.candidate_resonators.get("sheet_resonances")
        if not isinstance(payload, dict):
            payload = {"assignments": {}}
            scan.candidate_resonators["sheet_resonances"] = payload
        assignments = payload.get("assignments")
        if not isinstance(assignments, dict):
            payload["assignments"] = {}
        return payload

    def _load_resonances_from_sheet(self, sheet_path: Path) -> int:
        wb = load_workbook(sheet_path, data_only=True, read_only=True)
        try:
            ws = wb.active

            column_headers: dict[int, str] = {}
            for col_idx in range(2, ws.max_column + 1):
                cell = ws.cell(row=1, column=col_idx).value
                header = self._sheet_resonator_label(cell)
                column_headers[col_idx] = header

            row_records: list[dict] = []
            warnings: list[str] = []
            assignment_count = 0
            for row_idx, row in enumerate(ws.iter_rows(min_row=3, values_only=True), start=3):
                if not row:
                    continue
                identifier_raw = row[0]
                if identifier_raw is None or str(identifier_raw).strip() == "":
                    continue
                identifier = str(identifier_raw).strip()
                freq_entries: dict[int, dict] = {}
                for col_offset, cell in enumerate(row[1:], start=2):
                    resonator_number = column_headers.get(col_offset, "")
                    if cell is None or str(cell).strip() == "":
                        freq_entries[col_offset] = None
                        continue
                    try:
                        target_value = float(cell)
                    except Exception:
                        warnings.append(f"Row {row_idx}, column {col_offset}: skipped non-numeric frequency cell {cell!r}.")
                        freq_entries[col_offset] = None
                        continue
                    try:
                        scan, target_hz = self._find_scan_for_sheet_identifier(identifier, target_value)
                    except Exception as exc:
                        warnings.append(f"Row {row_idx}, column {col_offset}: {exc}")
                        freq_entries[col_offset] = None
                        continue
                    if resonator_number:
                        payload = self._sheet_resonance_attachment(scan)
                        assignments = payload["assignments"]
                        assignments[str(resonator_number)] = {
                            "frequency_hz": float(target_hz),
                            "sheet_path": str(sheet_path),
                            "sheet_name": str(ws.title),
                            "row": int(row_idx),
                            "column": int(col_offset),
                            "identifier": identifier,
                        }
                        assignment_count += 1
                    else:
                        warnings.append(
                            f"Row 1, column {col_offset}: blank resonator number header; frequency was plotted but not attached."
                        )
                    freq_entries[col_offset] = {
                        "scan": scan,
                        "target_hz": target_hz,
                        "target_value": target_value,
                        "resonator_number": resonator_number,
                    }
                row_records.append(
                    {
                        "row_idx": row_idx,
                        "identifier": identifier,
                        "entries": freq_entries,
                    }
                )

            data_columns = [col for col in range(2, ws.max_column + 1)]
        finally:
            wb.close()

        if not row_records or not data_columns:
            raise ValueError("No loadable resonance entries were found in the selected spreadsheet.")

        self.dataset.processing_history.append(
            _make_event(
                "load_resonances_from_sheet",
                {
                    "sheet_path": str(sheet_path),
                    "sheet_name": str(ws.title),
                    "assignment_count": int(assignment_count),
                },
            )
        )
        self._mark_dirty()
        self._autosave_dataset()
        for warning in warnings[:20]:
            self._log(f"Sheet resonance load warning: {warning}")
        self._log(f"Loaded {assignment_count} resonator assignment(s) from {sheet_path}")
        return assignment_count

    def _collect_attached_resonance_rows(self) -> tuple[list[dict], dict[int, str], list[int], list[str], str]:
        grouped_rows: dict[int, dict] = {}
        column_headers: dict[int, str] = {}
        warnings: list[str] = []
        sheet_labels: set[str] = set()

        for scan in self.dataset.vna_scans:
            payload = scan.candidate_resonators.get("sheet_resonances")
            if not isinstance(payload, dict):
                continue
            assignments = payload.get("assignments")
            if not isinstance(assignments, dict):
                continue
            for resonator_number, record in assignments.items():
                if not isinstance(record, dict):
                    continue
                try:
                    row_idx = int(record.get("row"))
                    col_idx = int(record.get("column"))
                    target_hz = float(record.get("frequency_hz"))
                except Exception:
                    warnings.append(f"{Path(scan.filename).name}: skipped malformed attached resonator record.")
                    continue
                identifier = str(record.get("identifier") or Path(scan.filename).name)
                sheet_path = str(record.get("sheet_path") or "")
                sheet_name = str(record.get("sheet_name") or "")
                if sheet_path or sheet_name:
                    sheet_labels.add(f"{Path(sheet_path).name} | {sheet_name}".strip(" |"))
                column_headers[col_idx] = str(resonator_number).strip()
                row_record = grouped_rows.setdefault(
                    row_idx,
                    {
                        "row_idx": row_idx,
                        "identifier": identifier,
                        "entries": {},
                    },
                )
                row_record["entries"][col_idx] = {
                    "scan": scan,
                    "target_hz": target_hz,
                    "target_value": target_hz,
                    "resonator_number": str(resonator_number).strip(),
                }

        if not grouped_rows or not column_headers:
            raise ValueError("No attached resonator assignments were found on the loaded VNA scans.")

        row_records = [grouped_rows[idx] for idx in sorted(grouped_rows)]
        data_columns = sorted(column_headers)
        source_label = "; ".join(sorted(label for label in sheet_labels if label))
        return row_records, column_headers, data_columns, warnings, source_label

    def _plot_resonance_rows(
        self,
        *,
        row_records: list[dict],
        column_headers: dict[int, str],
        data_columns: list[int],
        data_mode: str,
        warnings: list[str],
        source_label: str,
    ) -> list[Path]:
        if not row_records or not data_columns:
            raise ValueError("No plottable resonance entries are available.")

        out_dir = _dataset_dir(self.dataset) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_sheet_resonance_plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[Path] = []

        column_chunks = [data_columns[i : i + 3] for i in range(0, len(data_columns), 3)]
        half_window_hz = 0.5e6

        for page_idx, page_columns in enumerate(column_chunks, start=1):
            nrows = len(row_records)
            ncols = len(page_columns)
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(5.2 * ncols, max(2.2 * nrows, 4.0)),
                squeeze=False,
                sharex="col",
            )

            column_ranges: dict[int, tuple[float, float]] = {}
            for col in page_columns:
                targets = []
                for row_record in row_records:
                    entry = row_record["entries"].get(col)
                    if isinstance(entry, dict):
                        targets.append(float(entry["target_hz"]))
                if targets:
                    column_ranges[col] = (min(targets) - half_window_hz, max(targets) + half_window_hz)
                else:
                    column_ranges[col] = (0.0, 1.0)

            for row_pos, row_record in enumerate(row_records):
                for col_pos, col in enumerate(page_columns):
                    ax = axes[row_pos][col_pos]
                    entry = row_record["entries"].get(col)
                    if entry is None:
                        ax.axis("off")
                        continue

                    scan = entry["scan"]
                    freq = np.asarray(scan.freq, dtype=float)
                    if data_mode == "normalized":
                        norm = scan.baseline_filter.get("normalized", {})
                        amp, _phase = _read_polar_series(
                            norm,
                            amplitude_key="norm_amp",
                            phase_key="norm_phase_deg_unwrapped",
                        )
                        if amp.shape == scan.freq.shape:
                            y = np.asarray(amp, dtype=float)
                            y_label = "Normalized |S21|"
                        else:
                            y = np.asarray(scan.amplitude(), dtype=float)
                            y_label = "|S21|"
                            warnings.append(
                                f"{Path(scan.filename).name}: normalized data missing; used raw amplitude instead."
                            )
                    else:
                        y = np.asarray(scan.amplitude(), dtype=float)
                        y_label = "|S21|"

                    order = np.argsort(freq)
                    freq_sorted = freq[order]
                    y_sorted = y[order]
                    col_lo, col_hi = column_ranges[col]
                    full_mask = (freq_sorted >= col_lo) & (freq_sorted <= col_hi)
                    if not np.any(full_mask):
                        ax.text(0.5, 0.5, "No data in column range", ha="center", va="center")
                        ax.axis("off")
                        continue

                    ax.plot(
                        freq_sorted[full_mask] / 1.0e9,
                        y_sorted[full_mask],
                        color="0.65",
                        linewidth=0.8,
                        zorder=1,
                    )

                    target_hz = float(entry["target_hz"])
                    local_lo = target_hz - half_window_hz
                    local_hi = target_hz + half_window_hz
                    local_mask = (freq_sorted >= local_lo) & (freq_sorted <= local_hi)
                    if np.any(local_mask):
                        ax.plot(
                            freq_sorted[local_mask] / 1.0e9,
                            y_sorted[local_mask],
                            color="tab:blue",
                            linewidth=2.2,
                            zorder=2,
                        )

                    show_labels = row_pos == 0 and col_pos == 0
                    cand = scan.candidate_resonators
                    gaussian_freqs = np.asarray(
                        cand.get("gaussian_convolution", {}).get("candidate_freq", np.array([])),
                        dtype=float,
                    )
                    dsdf_freqs = np.asarray(
                        cand.get("dsdf_gaussian_convolution", {}).get("candidate_freq", np.array([])),
                        dtype=float,
                    )
                    for freqs_hz, style, label in (
                        (
                            gaussian_freqs,
                            dict(
                                linestyle="none",
                                marker="o",
                                markersize=7,
                                markerfacecolor="none",
                                markeredgewidth=1.5,
                                color="green",
                            ),
                            "Gaussian candidates",
                        ),
                        (
                            dsdf_freqs,
                            dict(linestyle="none", marker="D", markersize=5, color="red"),
                            "dS21/df peaks",
                        ),
                    ):
                        in_win = freqs_hz[(freqs_hz >= col_lo) & (freqs_hz <= col_hi)]
                        if in_win.size == 0:
                            continue
                        y_pts = [self._interpolate_y(freq_sorted, y_sorted, float(f)) for f in in_win]
                        ax.plot(in_win / 1.0e9, y_pts, label=(label if show_labels else None), **style)

                    target_y = self._interpolate_y(freq_sorted, y_sorted, target_hz)
                    ax.plot(
                        [target_hz / 1.0e9],
                        [target_y],
                        linestyle="none",
                        marker="s",
                        markersize=7,
                        color="black",
                        label=("Spreadsheet frequency" if show_labels else None),
                    )
                    ax.set_xlim(col_lo / 1.0e9, col_hi / 1.0e9)
                    ax.grid(True, alpha=0.3)
                    ax.set_title(
                        f"{row_record['identifier']} | {Path(scan.filename).name}\n{target_hz / 1.0e9:.9g} GHz",
                        fontsize=8,
                    )
                    if col_pos == 0:
                        ax.set_ylabel(y_label)
                    if row_pos == nrows - 1:
                        ax.set_xlabel("Frequency (GHz)")
                    if show_labels:
                        ax.legend(loc="upper right", fontsize=8)

            for col_pos, col in enumerate(page_columns):
                resonator_number = column_headers.get(col, "")
                resonator_label = (
                    f"Resonator {resonator_number}"
                    if resonator_number
                    else f"Sheet column {col}"
                )
                axes[0][col_pos].set_title(
                    f"{resonator_label} | {column_ranges[col][0] / 1.0e9:.9g} to {column_ranges[col][1] / 1.0e9:.9g} GHz",
                    fontsize=9,
                )

            fig.suptitle(
                f"Attached Resonance Plots | mode={'baseline normalized' if data_mode == 'normalized' else 'raw'} | page {page_idx}"
                + (f" | {source_label}" if source_label else ""),
                fontsize=12,
            )
            fig.tight_layout()
            out_path = out_dir / f"resonance_sheet_page_{page_idx:02d}.pdf"
            fig.savefig(out_path)
            plt.close(fig)
            saved_paths.append(out_path)

        self.dataset.processing_history.append(
            _make_event(
                "plot_attached_resonances",
                {
                    "data_mode": data_mode,
                    "plot_count": int(
                        sum(sum(1 for entry in r["entries"].values() if isinstance(entry, dict)) for r in row_records)
                    ),
                    "page_count": len(saved_paths),
                    "output_dir": str(out_dir),
                    "source_label": source_label,
                },
            )
        )
        self._mark_dirty()
        self._autosave_dataset()
        for warning in warnings[:20]:
            self._log(f"Attached resonance plot warning: {warning}")
        return saved_paths

    def _plot_attached_resonances(self, *, data_mode: str) -> list[Path]:
        row_records, column_headers, data_columns, warnings, source_label = self._collect_attached_resonance_rows()
        return self._plot_resonance_rows(
            row_records=row_records,
            column_headers=column_headers,
            data_columns=data_columns,
            data_mode=data_mode,
            warnings=warnings,
            source_label=source_label,
        )

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
        self.attached_res_edit_window.title("Edit Attached Resonators")
        self.attached_res_edit_window.geometry("1320x900")
        self.attached_res_edit_window.protocol("WM_DELETE_WINDOW", self._attached_resonance_editor_exit)

        controls = tk.Frame(self.attached_res_edit_window, padx=8, pady=8)
        controls.pack(side="top", fill="x")
        self.attached_res_edit_status_var = tk.StringVar(
            value="Normalized selected scans will be plotted."
        )
        tk.Label(controls, textvariable=self.attached_res_edit_status_var, anchor="w").pack(
            side="left", fill="x", expand=True
        )
        number_controls = tk.Frame(controls)
        number_controls.pack(side="right")
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
            controls, text="Delete Selected", width=14, command=self._attached_resonance_editor_delete_selected
        ).pack(side="right", padx=(8, 0))
        tk.Button(
            controls, text="Reset View", width=12, command=self._attached_resonance_editor_reset_view
        ).pack(side="right", padx=(8, 0))
        self.attached_res_edit_exit_button = tk.Button(
            controls, text="Exit", width=12, command=self._attached_resonance_editor_exit
        )
        self.attached_res_edit_exit_button.pack(side="right", padx=(8, 0))

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
        self._attached_res_edit_snapshot = self._attached_resonance_editor_capture_snapshot()
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
        self.attached_res_edit_truncate_var = None
        self.attached_res_edit_truncate_threshold_var = None
        self.attached_res_edit_truncate_threshold_scale = None
        self.attached_res_edit_add_button = None
        self.attached_res_edit_exit_button = None
        self.attached_res_edit_ax = None
        self._attached_res_edit_points = []
        self._attached_res_edit_selected = None
        self._attached_res_edit_pending_add = False
        self._attached_res_edit_default_xlim = None
        self._attached_res_edit_missing_normalized_warned = None
        self._attached_res_edit_snapshot = None
        self._attached_res_edit_changed = False

    def _attached_resonance_editor_capture_snapshot(self) -> dict:
        scan_payloads: dict[str, object] = {}
        scan_history_lengths: dict[str, int] = {}
        for scan in self._selected_scans():
            key = self._scan_key(scan)
            payload = scan.candidate_resonators.get("sheet_resonances")
            scan_payloads[key] = copy.deepcopy(payload) if isinstance(payload, dict) else None
            scan_history_lengths[key] = len(scan.processing_history)
        return {
            "scan_payloads": scan_payloads,
            "scan_history_lengths": scan_history_lengths,
            "dataset_processing_history_len": len(self.dataset.processing_history),
            "dataset_transcript_len": len(self.dataset.transcript),
            "was_dirty": self._dirty,
        }

    def _attached_resonance_editor_restore_snapshot(self) -> None:
        snapshot = self._attached_res_edit_snapshot
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
        self._refresh_status()
        self._reload_transcript_ui()

    def _attached_resonance_editor_save(self) -> bool:
        if not self._attached_res_edit_changed:
            if self.attached_res_edit_status_var is not None:
                self.attached_res_edit_status_var.set("Attached resonator edits are already saved.")
            return True
        self._mark_dirty()
        if not self._autosave_dataset():
            return False
        self._attached_res_edit_changed = False
        self._attached_res_edit_snapshot = self._attached_resonance_editor_capture_snapshot()
        if self.attached_res_edit_status_var is not None:
            self.attached_res_edit_status_var.set("Attached resonator edits saved.")
        self._log("Saved attached resonator edits.")
        return True

    def _attached_resonance_editor_exit(self) -> None:
        if not self._attached_res_edit_changed:
            self._close_attached_resonance_editor()
            return
        dialog = messagebox.Message(
            parent=self.attached_res_edit_window,
            title="Save attached resonator edits?",
            message="Save attached resonator edits before exiting?",
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
            if self.attached_res_edit_status_var is not None:
                self.attached_res_edit_status_var.set(message)
            self.attached_res_edit_figure.clear()
            ax = self.attached_res_edit_figure.add_subplot(111)
            ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
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
        offset_by_scan_key, tick_info = self._attached_resonance_editor_offset_map(
            rows,
            self._attached_resonance_editor_curve_spacing(),
        )

        freq_min = min(float(row["freq"][0]) for row in rows) / 1.0e9
        freq_max = max(float(row["freq"][-1]) for row in rows) / 1.0e9
        x_pad = max((freq_max - freq_min) * 0.01, 1e-6)

        resonator_tracks: dict[str, list[tuple[float, float]]] = {}
        resonator_markers: list[dict] = []
        y_text_offset = 0.18
        for row in rows:
            scan = row["scan"]
            offset = float(offset_by_scan_key[str(row["scan_key"])])
            freq_ghz = np.asarray(row["freq"], dtype=float) / 1.0e9
            amp_display = self._attached_resonance_editor_display_amp(row["amp"])
            y = amp_display + offset
            ax.plot(freq_ghz, y, linewidth=1.0, color="tab:blue", alpha=0.8, zorder=1)

            for resonator in row["resonators"]:
                target_hz = float(resonator["target_hz"])
                target_ghz = target_hz / 1.0e9
                y_pt = self._interpolate_y(row["freq"], amp_display, target_hz) + offset
                scan_key = str(row["scan_key"])
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

        for resonator_number, points in resonator_tracks.items():
            if len(points) < 2:
                continue
            points = sorted(points, key=lambda item: item[1], reverse=True)
            ax.plot(
                [pt[0] for pt in points],
                [pt[1] for pt in points],
                linestyle=":",
                linewidth=1.0,
                color="tab:red",
                alpha=0.9,
                zorder=2,
            )

        for point in resonator_markers:
            is_selected = self._attached_res_edit_selected == (point["scan_key"], point["resonator_number"])
            ax.plot(
                [point["x_ghz"]],
                [point["y"]],
                linestyle="none",
                marker="o",
                markersize=(9 if is_selected else 6),
                markerfacecolor="none",
                markeredgecolor=("black" if is_selected else "tab:red"),
                markeredgewidth=1.5,
                zorder=4,
            )
            ax.text(
                point["x_ghz"],
                point["y"] - y_text_offset,
                point["resonator_number"],
                ha="center",
                va="top",
                fontsize=8,
                color=("black" if is_selected else "tab:red"),
                zorder=5,
            )

        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Normalized |S21| + vertical offset")
        ax.grid(True, alpha=0.3)
        ax.set_yticks([item[0] for item in tick_info])
        ax.set_yticklabels([item[1] for item in tick_info], fontsize=8)

        y_low = min(
            float(np.min(self._attached_resonance_editor_display_amp(row["amp"])))
            + float(offset_by_scan_key[str(row["scan_key"])])
            for row in rows
        )
        y_high = max(
            float(np.max(self._attached_resonance_editor_display_amp(row["amp"])))
            + float(offset_by_scan_key[str(row["scan_key"])])
            for row in rows
        )
        ax.set_ylim(y_low - 0.2, y_high + 0.2)
        self._attached_res_edit_default_xlim = (freq_min - 0.5 * x_pad, freq_max + 2.0 * x_pad)
        if prior_xlim is not None:
            ax.set_xlim(prior_xlim)
        else:
            ax.set_xlim(self._attached_res_edit_default_xlim)

        if self.attached_res_edit_status_var is not None:
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
        self._attached_resonance_editor_update_add_button()

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
        self._render_attached_resonance_editor()

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

    def _attached_resonance_editor_next_unused_number(self) -> str:
        used_numbers: set[int] = set()
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
                    used_numbers.add(parsed)
        candidate = 1
        while candidate in used_numbers:
            candidate += 1
        return str(candidate)

    def _attached_resonance_editor_reset_working_number(self) -> None:
        if self.attached_res_edit_working_number_var is None:
            return
        self.attached_res_edit_working_number_var.set(self._attached_resonance_editor_next_unused_number())

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
        self._render_attached_resonance_editor()

    def _attached_resonance_editor_on_click(self, event) -> None:
        if self.attached_res_edit_ax is None or event.inaxes != self.attached_res_edit_ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self.attached_res_edit_toolbar is not None:
            mode = getattr(self.attached_res_edit_toolbar, "mode", "")
            if str(mode).strip():
                if self.attached_res_edit_status_var is not None:
                    self.attached_res_edit_status_var.set(
                        f"Navigation mode active ({str(mode).strip()}). Finish pan/zoom to resume editing."
                    )
                return

        nearest = self._attached_resonance_editor_find_nearest_point(float(event.xdata), float(event.ydata))
        if self._attached_res_edit_pending_add:
            self._attached_resonance_editor_add_at_click(float(event.xdata), float(event.ydata))
            return

        if event.dblclick and self._attached_res_edit_selected is not None:
            self._attached_resonance_editor_move_selected(float(event.xdata), float(event.ydata))
            return

        if nearest is None:
            self._attached_res_edit_selected = None
            if self.attached_res_edit_status_var is not None:
                self.attached_res_edit_status_var.set("No resonator selected.")
            self._render_attached_resonance_editor()
            return

        self._attached_res_edit_selected = (nearest["scan_key"], nearest["resonator_number"])
        if self.attached_res_edit_status_var is not None:
            self.attached_res_edit_status_var.set(
                f"Selected resonator {nearest['resonator_number']} on {Path(nearest['scan'].filename).name}."
            )
        self._render_attached_resonance_editor()

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

    def _attached_resonance_editor_add_at_click(self, x_ghz: float, y_val: float) -> None:
        rows, _warnings = self._selected_scans_for_attached_resonance_editor()
        if not rows:
            return
        visible_range_hz = None
        if self.attached_res_edit_ax is not None:
            try:
                x0, x1 = self.attached_res_edit_ax.get_xlim()
                lo_ghz, hi_ghz = (float(x0), float(x1)) if float(x0) <= float(x1) else (float(x1), float(x0))
                visible_range_hz = (lo_ghz * 1.0e9, hi_ghz * 1.0e9)
            except Exception:
                visible_range_hz = None
        offset_by_scan_key, _tick_info = self._attached_resonance_editor_offset_map(
            rows,
            self._attached_resonance_editor_curve_spacing(),
        )
        best_row = None
        best_distance = None
        for row in rows:
            offset = float(offset_by_scan_key[str(row["scan_key"])])
            amp_display = self._attached_resonance_editor_display_amp(row["amp"])
            y_trace = self._interpolate_y(row["freq"], amp_display, x_ghz * 1.0e9) + offset
            distance = abs(y_trace - y_val)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_row = row
        if best_row is None or (best_distance is not None and best_distance > 0.35):
            if self.attached_res_edit_status_var is not None:
                self.attached_res_edit_status_var.set("Click closer to a scan trace to add a resonator.")
            return
        resonator_number = self._attached_resonance_editor_working_number()
        scan = best_row["scan"]
        target_hz = self._attached_resonance_editor_minimum_near_click(
            best_row["freq"],
            best_row["amp"],
            x_ghz * 1.0e9,
            visible_range_hz=visible_range_hz,
        )
        if target_hz is None:
            detail = "Unable to find a visible local minimum within the current x-range."
            if visible_range_hz is not None:
                detail = (
                    f"Unable to find a visible local minimum. Click={x_ghz:.9g} GHz, "
                    f"visible range={visible_range_hz[0] / 1.0e9:.9g} to {visible_range_hz[1] / 1.0e9:.9g} GHz."
                )
            if self.attached_res_edit_status_var is not None:
                self.attached_res_edit_status_var.set(detail)
            messagebox.showwarning("Add resonator failed", detail, parent=self.attached_res_edit_window)
            return
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
        payload = self._sheet_resonance_attachment(scan)
        assignments = payload["assignments"]
        assignments[resonator_number] = {
            "frequency_hz": target_hz,
            "sheet_path": "",
            "sheet_name": "",
            "row": 0,
            "column": 0,
            "identifier": Path(scan.filename).name,
        }
        self.dataset.processing_history.append(
            _make_event(
                "add_attached_resonator",
                {"scan": scan.filename, "resonator_number": resonator_number, "frequency_hz": target_hz},
            )
        )
        self._attached_res_edit_changed = True
        self._attached_res_edit_selected = (self._scan_key(scan), resonator_number)
        self._render_attached_resonance_editor()

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
        self._render_attached_resonance_editor()

def main() -> None:
    root = tk.Tk()
    DataAnalysisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
