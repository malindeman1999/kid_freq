from __future__ import annotations

import queue
import threading
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter import scrolledtext, ttk

try:
    import winsound
except Exception:  # pragma: no cover - non-Windows fallback
    winsound = None

from analysis_gui_support.analysis_io import (
    _load_dataset,
    _read_app_state,
)
from analysis_gui_support.analysis_models import (
    VNAScan,
    _read_polar_series,
)
from analysis_gui_support.gui_mixins.analysis_selector_plot_mixin import AnalysisSelectorPlotMixin
from analysis_gui_support.gui_mixins.attached_resonance_editor_mixin import (
    AttachedResonanceEditorMixin,
)
from analysis_gui_support.gui_mixins.baseline_filter_mixin import BaselineFilterMixin
from analysis_gui_support.gui_mixins.dataset_lifecycle_mixin import DatasetLifecycleMixin
from analysis_gui_support.gui_mixins.dsdf_convolution_mixin import DSDFConvolutionMixin
from analysis_gui_support.gui_mixins.gaussian_convolution_mixin import GaussianConvolutionMixin
from analysis_gui_support.gui_mixins.interpolation_smooth_mixin import InterpolationSmoothMixin
from analysis_gui_support.gui_mixins.normalization_mixin import NormalizationMixin
from analysis_gui_support.gui_mixins.phase_correction2_mixin import PhaseCorrection2Mixin
from analysis_gui_support.gui_mixins.resonance_selection_mixin import ResonanceSelectionMixin
from analysis_gui_support.gui_mixins.resonance_sheet_io_mixin import ResonanceSheetIOMixin
from analysis_gui_support.gui_mixins.resonator_neighbor_analysis_mixin import (
    ResonatorNeighborAnalysisMixin,
)
from analysis_gui_support.gui_mixins.scan_date_tools_mixin import ScanDateToolsMixin
from analysis_gui_support.gui_mixins.scan_evolution_mixin import ScanEvolutionMixin
from analysis_gui_support.gui_mixins.scan_io_mixin import ScanIOMixin
from analysis_gui_support.gui_mixins.third_phase_correction_mixin import ThirdPhaseCorrectionMixin
from analysis_gui_support.gui_mixins.synthetic_generator_mixin import SyntheticGeneratorMixin
from analysis_gui_support.gui_mixins.unwrap_phase_mixin import UnwrapPhaseMixin

class DataAnalysisGUI(
    AnalysisSelectorPlotMixin,
    AttachedResonanceEditorMixin,
    BaselineFilterMixin,
    DatasetLifecycleMixin,
    DSDFConvolutionMixin,
    GaussianConvolutionMixin,
    InterpolationSmoothMixin,
    NormalizationMixin,
    PhaseCorrection2Mixin,
    ResonanceSheetIOMixin,
    ResonatorNeighborAnalysisMixin,
    ResonanceSelectionMixin,
    ScanDateToolsMixin,
    ScanEvolutionMixin,
    ScanIOMixin,
    ThirdPhaseCorrectionMixin,
    SyntheticGeneratorMixin,
    UnwrapPhaseMixin,
):
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("VNA Data Analysis")
        self.dataset_path = _read_app_state().resolve()
        self.dataset = _load_dataset(self.dataset_path)
        self._scrollable_plot_hosts: Dict[str, Dict[str, object]] = {}

        self.dataset_meta_var = tk.StringVar()
        self.dataset_label_var = tk.StringVar()
        self.scan_count_var = tk.StringVar()
        self.selection_var = tk.StringVar()
        self.saved_var = tk.StringVar()
        self.update_dates_mode_var = tk.StringVar(value="dir")
        self.synth_button: Optional[tk.Button] = None
        self.select_scans_button: Optional[tk.Button] = None
        self.unwrap_button: Optional[tk.Button] = None
        self.phase2_button: Optional[tk.Button] = None
        self.phase3_button: Optional[tk.Button] = None
        self.baseline_button: Optional[tk.Button] = None
        self.interp_button: Optional[tk.Button] = None
        self.norm_apply_large_button: Optional[tk.Button] = None
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
        self._baseline_target_scan_keys_override: Optional[set[str]] = None
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
        self.plot_scans_use_unwrapped_phase_var: Optional[tk.BooleanVar] = None
        self.plot_scans_status_var: Optional[tk.StringVar] = None
        self.scan_evolution_window: Optional[tk.Toplevel] = None
        self.scan_evolution_canvas: Optional[FigureCanvasTkAgg] = None
        self.scan_evolution_toolbar: Optional[NavigationToolbar2Tk] = None
        self.scan_evolution_figure: Optional[Figure] = None
        self.scan_evolution_status_var: Optional[tk.StringVar] = None
        self.scan_evolution_mod360_var: Optional[tk.BooleanVar] = None
        self.scan_evolution_show_gaussian_var: Optional[tk.BooleanVar] = None
        self.scan_evolution_show_dsdf_var: Optional[tk.BooleanVar] = None
        self.scan_evolution_show_phase_2pi_var: Optional[tk.BooleanVar] = None
        self.scan_evolution_show_phase_vna_var: Optional[tk.BooleanVar] = None
        self.scan_evolution_show_phase_other_var: Optional[tk.BooleanVar] = None
        self.scan_evolution_show_attached_res_var: Optional[tk.BooleanVar] = None
        self._scan_evolution_scan_key: Optional[str] = None
        self._scan_evolution_stage_rows: List[dict] = []
        self._scan_evolution_axes_rows: List[tuple[object, object, object]] = []
        self._scan_evolution_syncing_xlim: bool = False
        self.res_shift_corr_window: Optional[tk.Toplevel] = None
        self.res_shift_corr_canvas: Optional[FigureCanvasTkAgg] = None
        self.res_shift_corr_toolbar: Optional[NavigationToolbar2Tk] = None
        self.res_shift_corr_figure: Optional[Figure] = None
        self.res_shift_corr_status_var: Optional[tk.StringVar] = None
        self._res_shift_corr_axes: tuple[object, object] | None = None
        self.res_pair_dfdiff_hist_window: Optional[tk.Toplevel] = None
        self.res_pair_dfdiff_hist_canvas: Optional[FigureCanvasTkAgg] = None
        self.res_pair_dfdiff_hist_toolbar: Optional[NavigationToolbar2Tk] = None
        self.res_pair_dfdiff_hist_figure: Optional[Figure] = None
        self.res_pair_dfdiff_hist_status_var: Optional[tk.StringVar] = None
        self.res_pair_dfdiff_hist_sep_mhz_var: Optional[tk.DoubleVar] = None
        self.res_pair_dfdiff_hist_bin_mhz_var: Optional[tk.DoubleVar] = None
        self.res_pair_dfdiff_hist_capture_var: Optional[tk.DoubleVar] = None
        self.res_pair_dfdiff_hist_fit_mode_var: Optional[tk.StringVar] = None
        self.res_pair_dfdiff_hist_num_res_var: Optional[tk.IntVar] = None
        self.res_pair_dfdiff_hist_center_ghz_var: Optional[tk.DoubleVar] = None
        self.res_pair_dfdiff_hist_dfrel_var: Optional[tk.DoubleVar] = None
        self.res_pair_dfdiff_hist_freq_jitter_khz_var: Optional[tk.DoubleVar] = None
        self.res_pair_dfdiff_hist_sep_scale: Optional[tk.Scale] = None
        self.res_pair_dfdiff_hist_bin_scale: Optional[tk.Scale] = None
        self.res_pair_dfdiff_hist_capture_scale: Optional[tk.Scale] = None
        self.res_pair_dfdiff_hist_freq_jitter_scale: Optional[tk.Scale] = None
        self._res_pair_dfdiff_hist_ax = None
        self.res_neighbor_dfrel_window: Optional[tk.Toplevel] = None
        self.res_neighbor_dfrel_canvas: Optional[FigureCanvasTkAgg] = None
        self.res_neighbor_dfrel_toolbar: Optional[NavigationToolbar2Tk] = None
        self.res_neighbor_dfrel_figure: Optional[Figure] = None
        self.res_neighbor_dfrel_status_var: Optional[tk.StringVar] = None
        self.res_neighbor_dfrel_sep_rel_var: Optional[tk.DoubleVar] = None
        self.res_neighbor_dfrel_show_iqr_var: Optional[tk.BooleanVar] = None
        self.res_neighbor_dfrel_mode_var: Optional[tk.StringVar] = None
        self.res_neighbor_dfrel_initial_date_var: Optional[tk.StringVar] = None
        self.res_neighbor_dfrel_sep_scale: Optional[tk.Scale] = None
        self._res_neighbor_dfrel_ax = None
        self.res_neighbor_top_rates_window: Optional[tk.Toplevel] = None
        self.res_neighbor_top_rates_status_var: Optional[tk.StringVar] = None
        self.res_neighbor_top_rates_tree: Optional[ttk.Treeview] = None
        self.res_neighbor_top_rates_text: Optional[tk.Text] = None
        self.res_neighbor_corr_window: Optional[tk.Toplevel] = None
        self.res_neighbor_corr_canvas: Optional[FigureCanvasTkAgg] = None
        self.res_neighbor_corr_toolbar: Optional[NavigationToolbar2Tk] = None
        self.res_neighbor_corr_figure: Optional[Figure] = None
        self.res_neighbor_corr_status_var: Optional[tk.StringVar] = None
        self.res_neighbor_corr_sep_rel_var: Optional[tk.DoubleVar] = None
        self.res_neighbor_corr_initial_date_var: Optional[tk.StringVar] = None
        self.res_neighbor_corr_show_curves_var: Optional[tk.BooleanVar] = None
        self.res_neighbor_corr_sep_scale: Optional[tk.Scale] = None
        self._res_neighbor_corr_axes: tuple[object, object] | None = None
        self.res_neighbor_scan_window: Optional[tk.Toplevel] = None
        self.res_neighbor_scan_canvas: Optional[FigureCanvasTkAgg] = None
        self.res_neighbor_scan_toolbar: Optional[NavigationToolbar2Tk] = None
        self.res_neighbor_scan_figure: Optional[Figure] = None
        self.res_neighbor_scan_status_var: Optional[tk.StringVar] = None
        self._res_neighbor_scan_source: str = "dfrel"
        self._res_neighbor_scan_ax = None
        self.attached_res_edit_window: Optional[tk.Toplevel] = None
        self.attached_res_edit_canvas: Optional[FigureCanvasTkAgg] = None
        self.attached_res_edit_toolbar: Optional[NavigationToolbar2Tk] = None
        self.attached_res_edit_figure: Optional[Figure] = None
        self.attached_res_edit_status_var: Optional[tk.StringVar] = None
        self.attached_res_edit_working_number_var: Optional[tk.StringVar] = None
        self.attached_res_edit_working_number_spinbox: Optional[tk.Spinbox] = None
        self.attached_res_edit_spacing_var: Optional[tk.DoubleVar] = None
        self.attached_res_edit_spacing_scale: Optional[tk.Scale] = None
        self.attached_res_edit_search_window_khz_var: Optional[tk.DoubleVar] = None
        self.attached_res_edit_search_window_scale: Optional[tk.Scale] = None
        self.attached_res_edit_truncate_var: Optional[tk.BooleanVar] = None
        self.attached_res_edit_truncate_threshold_var: Optional[tk.DoubleVar] = None
        self.attached_res_edit_truncate_threshold_scale: Optional[tk.Scale] = None
        self.attached_res_edit_add_button: Optional[tk.Button] = None
        self.attached_res_edit_renumber_button: Optional[tk.Button] = None
        self.attached_res_edit_undo_button: Optional[tk.Button] = None
        self.attached_res_edit_save_button: Optional[tk.Button] = None
        self.attached_res_edit_exit_button: Optional[tk.Button] = None
        self.attached_res_edit_ax = None
        self._attached_res_edit_points: List[dict] = []
        self._attached_res_edit_rows_cache: List[dict] = []
        self._attached_res_edit_offset_by_scan_key: Dict[str, float] = {}
        self._attached_res_edit_selected: Optional[tuple[str, str]] = None
        self._attached_res_edit_pending_add: bool = False
        self._attached_res_edit_default_xlim: Optional[tuple[float, float]] = None
        self._attached_res_edit_missing_normalized_warned: Optional[tuple[str, ...]] = None
        self._attached_res_edit_snapshot: Optional[dict] = None
        self._attached_res_edit_undo_stack: List[dict] = []
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

    def _ensure_scrollable_plot_host(self, key: str, window: tk.Misc) -> tuple[tk.Frame, tk.Frame]:
        host = self._scrollable_plot_hosts.get(key)
        if host is not None:
            return host["toolbar_frame"], host["inner_frame"]

        container = tk.Frame(window)
        container.pack(side="top", fill="both", expand=True)
        toolbar_frame = tk.Frame(container)
        toolbar_frame.pack(side="top", fill="x")
        viewport = tk.Frame(container)
        viewport.pack(side="top", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(viewport, orient="vertical")
        scrollbar.pack(side="right", fill="y")
        scroll_canvas = tk.Canvas(viewport, highlightthickness=0, yscrollcommand=scrollbar.set)
        scroll_canvas.pack(side="left", fill="both", expand=True)
        inner_frame = tk.Frame(scroll_canvas)
        inner_window = scroll_canvas.create_window((0, 0), window=inner_frame, anchor="nw")
        scrollbar.configure(command=scroll_canvas.yview)

        def _sync_scrollregion(_event=None) -> None:
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

        def _sync_inner_width(event) -> None:
            scroll_canvas.itemconfigure(inner_window, width=event.width)

        def _on_mousewheel(event) -> str:
            delta = int(getattr(event, "delta", 0))
            if delta == 0:
                return "break"
            step = -int(delta / 120) if abs(delta) >= 120 else (-1 if delta > 0 else 1)
            scroll_canvas.yview_scroll(step, "units")
            return "break"

        inner_frame.bind("<Configure>", _sync_scrollregion)
        scroll_canvas.bind("<Configure>", _sync_inner_width)
        scroll_canvas.bind("<MouseWheel>", _on_mousewheel)
        inner_frame.bind("<MouseWheel>", _on_mousewheel)

        self._scrollable_plot_hosts[key] = {
            "container": container,
            "toolbar_frame": toolbar_frame,
            "scroll_canvas": scroll_canvas,
            "inner_frame": inner_frame,
            "inner_window": inner_window,
        }
        return toolbar_frame, inner_frame

    def _destroy_scrollable_plot_host(self, key: str) -> None:
        host = self._scrollable_plot_hosts.pop(key, None)
        if host is None:
            return
        container = host.get("container")
        if isinstance(container, tk.Misc) and container.winfo_exists():
            container.destroy()

    def _set_scrollable_figure_size(
        self,
        key: str,
        figure: Optional[Figure],
        *,
        canvas_agg: Optional[FigureCanvasTkAgg] = None,
        width_in: float,
        row_count: int,
        row_height_in: float,
        min_height_in: float,
    ) -> None:
        if figure is None:
            return
        host = self._scrollable_plot_hosts.get(key)
        scroll_canvas = None
        inner_window = None
        if host is not None:
            scroll_canvas = host.get("scroll_canvas")
            inner_window = host.get("inner_window")

        height_in = max(min_height_in, row_count * row_height_in)
        dpi = float(figure.get_dpi()) if figure.get_dpi() else 100.0

        # Keep figures at least as wide as the visible viewport so toggles and first render
        # do not temporarily collapse plot width until a manual window resize occurs.
        if isinstance(scroll_canvas, tk.Canvas) and scroll_canvas.winfo_exists():
            scroll_canvas.update_idletasks()
            viewport_px = int(scroll_canvas.winfo_width())
            if viewport_px > 10 and dpi > 0:
                width_in = max(width_in, float(viewport_px) / dpi)

        figure.set_size_inches(width_in, height_in, forward=True)
        if canvas_agg is not None:
            widget = canvas_agg.get_tk_widget()
            widget.configure(
                width=max(1, int(round(width_in * dpi))),
                height=max(1, int(round(height_in * dpi))),
            )
        if isinstance(scroll_canvas, tk.Canvas) and scroll_canvas.winfo_exists():
            scroll_canvas.update_idletasks()
            if isinstance(inner_window, int):
                scroll_canvas.itemconfigure(inner_window, width=max(1, int(scroll_canvas.winfo_width())))
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

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
        button_area = tk.Frame(left)
        button_area.pack(anchor="w")
        button_col1 = tk.Frame(button_area)
        button_col1.pack(side="left", anchor="n")
        button_col2 = tk.Frame(button_area, padx=8)
        button_col2.pack(side="left", anchor="n")

        button_width = 24
        left_button_specs: list[dict[str, object]] = []
        right_button_specs: list[dict[str, object]] = []

        left_button_specs.append({"text": "New Dataset", "command": self.start_new_dataset})
        left_button_specs.append({"text": "Rename Dataset", "command": self.rename_dataset_prefix})
        self.synth_button = tk.Button(
            button_col1, text="Generate Synthetic Data", width=button_width, command=self.open_synthetic_generator_window
        )
        left_button_specs.append({"button": self.synth_button})
        left_button_specs.append({"text": "Load Different Dataset", "command": self.load_different_dataset})
        left_button_specs.append({"text": "Load VNA Scan(s)", "command": self.load_vna_scan})
        left_button_specs.append({"text": "Remove VNA Scan(s)", "command": self.remove_vna_scans})
        left_button_specs.append({"text": "Load Resonators From Sheet", "command": self.open_resonance_sheet_loader})
        left_button_specs.append({"text": "Save Resonators To Sheet", "command": self.open_resonance_sheet_saver})
        self.select_scans_button = tk.Button(
            button_col1, text="Select Scans for Analysis", width=button_width, command=self.open_analysis_selector
        )
        left_button_specs.append({"button": self.select_scans_button})
        left_button_specs.append({"text": "Group Selected Scans", "command": self.group_selected_scans_for_plotting})
        left_button_specs.append({"text": "Plot Selected VNA Scans", "command": self.plot_selected_vna_scans})
        self.unwrap_button = tk.Button(
            button_col1, text="Phase Correction 1", width=button_width, command=self.open_unwrap_phase_window
        )
        left_button_specs.append({"button": self.unwrap_button})
        self.phase2_button = tk.Button(
            button_col1, text="Phase Correction 2", width=button_width, command=self.open_second_phase_correction_window
        )
        left_button_specs.append({"button": self.phase2_button})
        self.phase3_button = tk.Button(
            button_col1, text="Phase Correction 3", width=button_width, command=self.open_third_phase_correction_window
        )
        left_button_specs.append({"button": self.phase3_button})
        self.baseline_button = tk.Button(
            button_col1, text="Baseline Filtering", width=button_width, command=self.open_baseline_filter_window
        )
        left_button_specs.append({"button": self.baseline_button})
        self.interp_button = tk.Button(
            button_col1, text="Interp + Smooth", width=button_width, command=self.open_interp_smooth_window
        )
        left_button_specs.append({"button": self.interp_button})
        self.norm_button = tk.Button(
            button_col1, text="Normalize Baseline", width=button_width, command=self.open_normalization_window
        )
        left_button_specs.append({"button": self.norm_button})
        self.norm_apply_large_button = tk.Button(
            button_col1, text="Apply Large-Scan Baseline", width=button_width, command=self.apply_large_scan_baseline_to_selected
        )
        left_button_specs.append({"button": self.norm_apply_large_button})
        left_button_specs.append({"text": "Scan Evolution", "command": self.open_scan_evolution_window})

        self.gauss_button = tk.Button(
            button_col2, text="Gaussian Convolve |S21|", width=button_width, command=self.open_gaussian_convolution_window
        )
        right_button_specs.append({"button": self.gauss_button})
        self.dsdf_button = tk.Button(
            button_col2, text="Gaussian Convolve |dS21/df|", width=button_width, command=self.open_dsdf_convolution_window
        )
        right_button_specs.append({"button": self.dsdf_button})
        self.res_button = tk.Button(
            button_col2, text="Resonance Fitting", width=button_width, command=self.open_resonance_selection_window
        )
        right_button_specs.append({"button": self.res_button})
        right_button_specs.append({"text": "Mark Res. on Sel. Scans", "command": self.open_attached_resonance_editor})
        right_button_specs.append(
            {
                "text": "Reset Selected Scans",
                "command": self.clear_selected_scan_attachments,
            }
        )
        right_button_specs.append({"text": "Plot Resonator Markers", "command": self.open_attached_resonance_plotter})
        right_button_specs.append({"text": "Update Dates From Path", "command": self.open_update_dates_dialog})
        right_button_specs.append({"text": "Reorder Scans By Date", "command": self.reorder_vna_scans_by_date})
        right_button_specs.append({"text": "Pair df/f vs Time", "command": self.open_resonator_neighbor_dfrel_window})
        right_button_specs.append({"text": "Pair Self Corr. vs Time", "command": self.open_resonator_neighbor_corr_window})
        right_button_specs.append({"text": "Shift Correl. Between Freqs.", "command": self.open_resonator_shift_correlation_window})
        right_button_specs.append({"text": "Histogram df2-df1", "command": self.open_resonator_pair_dfdiff_hist_window})

        for parent, specs in ((button_col1, left_button_specs), (button_col2, right_button_specs)):
            for spec in specs:
                button = spec.get("button")
                if not isinstance(button, tk.Button):
                    button = tk.Button(
                        parent,
                        text=str(spec["text"]),
                        width=button_width,
                        command=spec["command"],
                    )
                button.pack(anchor="w", pady=spec.get("pady", 2))

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

    def _select_multiple_setting_options(
        self,
        title: str,
        prompt: str,
        options: List[str],
        default_indices: Optional[List[int]] = None,
    ) -> List[int]:
        if not options:
            return []

        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("760x420")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text=prompt, anchor="w", justify="left", wraplength=720).pack(
            fill="x", padx=10, pady=(10, 4)
        )
        listbox = tk.Listbox(dialog, width=110, height=16, selectmode=tk.MULTIPLE)
        listbox.pack(fill="both", expand=True, padx=10, pady=4)
        for item in options:
            listbox.insert(tk.END, item)

        defaults = sorted({int(idx) for idx in (default_indices or []) if 0 <= int(idx) < len(options)})
        if defaults:
            for idx in defaults:
                listbox.selection_set(idx)
            listbox.see(defaults[0])

        selected_indices: Dict[str, List[int]] = {"value": []}

        def choose() -> None:
            selected_indices["value"] = [int(idx) for idx in listbox.curselection()]
            dialog.destroy()

        def cancel() -> None:
            dialog.destroy()

        btns = tk.Frame(dialog)
        btns.pack(fill="x", padx=10, pady=(4, 10))
        tk.Button(btns, text="Use Selected", command=choose).pack(side="right")
        tk.Button(btns, text="Cancel", command=cancel).pack(side="right", padx=(0, 8))

        dialog.wait_window()
        return selected_indices["value"]

    def _confirm_bulk_text_changes(self, title: str, prompt: str, lines: List[str]) -> bool:
        if not lines:
            return False
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("860x460")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text=prompt, anchor="w", justify="left", wraplength=820).pack(
            fill="x", padx=10, pady=(10, 4)
        )
        listbox = tk.Listbox(dialog, width=125, height=18)
        listbox.pack(fill="both", expand=True, padx=10, pady=4)
        for line in lines:
            listbox.insert(tk.END, line)

        decision: Dict[str, bool] = {"approve": False}

        def approve() -> None:
            decision["approve"] = True
            dialog.destroy()

        def cancel() -> None:
            dialog.destroy()

        btns = tk.Frame(dialog)
        btns.pack(fill="x", padx=10, pady=(4, 10))
        tk.Button(btns, text="Apply All", command=approve).pack(side="right")
        tk.Button(btns, text="Cancel", command=cancel).pack(side="right", padx=(0, 8))

        dialog.wait_window()
        return bool(decision["approve"])

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

    def _selected_scans_have_attached_filter(self) -> bool:
        scans = self._baseline_pipeline_selected_scans()
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
        if self._baseline_target_scan_keys_override is not None:
            override = self._baseline_target_scan_keys_override
            return [scan for scan in (scans if scans else list(self.dataset.vna_scans)) if self._scan_key(scan) in override]
        return scans if scans else list(self.dataset.vna_scans)

    @staticmethod
    def _scan_marked_omitted_from_baseline_fit(scan: VNAScan) -> bool:
        bf = scan.baseline_filter
        return isinstance(bf, dict) and bool(bf.get("omit_from_baseline_fit", False))

    def _baseline_pipeline_selected_scans(self) -> list[VNAScan]:
        scans = self._selected_scans()
        if not scans:
            return []
        if self._baseline_target_scan_keys_override is not None:
            override = self._baseline_target_scan_keys_override
            return [scan for scan in scans if self._scan_key(scan) in override]
        return [scan for scan in scans if not self._scan_marked_omitted_from_baseline_fit(scan)]

    def _baseline_pipeline_omitted_selected_scans(self) -> list[VNAScan]:
        scans = self._selected_scans()
        if not scans:
            return []
        if self._baseline_target_scan_keys_override is not None:
            override = self._baseline_target_scan_keys_override
            return [scan for scan in scans if self._scan_key(scan) not in override]
        return [scan for scan in scans if self._scan_marked_omitted_from_baseline_fit(scan)]

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
        scans = self._baseline_pipeline_selected_scans()
        done_count = sum(1 for scan in scans if self._has_valid_interp_output(scan))
        self._configure_action_button(
            self.interp_button,
            available=bool(scans) and all(self._has_valid_baseline_filter_output(scan) for scan in scans),
            done_count=done_count,
            total_count=len(scans),
        )

    def _selected_scans_have_attached_interp_data(self) -> bool:
        scans = self._baseline_pipeline_selected_scans()
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
        scans = self._baseline_pipeline_selected_scans()
        done_count = sum(1 for scan in scans if self._has_valid_normalized_output(scan))
        self._configure_action_button(
            self.norm_button,
            available=bool(scans) and all(self._has_valid_interp_output(scan) for scan in scans),
            done_count=done_count,
            total_count=len(scans),
        )
        self._configure_action_button(
            self.norm_apply_large_button,
            available=bool(scans)
            and all(self._has_valid_phase3_output(scan) for scan in scans)
            and any(self._has_valid_interp_output(scan) for scan in scans),
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


def main() -> None:
    root = tk.Tk()
    DataAnalysisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
