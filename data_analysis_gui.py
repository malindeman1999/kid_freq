from __future__ import annotations

import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter import filedialog, messagebox, simpledialog, scrolledtext, ttk

from analysis_io import (
    DATASETS_DIR,
    DEFAULT_DATASET_FILE,
    _dataset_pickle_path,
    _load_dataset,
    _load_vna_npy,
    _read_app_state,
    _safe_name,
    _save_dataset,
    _write_app_state,
)
from analysis_models import Dataset, VNAScan, _current_user, _make_event
from analysis_selector_plot_mixin import AnalysisSelectorPlotMixin
from baseline_filter_mixin import BaselineFilterMixin
from dsdf_convolution_mixin import DSDFConvolutionMixin
from gaussian_convolution_mixin import GaussianConvolutionMixin
from interpolation_smooth_mixin import InterpolationSmoothMixin
from normalization_mixin import NormalizationMixin
from resonance_selection_mixin import ResonanceSelectionMixin
from second_phase_correction_mixin import SecondPhaseCorrectionMixin
from synthetic_generator_mixin import SyntheticGeneratorMixin
from unwrap_phase_mixin import UnwrapPhaseMixin

class DataAnalysisGUI(
    AnalysisSelectorPlotMixin,
    BaselineFilterMixin,
    DSDFConvolutionMixin,
    GaussianConvolutionMixin,
    InterpolationSmoothMixin,
    NormalizationMixin,
    ResonanceSelectionMixin,
    SecondPhaseCorrectionMixin,
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
        self.unwrap_button: Optional[tk.Button] = None
        self.phase2_button: Optional[tk.Button] = None
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
        self.unwrap_p_random_var: Optional[tk.StringVar] = None
        self.unwrap_p_random_entry: Optional[tk.Entry] = None
        self.unwrap_auto_y_var: Optional[tk.BooleanVar] = None
        self.unwrap_status_var: Optional[tk.StringVar] = None
        self.unwrap_attach_button: Optional[tk.Button] = None
        self.unwrap_preview: Dict[str, Dict[str, np.ndarray]] = {}
        self.phase2_window: Optional[tk.Toplevel] = None
        self.phase2_canvas: Optional[FigureCanvasTkAgg] = None
        self.phase2_toolbar: Optional[NavigationToolbar2Tk] = None
        self.phase2_figure: Optional[Figure] = None
        self.phase2_auto_y_var: Optional[tk.BooleanVar] = None
        self.phase2_status_var: Optional[tk.StringVar] = None
        self.res_button: Optional[tk.Button] = None
        self.res_window: Optional[tk.Toplevel] = None
        self.res_canvas: Optional[FigureCanvasTkAgg] = None
        self.res_toolbar: Optional[NavigationToolbar2Tk] = None
        self.res_figure: Optional[Figure] = None
        self.res_status_var: Optional[tk.StringVar] = None
        self.res_auto_y_var: Optional[tk.BooleanVar] = None
        self.res_use_corrected_var: Optional[tk.BooleanVar] = None
        self.res_show_phase_var: Optional[tk.BooleanVar] = None
        self.res_span_selector = None
        self.res_amp_ax = None
        self.res_iq_ax = None
        self._res_scan_key: Optional[str] = None
        self._res_selected_range: Optional[tuple[float, float]] = None
        self._res_manual_ylim: Optional[tuple[float, float]] = None
        self._last_resonance_scan_key: Optional[str] = None
        self._build_layout()
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
        tk.Button(left, text="Load VNA Scan(s) (.npy)", width=24, command=self.load_vna_scan).pack(
            anchor="w", pady=2
        )
        tk.Button(
            left, text="Remove VNA Scan(s)", width=24, command=self.remove_vna_scans
        ).pack(anchor="w", pady=2)
        tk.Button(
            left, text="Select Scans for Analysis", width=24, command=self.open_analysis_selector
        ).pack(anchor="w", pady=2)
        tk.Button(
            left, text="Plot Selected VNA Scans", width=24, command=self.plot_selected_vna_scans
        ).pack(anchor="w", pady=2)
        self.unwrap_button = tk.Button(
            left, text="Phase Correction", width=24, command=self.open_unwrap_phase_window
        )
        self.unwrap_button.pack(anchor="w", pady=2)
        self.phase2_button = tk.Button(
            left, text="Phase Correction 2", width=24, command=self.open_second_phase_correction_window
        )
        self.phase2_button.pack(anchor="w", pady=2)
        tk.Button(
            left, text="Baseline Filtering", width=24, command=self.open_baseline_filter_window
        ).pack(anchor="w", pady=2)
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
            fd = np.asarray(bf.get("filtered_data_complex"))
            if fd.ndim != 2 or fd.shape[0] != 2:
                return False
            if fd.shape[1] == 0:
                return False
        return True

    def _update_interp_button_state(self) -> None:
        if self.interp_button is None:
            return
        if self._selected_scans_have_attached_filter():
            self.interp_button.configure(
                state="normal", bg="light green", activebackground="light green"
            )
        else:
            self.interp_button.configure(
                state="disabled", bg="light grey", activebackground="light grey"
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
            interp_complex = np.asarray(interp.get("interp_complex"))
            if interp_complex.ndim != 1:
                return False
            if interp_complex.size == 0:
                return False
            if interp_complex.size != scan.freq.size:
                return False
        return True

    def _update_norm_button_state(self) -> None:
        if self.norm_button is None:
            return
        if self._selected_scans_have_attached_interp_data():
            self.norm_button.configure(
                state="normal", bg="light green", activebackground="light green"
            )
        else:
            self.norm_button.configure(
                state="disabled", bg="light grey", activebackground="light grey"
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
            norm_complex = np.asarray(norm.get("norm_complex"))
            if norm_complex.ndim != 1:
                return False
            if norm_complex.size != scan.freq.size:
                return False
        return True

    def _update_res_button_state(self) -> None:
        if self.res_button is None:
            return
        if self._selected_scans_have_attached_normalized():
            self.res_button.configure(
                state="normal", bg="light green", activebackground="light green"
            )
        else:
            self.res_button.configure(
                state="disabled", bg="light grey", activebackground="light grey"
            )

    def _update_gauss_button_state(self) -> None:
        if self.gauss_button is None:
            return
        if self._selected_scans_have_attached_normalized():
            self.gauss_button.configure(
                state="normal", bg="light green", activebackground="light green"
            )
        else:
            self.gauss_button.configure(
                state="disabled", bg="light grey", activebackground="light grey"
            )

    def _update_dsdf_button_state(self) -> None:
        if self.dsdf_button is None:
            return
        if self._selected_scans_have_attached_normalized():
            self.dsdf_button.configure(
                state="normal", bg="light green", activebackground="light green"
            )
        else:
            self.dsdf_button.configure(
                state="disabled", bg="light grey", activebackground="light grey"
            )

    def _update_unwrap_button_state(self) -> None:
        if self.unwrap_button is None:
            return
        if len(self._selected_scans()) > 0:
            self.unwrap_button.configure(
                state="normal", bg="light green", activebackground="light green"
            )
        else:
            self.unwrap_button.configure(
                state="disabled", bg="light grey", activebackground="light grey"
            )

    def _update_phase2_button_state(self) -> None:
        if self.phase2_button is None:
            return
        scans = self._selected_scans()
        if scans and all(scan.has_dewrapped_phase() for scan in scans):
            self.phase2_button.configure(
                state="normal", bg="light green", activebackground="light green"
            )
        else:
            self.phase2_button.configure(
                state="disabled", bg="light grey", activebackground="light grey"
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
            title="Select VNA scan file(s) (.npy)",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
        )
        if not path_texts:
            return

        added_count = 0
        failed: List[str] = []
        for path_text in path_texts:
            path = Path(path_text)
            try:
                scan = _load_vna_npy(path)
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
            except Exception as exc:
                failed.append(f"{path.name}: {exc}")

        self._refresh_status()
        self._log(f"VNA load result: added={added_count}, failed={len(failed)}")
        if added_count > 0:
            self._autosave_dataset()

        if added_count > 0 and not failed:
            messagebox.showinfo(
                "VNA scans loaded",
                f"Added {added_count} scan(s). Dataset saved automatically.",
            )
            return

        if added_count > 0 and failed:
            messagebox.showwarning(
                "VNA scan load partial",
                f"Added {added_count} scan(s), {len(failed)} failed.\n\n"
                + "\n".join(failed[:8]),
            )
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

def main() -> None:
    root = tk.Tk()
    DataAnalysisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
