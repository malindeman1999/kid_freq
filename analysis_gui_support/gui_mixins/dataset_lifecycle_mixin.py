from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

from tkinter import filedialog, messagebox, simpledialog

from analysis_gui_support.analysis_io import (
    DATASETS_DIR,
    DEFAULT_DATASET_FILE,
    _dataset_pickle_path,
    _load_dataset,
    _safe_name,
    _save_dataset,
    _write_app_state,
)
from analysis_gui_support.analysis_models import Dataset, _make_event

class DatasetLifecycleMixin:
    def _dataset_res_neighbor_initial_date(self) -> str:
        return str(getattr(self.dataset, "res_neighbor_initial_date", "") or "")


    def _sync_res_neighbor_initial_date(self, *, autosave: bool = False) -> None:
        if self.res_neighbor_dfrel_initial_date_var is None:
            return
        new_value = str(self.res_neighbor_dfrel_initial_date_var.get()).strip()
        if new_value == self._dataset_res_neighbor_initial_date():
            return
        self.dataset.res_neighbor_initial_date = new_value
        self._mark_dirty()
        self._refresh_status()
        if autosave:
            self._autosave_dataset()


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


    def _path_identity_from_stem(self, path: Path) -> tuple[str, str]:
        match = re.match(r"^(?P<prefix>\d{8}_\d{6})_(?P<name>.+)$", path.stem)
        if not match:
            return "", ""
        return match.group("prefix"), match.group("name")


    def _backfill_dataset_identity_from_path(self) -> None:
        prefix, name = self._path_identity_from_stem(self.dataset_path.resolve())
        if not prefix or not name:
            return
        if not self.dataset.created_at:
            try:
                dt = datetime.strptime(prefix, "%Y%m%d_%H%M%S")
                self.dataset.created_at = dt.isoformat(timespec="seconds")
            except Exception:
                pass
        if not self.dataset.dataset_name:
            self.dataset.dataset_name = name


    def _reconcile_dataset_path_for_save(self) -> None:
        current_path = self.dataset_path.resolve()
        self._backfill_dataset_identity_from_path()
        if self.dataset.dataset_name and self.dataset.created_at:
            target_path = _dataset_pickle_path(self.dataset).resolve()
        else:
            target_path = current_path

        if target_path == current_path:
            self.dataset_path = target_path
            return

        if target_path.exists():
            raise FileExistsError(f"Target dataset file already exists:\n{target_path}")

        current_exists = current_path.exists()
        current_dir = current_path.parent.resolve()
        target_dir = target_path.parent.resolve()

        if current_exists and current_dir == target_dir:
            current_path.rename(target_path)
        elif current_exists and current_dir != target_dir:
            if current_dir == DATASETS_DIR.resolve():
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(current_path), str(target_path))
            else:
                if target_dir.exists():
                    raise FileExistsError(f"Target dataset folder already exists:\n{target_dir}")
                shutil.move(str(current_dir), str(target_dir))
                moved_path = target_dir / current_path.name
                if moved_path.exists() and moved_path != target_path:
                    moved_path.rename(target_path)
        else:
            target_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_path = target_path


    def _persist_dataset(self) -> bool:
        try:
            if not self.dataset.created_at:
                self.dataset.created_at = datetime.now().isoformat(timespec="seconds")
            self._reconcile_dataset_path_for_save()

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
        if self.res_neighbor_dfrel_initial_date_var is not None:
            self.res_neighbor_dfrel_initial_date_var.set(self._dataset_res_neighbor_initial_date())
        self._reload_transcript_ui()
        self._mark_clean()
        self._refresh_status()
        self._log(f"Started new empty dataset: {cleaned_name}")


    def rename_dataset_prefix(self) -> None:
        if not self.dataset.dataset_name or not self.dataset.created_at:
            messagebox.showwarning(
                "Rename unavailable",
                "This dataset does not have a saved prefix yet. Create or save the dataset first.",
            )
            return

        current_name = str(self.dataset.dataset_name).strip()
        proposed_name = simpledialog.askstring(
            "Rename Dataset Prefix",
            "Enter the new prefix to use for the dataset folder and pickle filename:",
            initialvalue=current_name,
            parent=self.root,
        )
        if proposed_name is None:
            return

        cleaned_name = _safe_name(proposed_name)
        if not cleaned_name:
            messagebox.showwarning("Invalid prefix", "Please enter a non-empty dataset prefix.")
            return
        if cleaned_name == current_name:
            return

        old_dataset_name = current_name
        old_dataset_path = self.dataset_path.resolve()
        old_dataset_source = str(self.dataset.source_file)
        old_dataset_dir = old_dataset_path.parent.resolve()
        old_history_len = len(self.dataset.processing_history)
        old_dir_exists = old_dataset_dir.exists()
        moved_dir = False
        legacy_pickle_path: Optional[Path] = None

        try:
            self.dataset.dataset_name = cleaned_name
            new_dataset_path = _dataset_pickle_path(self.dataset).resolve()
            new_dataset_dir = new_dataset_path.parent.resolve()

            if new_dataset_dir.exists():
                raise FileExistsError(f"Target dataset folder already exists:\n{new_dataset_dir}")

            if old_dir_exists and old_dataset_dir != new_dataset_dir:
                shutil.move(str(old_dataset_dir), str(new_dataset_dir))
                moved_dir = True

            if moved_dir:
                legacy_pickle_path = (new_dataset_dir / old_dataset_path.name).resolve()
            else:
                legacy_pickle_path = old_dataset_path

            self.dataset_path = new_dataset_path
            self.dataset.source_file = str(new_dataset_path)
            self.dataset.processing_history.append(
                _make_event(
                    "rename_dataset_prefix",
                    {
                        "old_dataset_name": old_dataset_name,
                        "new_dataset_name": cleaned_name,
                        "old_dataset_path": str(old_dataset_path),
                        "new_dataset_path": str(new_dataset_path),
                    },
                )
            )
            _save_dataset(self.dataset, self.dataset_path)
            if (
                legacy_pickle_path is not None
                and legacy_pickle_path.exists()
                and legacy_pickle_path != self.dataset_path
            ):
                legacy_pickle_path.unlink()
            _write_app_state(self.dataset_path)
            self._mark_clean()
            self._refresh_status()
            self._log(f"Renamed dataset prefix from {old_dataset_name} to {cleaned_name}.")
        except Exception as exc:
            if moved_dir and new_dataset_dir.exists() and not old_dataset_dir.exists():
                try:
                    shutil.move(str(new_dataset_dir), str(old_dataset_dir))
                except Exception:
                    pass
            del self.dataset.processing_history[old_history_len:]
            self.dataset.dataset_name = old_dataset_name
            self.dataset_path = old_dataset_path
            self.dataset.source_file = old_dataset_source
            self._mark_dirty()
            self._refresh_status()
            self._log(f"Rename failed: {exc}")
            messagebox.showerror("Rename failed", str(exc))


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
            if self.res_neighbor_dfrel_initial_date_var is not None:
                self.res_neighbor_dfrel_initial_date_var.set(self._dataset_res_neighbor_initial_date())
            self._reload_transcript_ui()
            self._mark_clean()
            self._refresh_status()
            self._log(f"Dataset loaded: {self.dataset_path}")
            messagebox.showinfo("Dataset loaded", f"Loaded dataset:\n{self.dataset_path}")
        except Exception as exc:
            self._log(f"Load failed: {exc}")
            messagebox.showerror("Load failed", str(exc))
