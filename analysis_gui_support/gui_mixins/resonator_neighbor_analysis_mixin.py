from __future__ import annotations

from .resonator_neighbor_corr_window_mixin import ResonatorNeighborCorrWindowMixin
from .resonator_neighbor_data_mixin import ResonatorNeighborDataMixin
from .resonator_neighbor_dfrel_window_mixin import ResonatorNeighborDfrelWindowMixin
from .resonator_shift_hist_mixin import ResonatorShiftHistMixin


class ResonatorNeighborAnalysisMixin(
    ResonatorNeighborDataMixin,
    ResonatorNeighborCorrWindowMixin,
    ResonatorNeighborDfrelWindowMixin,
    ResonatorShiftHistMixin,
):
    pass
