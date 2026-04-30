"""TopologyCandidate dataclass: the contract from Stage 2 to Stage 3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TopologyCandidate:
    """One valid (Q, 2p, layers, pitch) topology with its winding-factor signature.

    Mechanical-engineering analogy: think of this as one row in a "candidate
    gear ratios" table. Each row represents a viable kinematic configuration
    that passes the basic sanity checks; downstream stages then size the actual
    parts (turns, gauge) on top of the chosen ratio.
    """

    Q: int
    """Slot count."""

    pole_count: int
    """2p -- always even."""

    layers: int
    """1 (single-layer) or 2 (double-layer). V1 produces only double-layer."""

    coil_pitch_slots: int
    """Coil span in integer slot pitches."""

    parallel_paths: int
    """Number of parallel paths in the winding (a.k.a. 'a' in textbooks)."""

    connection_matrix: NDArray[np.int_]
    """Q x 3 integer matrix. Row i, column j holds the signed contribution of
    slot i to phase j (column 0=A, 1=B, 2=C). Sign indicates winding direction;
    magnitude is the number of layers/coil-sides assigned to that phase."""

    winding_factor_fundamental: float
    winding_factor_5th: float
    winding_factor_7th: float
    winding_factor_11th: float
    winding_factor_13th: float

    is_balanced: bool
    """Star-of-slots balance check (3-phase symmetry)."""

    cogging_period_mech_deg: float
    """360 / LCM(Q, 2p) -- one mechanical period of the cogging waveform."""

    topology_score: float
    """Heuristic ranking score, higher is better."""

    coils_per_phase: int
    """Number of coils belonging to one phase, summed over all parallel paths."""

    coils_in_series_per_path: int
    """Number of coils in series within a single parallel path."""

    def topology_id(self) -> str:
        """Human-readable identifier: e.g. 'Q12_p10_L2_y1'."""
        return f"Q{self.Q}_p{self.pole_count}_L{self.layers}_y{self.coil_pitch_slots}"
