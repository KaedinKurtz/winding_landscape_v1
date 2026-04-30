"""WindingCandidate dataclass: the contract from Stage 3 to Stage 4."""

from __future__ import annotations

from dataclasses import dataclass

from winding_landscape.topology.topology_candidate import TopologyCandidate


@dataclass(frozen=True)
class WindingCandidate:
    """One fully-specified winding design (topology + turns + gauge).

    Mechanical-engineering analogy: at this point we've picked our gear ratio
    AND chosen the gear module / face width / material. The design is
    geometrically specified; what remains is to evaluate its performance.
    """

    topology: TopologyCandidate

    turns_per_coil: int
    wire_gauge_AWG: int
    wire_diameter_bare_mm: float
    wire_diameter_insulated_mm: float
    wire_area_mm2: float
    strands_per_conductor: int

    manufacturing_method: str  # always "random_round" in V1

    slot_fill_factor_actual: float
    """Actual fill = (insulated copper area) / slot useful area."""

    slot_fill_factor_limit: float
    """Achievable fill factor for this manufacturing method (from constraints)."""

    fit_status: str
    """One of: 'fits', 'no_fit_area', 'no_fit_opening'."""

    end_turn_length_estimate_mm: float
    """Per-coil single-side end-turn length, used in resistance calc."""

    total_copper_mass_g: float
    R_phase_ohm_20C: float
    current_density_at_target_continuous_A_per_mm2: float

    def is_geometrically_feasible(self) -> bool:
        return self.fit_status == "fits"
