"""Tests for Stage 6: feasibility classifier priority ordering."""

from __future__ import annotations

from winding_landscape.config import Constraints
from winding_landscape.feasibility.checker import classify_feasibility
from winding_landscape.performance.characterized_design import CharacterizedDesign
from winding_landscape.topology.topology_candidate import TopologyCandidate
from winding_landscape.winding.winding_candidate import WindingCandidate

import numpy as np


def _make_design(
    fit_status: str = "fits",
    peak_torque: float = 100.0,
    max_speed: float = 100000.0,
    cont_torque_thermal: float = 100.0,
) -> CharacterizedDesign:
    topo = TopologyCandidate(
        Q=12, pole_count=10, layers=2, coil_pitch_slots=1, parallel_paths=1,
        connection_matrix=np.zeros((12, 3), dtype=int),
        winding_factor_fundamental=0.933, winding_factor_5th=0.067,
        winding_factor_7th=0.067, winding_factor_11th=0.067, winding_factor_13th=0.933,
        is_balanced=True, cogging_period_mech_deg=6.0,
        topology_score=1.0, coils_per_phase=8, coils_in_series_per_path=8,
    )
    wind = WindingCandidate(
        topology=topo, turns_per_coil=20, wire_gauge_AWG=20,
        wire_diameter_bare_mm=0.812, wire_diameter_insulated_mm=0.869,
        wire_area_mm2=0.518, strands_per_conductor=1,
        manufacturing_method="random_round",
        slot_fill_factor_actual=0.40, slot_fill_factor_limit=0.42,
        fit_status=fit_status, end_turn_length_estimate_mm=10.0,
        total_copper_mass_g=50.0, R_phase_ohm_20C=0.05,
        current_density_at_target_continuous_A_per_mm2=5.0,
    )
    d = CharacterizedDesign(winding=wind)
    d.peak_torque_at_max_current_Nm = peak_torque
    d.max_speed_at_supply_voltage_rpm = max_speed
    d.continuous_torque_thermal_limit_Nm = cont_torque_thermal
    return d


def test_all_pass_is_feasible():
    c = Constraints()
    d = _make_design()
    classify_feasibility(d, c)
    assert d.feasibility_status == "feasible"


def test_fit_takes_priority():
    c = Constraints()
    # All other constraints fail too; fit is reported.
    d = _make_design(
        fit_status="no_fit_area",
        peak_torque=0.0, max_speed=0.0, cont_torque_thermal=0.0,
    )
    classify_feasibility(d, c)
    assert d.feasibility_status == "infeasible_fit"
    # And the notes should mention all the issues, not just fit.
    assert "fit" in d.feasibility_notes
    assert "peak torque" in d.feasibility_notes
    assert "max speed" in d.feasibility_notes
    assert "thermal" in d.feasibility_notes


def test_current_takes_priority_over_voltage_and_thermal():
    c = Constraints()
    d = _make_design(
        fit_status="fits", peak_torque=0.0, max_speed=0.0, cont_torque_thermal=0.0,
    )
    classify_feasibility(d, c)
    assert d.feasibility_status == "infeasible_current"


def test_voltage_takes_priority_over_thermal():
    c = Constraints()
    d = _make_design(
        fit_status="fits", peak_torque=100.0, max_speed=0.0, cont_torque_thermal=0.0,
    )
    classify_feasibility(d, c)
    assert d.feasibility_status == "infeasible_voltage"


def test_thermal_only():
    c = Constraints()
    d = _make_design(
        fit_status="fits", peak_torque=100.0, max_speed=100000.0, cont_torque_thermal=0.0,
    )
    classify_feasibility(d, c)
    assert d.feasibility_status == "infeasible_thermal"
