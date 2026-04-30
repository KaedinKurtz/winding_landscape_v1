"""Tests for Stage 5: thermal characterization."""

from __future__ import annotations

from winding_landscape.performance.electromagnetic import characterize_electromagnetic
from winding_landscape.performance.thermal import (
    _solve_temperatures,
    characterize_thermal,
)
from winding_landscape.topology.enumeration import enumerate_topologies
from winding_landscape.winding.enumeration import enumerate_windings


def _make_design(synthetic_geometry, reasonable_constraints, materials_db):
    """Helper: return one fully EM-characterized feasible design."""
    tops = enumerate_topologies(synthetic_geometry, reasonable_constraints)
    ws = enumerate_windings(tops[0], synthetic_geometry, reasonable_constraints, materials_db)
    feasible = [w for w in ws if w.fit_status == "fits"]
    assert feasible
    return characterize_electromagnetic(
        feasible[0], synthetic_geometry, reasonable_constraints, materials_db
    )


def test_thermal_outputs_populated(
    synthetic_geometry, reasonable_constraints, materials_db
):
    d = _make_design(synthetic_geometry, reasonable_constraints, materials_db)
    d = characterize_thermal(d, synthetic_geometry, reasonable_constraints, materials_db)
    assert d.predicted_winding_temp_at_continuous_C > reasonable_constraints.thermal_envelope.ambient_temp_C - 1
    assert d.copper_loss_at_continuous_W >= 0
    assert d.continuous_torque_thermal_limit_Nm > 0
    assert d.thermal_iterations_used > 0


def test_thermal_temp_above_ambient(
    synthetic_geometry, reasonable_constraints, materials_db
):
    """Winding temp must always be >= ambient (only sources are losses; no cooler)."""
    d = _make_design(synthetic_geometry, reasonable_constraints, materials_db)
    d = characterize_thermal(d, synthetic_geometry, reasonable_constraints, materials_db)
    assert d.predicted_winding_temp_at_continuous_C >= reasonable_constraints.thermal_envelope.ambient_temp_C
    assert d.predicted_iron_temp_at_continuous_C >= reasonable_constraints.thermal_envelope.ambient_temp_C


def test_thermal_winding_hotter_than_iron(
    synthetic_geometry, reasonable_constraints, materials_db
):
    """In steady-state with copper loss, winding must be hotter than the iron core."""
    d = _make_design(synthetic_geometry, reasonable_constraints, materials_db)
    d = characterize_thermal(d, synthetic_geometry, reasonable_constraints, materials_db)
    if d.copper_loss_at_continuous_W > 0:
        assert d.predicted_winding_temp_at_continuous_C >= d.predicted_iron_temp_at_continuous_C


def test_solve_zero_current_zero_loss():
    """At I=0 and zero iron loss, all temps should be at or very near ambient.

    The under-relaxed iteration converges within _TEMP_TOL_C of the true value;
    we allow 2 degC slack here to account for that."""
    Tw, Ti, Pcu, R, n = _solve_temperatures(
        I_phase_rms=0.0,
        P_iron_W=0.0,
        R_phase_20C=0.5,
        R_w_to_iron=0.5,
        R_iron_to_housing=0.5,
        R_housing_to_amb=0.5,
        T_ambient=25.0,
        copper_temp_coeff=0.00393,
    )
    assert abs(Tw - 25.0) < 2.0
    assert Pcu == 0.0
    assert abs(Ti - 25.0) < 1e-6
    assert Ti == 25.0
    assert Pcu == 0.0


def test_solve_increasing_current_increases_temp():
    """Higher current must always give higher winding temperature."""
    args = dict(
        P_iron_W=2.0,
        R_phase_20C=0.5,
        R_w_to_iron=0.5,
        R_iron_to_housing=0.5,
        R_housing_to_amb=0.5,
        T_ambient=25.0,
        copper_temp_coeff=0.00393,
    )
    Tw_1, _, _, _, _ = _solve_temperatures(I_phase_rms=1.0, **args)
    Tw_5, _, _, _, _ = _solve_temperatures(I_phase_rms=5.0, **args)
    Tw_10, _, _, _, _ = _solve_temperatures(I_phase_rms=10.0, **args)
    assert Tw_1 < Tw_5 < Tw_10


def test_solve_converges_in_reasonable_iterations():
    _, _, _, _, n = _solve_temperatures(
        I_phase_rms=5.0,
        P_iron_W=2.0,
        R_phase_20C=0.5,
        R_w_to_iron=0.5,
        R_iron_to_housing=0.5,
        R_housing_to_amb=0.5,
        T_ambient=25.0,
        copper_temp_coeff=0.00393,
    )
    assert 1 <= n < 30


def test_thermal_limit_torque_positive(
    synthetic_geometry, reasonable_constraints, materials_db
):
    d = _make_design(synthetic_geometry, reasonable_constraints, materials_db)
    d = characterize_thermal(d, synthetic_geometry, reasonable_constraints, materials_db)
    assert d.continuous_torque_thermal_limit_Nm > 0
