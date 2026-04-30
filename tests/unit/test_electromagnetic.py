"""Tests for Stage 4: electromagnetic characterization."""

from __future__ import annotations

import math

import pytest

from winding_landscape.performance.electromagnetic import (
    _carter_coefficient,
    _saturation_factor,
    _steinmetz_loss,
    characterize_electromagnetic,
)
from winding_landscape.topology.enumeration import enumerate_topologies
from winding_landscape.winding.enumeration import enumerate_windings


# ---------- Carter's coefficient ----------

def test_carter_no_slots_gives_unity():
    """If slot_opening = 0, Carter's coefficient must be exactly 1.0."""
    kc = _carter_coefficient(slot_opening_mm=0.0, airgap_mm=1.0, slot_pitch_mm=10.0)
    assert kc == pytest.approx(1.0)


def test_carter_increases_with_opening():
    """Wider slot openings -> larger effective airgap correction."""
    kc1 = _carter_coefficient(slot_opening_mm=1.0, airgap_mm=1.0, slot_pitch_mm=10.0)
    kc2 = _carter_coefficient(slot_opening_mm=3.0, airgap_mm=1.0, slot_pitch_mm=10.0)
    assert 1.0 < kc1 < kc2


def test_carter_typical_values():
    """For a typical motor (s/g=4, tau/g=10), kc ~ 1.2-1.4."""
    kc = _carter_coefficient(slot_opening_mm=2.0, airgap_mm=0.5, slot_pitch_mm=5.0)
    assert 1.1 < kc < 1.6


# ---------- Saturation factor ----------

def test_saturation_no_derate_below_knee(materials_db):
    m19 = materials_db.steels["M19_29Ga"]
    f = _saturation_factor(B_tooth_T=1.0, steel=m19)
    assert f == 1.0


def test_saturation_floors_above_limit(materials_db):
    m19 = materials_db.steels["M19_29Ga"]
    f = _saturation_factor(B_tooth_T=2.5, steel=m19)
    assert f == pytest.approx(0.7)


def test_saturation_monotonic(materials_db):
    m19 = materials_db.steels["M19_29Ga"]
    Bs = [1.0, 1.5, 1.7, 1.9, 2.0, 2.1, 2.2, 2.5]
    factors = [_saturation_factor(B, m19) for B in Bs]
    for i in range(len(factors) - 1):
        assert factors[i + 1] <= factors[i] + 1e-9


# ---------- Steinmetz loss ----------

def test_steinmetz_zero_at_zero_freq(materials_db):
    m19 = materials_db.steels["M19_29Ga"]
    P = _steinmetz_loss(m19, f_Hz=0.0, B_T=1.5, volume_m3=1e-3)
    assert P == 0.0


def test_steinmetz_increases_with_frequency(materials_db):
    m19 = materials_db.steels["M19_29Ga"]
    P_60 = _steinmetz_loss(m19, f_Hz=60.0, B_T=1.0, volume_m3=1e-3)
    P_400 = _steinmetz_loss(m19, f_Hz=400.0, B_T=1.0, volume_m3=1e-3)
    P_2000 = _steinmetz_loss(m19, f_Hz=2000.0, B_T=1.0, volume_m3=1e-3)
    assert 0 < P_60 < P_400 < P_2000


def test_steinmetz_increases_with_flux_density(materials_db):
    m19 = materials_db.steels["M19_29Ga"]
    P_05 = _steinmetz_loss(m19, f_Hz=400.0, B_T=0.5, volume_m3=1e-3)
    P_10 = _steinmetz_loss(m19, f_Hz=400.0, B_T=1.0, volume_m3=1e-3)
    P_15 = _steinmetz_loss(m19, f_Hz=400.0, B_T=1.5, volume_m3=1e-3)
    assert 0 < P_05 < P_10 < P_15


def test_steinmetz_typical_magnitude(materials_db):
    """At 400Hz, 1T, 1L of M19 should dissipate ~50 W (order-of-magnitude).

    M19 datasheet says ~5 W/lb at 60Hz/1.5T. At 400Hz/1T this is comparable
    in W/m^3, and 1L ~ 7.65 kg of steel -> tens of W."""
    m19 = materials_db.steels["M19_29Ga"]
    P = _steinmetz_loss(m19, f_Hz=400.0, B_T=1.0, volume_m3=1e-3)
    assert 1.0 < P < 1000.0  # very loose order of magnitude


# ---------- Full Stage 4 integration ----------

def test_em_outputs_populated(synthetic_geometry, reasonable_constraints, materials_db):
    tops = enumerate_topologies(synthetic_geometry, reasonable_constraints)
    ws = enumerate_windings(tops[0], synthetic_geometry, reasonable_constraints, materials_db)
    feasible = [w for w in ws if w.fit_status == "fits"]
    assert feasible, "Need at least one fitting design for this test"
    d = characterize_electromagnetic(
        feasible[0], synthetic_geometry, reasonable_constraints, materials_db
    )
    assert d.Ke_V_per_rad_per_s > 0
    assert d.Kt_Nm_per_A > 0
    assert d.Ls_synchronous_mH > 0
    assert d.peak_torque_at_max_current_Nm > 0
    assert d.flux_per_pole_Wb > 0
    assert d.tooth_flux_density_peak_T > 0


def test_kt_equals_1p5_ke(synthetic_geometry, reasonable_constraints, materials_db):
    """For sinusoidal vector control, Kt = 1.5 * Ke.

    This is the documented assumption in electromagnetic.py; ensures we don't
    accidentally regress to a different convention."""
    tops = enumerate_topologies(synthetic_geometry, reasonable_constraints)
    ws = enumerate_windings(tops[0], synthetic_geometry, reasonable_constraints, materials_db)
    feasible = [w for w in ws if w.fit_status == "fits"]
    d = characterize_electromagnetic(
        feasible[0], synthetic_geometry, reasonable_constraints, materials_db
    )
    assert math.isclose(d.Kt_Nm_per_A, 1.5 * d.Ke_V_per_rad_per_s, rel_tol=1e-9)


def test_ke_scales_linearly_with_turns(
    synthetic_geometry, reasonable_constraints, materials_db
):
    """Ke ~ N_total * kw1 * Phi * p, so Ke / turns_per_coil should be constant for one topology."""
    tops = enumerate_topologies(synthetic_geometry, reasonable_constraints)
    ws = enumerate_windings(tops[0], synthetic_geometry, reasonable_constraints, materials_db)
    fits = [w for w in ws if w.fit_status == "fits"]
    # Fix the gauge to remove confounding effects.
    if not fits:
        return
    target_gauge = fits[0].wire_gauge_AWG
    same_gauge = [w for w in fits if w.wire_gauge_AWG == target_gauge]
    if len(same_gauge) < 2:
        return
    ke_per_turn = [
        characterize_electromagnetic(w, synthetic_geometry, reasonable_constraints, materials_db).Ke_V_per_rad_per_s
        / w.turns_per_coil
        for w in same_gauge[:5]
    ]
    assert max(ke_per_turn) - min(ke_per_turn) < 1e-6 * max(ke_per_turn)
