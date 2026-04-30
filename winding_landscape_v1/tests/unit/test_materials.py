"""Tests for the materials database."""

from __future__ import annotations

import math

import pytest


def test_load_succeeds(materials_db):
    assert len(materials_db.steels) > 0
    assert len(materials_db.magnets) > 0
    assert len(materials_db.insulation) > 0
    assert len(materials_db.awg) > 0


def test_default_grades_present(materials_db):
    assert "M19_29Ga" in materials_db.steels
    assert "N42SH" in materials_db.magnets
    assert "F" in materials_db.insulation


def test_awg_formula(materials_db):
    """Bare diameter must agree with d_mm = 0.127 * 92^((36-AWG)/39) within 1%."""
    for entry in materials_db.awg.values():
        expected = 0.127 * math.pow(92.0, (36.0 - entry.awg) / 39.0)
        rel_err = abs(entry.diameter_bare_mm - expected) / expected
        assert rel_err < 0.01, (
            f"AWG {entry.awg}: bare diameter {entry.diameter_bare_mm} disagrees with "
            f"formula prediction {expected:.4f} by {rel_err * 100:.2f}%"
        )


def test_awg_area_consistency(materials_db):
    """area_bare_mm2 must equal pi/4 * diameter_bare_mm^2 within 1%."""
    for entry in materials_db.awg.values():
        expected_area = math.pi / 4.0 * entry.diameter_bare_mm**2
        rel_err = abs(entry.area_bare_mm2 - expected_area) / expected_area
        assert rel_err < 0.01, (
            f"AWG {entry.awg}: area {entry.area_bare_mm2} != "
            f"pi/4*d^2 = {expected_area:.4f} (rel_err={rel_err*100:.2f}%)"
        )


def test_awg_insulated_larger_than_bare(materials_db):
    for entry in materials_db.awg.values():
        assert entry.diameter_insulated_mm > entry.diameter_bare_mm, (
            f"AWG {entry.awg}: insulated diameter not > bare diameter"
        )
        # And by a sane amount: typical magnet wire build is 50-150 um.
        d_insul = entry.diameter_insulated_mm - entry.diameter_bare_mm
        assert 0.02 < d_insul < 0.30, (
            f"AWG {entry.awg}: insulation thickness {d_insul:.3f} mm seems wrong"
        )


def test_copper_resistivity_temp_correction(materials_db):
    """Cu rho at 100 C should be ~ 1.724e-8 * (1 + 0.00393 * 80) ~ 2.266e-8."""
    rho_100 = materials_db.copper_resistivity_at(100.0)
    expected = 1.724e-8 * (1.0 + 0.00393 * 80.0)
    assert abs(rho_100 - expected) / expected < 1e-6


def test_magnet_br_temp_correction(materials_db):
    """N42SH B_r at 100 C should be lower than at 20 C by about (1 - 0.0012*80) = 90.4%."""
    n42sh = materials_db.magnets["N42SH"]
    Br_100 = n42sh.B_r_at_temp(100.0)
    Br_20 = n42sh.B_r_at_20C_T
    ratio = Br_100 / Br_20
    expected = 1.0 + n42sh.B_r_temp_coefficient_per_C * 80.0
    assert abs(ratio - expected) < 1e-6


def test_steel_BH_monotonic(materials_db):
    """The BH curve should be monotonically increasing (no field reversals in DC magnetization)."""
    for steel in materials_db.steels.values():
        H = steel.BH_curve[:, 0]
        B = steel.BH_curve[:, 1]
        assert all(H[i + 1] >= H[i] for i in range(len(H) - 1)), (
            f"Steel {steel.name}: H values not monotonic"
        )
        assert all(B[i + 1] >= B[i] for i in range(len(B) - 1)), (
            f"Steel {steel.name}: B values not monotonic"
        )


def test_steel_h_at_b_lookup(materials_db):
    """Inverse BH lookup should be reasonable -- and monotonic in B."""
    m19 = materials_db.steels["M19_29Ga"]
    H_at_1T = m19.H_at_B(1.0)
    H_at_15T = m19.H_at_B(1.5)
    H_at_18T = m19.H_at_B(1.8)
    assert H_at_1T < H_at_15T < H_at_18T
    # At deep saturation the field should be very high (thousands of A/m).
    assert m19.H_at_B(2.5) > 50_000


def test_get_awg_raises_on_missing(materials_db):
    with pytest.raises(KeyError):
        materials_db.get_awg(999)
