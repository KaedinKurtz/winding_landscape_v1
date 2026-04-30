"""Tests for Stage 3: winding (turns x gauge) enumeration and fit logic."""

from __future__ import annotations

from winding_landscape.topology.enumeration import enumerate_topologies
from winding_landscape.winding.enumeration import enumerate_windings


def test_enumeration_produces_candidates(
    synthetic_geometry, reasonable_constraints, materials_db
):
    tops = enumerate_topologies(synthetic_geometry, reasonable_constraints)
    assert tops
    ws = enumerate_windings(tops[0], synthetic_geometry, reasonable_constraints, materials_db)
    assert len(ws) > 0


def test_fit_status_categories(synthetic_geometry, reasonable_constraints, materials_db):
    """fit_status takes only one of three values."""
    tops = enumerate_topologies(synthetic_geometry, reasonable_constraints)
    ws = enumerate_windings(tops[0], synthetic_geometry, reasonable_constraints, materials_db)
    for w in ws:
        assert w.fit_status in {"fits", "no_fit_area", "no_fit_opening"}


def test_fit_implies_low_fill(synthetic_geometry, reasonable_constraints, materials_db):
    """Designs flagged as 'fits' must have fill below the manufacturing limit."""
    tops = enumerate_topologies(synthetic_geometry, reasonable_constraints)
    ws = enumerate_windings(tops[0], synthetic_geometry, reasonable_constraints, materials_db)
    fill_limit = reasonable_constraints.manufacturing_constraints.achievable_fill_factor_random
    for w in ws:
        if w.fit_status == "fits":
            # Actual fill is the fraction of slot useful area occupied; it must
            # not exceed the limit (with the small numerical buffer the limit
            # already includes).
            assert w.slot_fill_factor_actual <= fill_limit + 1e-9


def test_no_fit_area_implies_high_fill(synthetic_geometry, reasonable_constraints, materials_db):
    tops = enumerate_topologies(synthetic_geometry, reasonable_constraints)
    ws = enumerate_windings(tops[0], synthetic_geometry, reasonable_constraints, materials_db)
    fill_limit = reasonable_constraints.manufacturing_constraints.achievable_fill_factor_random
    for w in ws:
        if w.fit_status == "no_fit_area":
            assert w.slot_fill_factor_actual > fill_limit


def test_no_fit_opening_implies_wire_too_thick(
    synthetic_geometry, reasonable_constraints, materials_db
):
    tops = enumerate_topologies(synthetic_geometry, reasonable_constraints)
    ws = enumerate_windings(tops[0], synthetic_geometry, reasonable_constraints, materials_db)
    for w in ws:
        if w.fit_status == "no_fit_opening":
            assert w.wire_diameter_insulated_mm > 0.95 * synthetic_geometry.slot_opening_width_mm - 1e-9


def test_resistance_positive(synthetic_geometry, reasonable_constraints, materials_db):
    tops = enumerate_topologies(synthetic_geometry, reasonable_constraints)
    ws = enumerate_windings(tops[0], synthetic_geometry, reasonable_constraints, materials_db)
    for w in ws:
        assert w.R_phase_ohm_20C > 0


def test_resistance_scales_with_turns(synthetic_geometry, reasonable_constraints, materials_db):
    """At fixed gauge, R should scale linearly with turns."""
    tops = enumerate_topologies(synthetic_geometry, reasonable_constraints)
    ws = enumerate_windings(tops[0], synthetic_geometry, reasonable_constraints, materials_db)
    # Group by gauge.
    by_gauge: dict[int, list] = {}
    for w in ws:
        by_gauge.setdefault(w.wire_gauge_AWG, []).append(w)
    for gauge, lst in by_gauge.items():
        if len(lst) < 3:
            continue
        lst.sort(key=lambda w: w.turns_per_coil)
        r_per_turn = [w.R_phase_ohm_20C / w.turns_per_coil for w in lst]
        # All r-per-turn should be the same (within float tolerance) at fixed gauge.
        assert max(r_per_turn) - min(r_per_turn) < 1e-9 * max(r_per_turn)


def test_copper_mass_positive(synthetic_geometry, reasonable_constraints, materials_db):
    tops = enumerate_topologies(synthetic_geometry, reasonable_constraints)
    ws = enumerate_windings(tops[0], synthetic_geometry, reasonable_constraints, materials_db)
    for w in ws:
        assert w.total_copper_mass_g > 0
