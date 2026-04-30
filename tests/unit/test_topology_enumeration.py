"""Tests for Stage 2: topology enumeration."""

from __future__ import annotations

from winding_landscape.topology.enumeration import enumerate_topologies


def test_enumeration_runs(synthetic_geometry, reasonable_constraints):
    tops = enumerate_topologies(synthetic_geometry, reasonable_constraints)
    assert len(tops) > 0


def test_all_balanced(synthetic_geometry, reasonable_constraints):
    """Stage 2 must filter out unbalanced topologies."""
    tops = enumerate_topologies(synthetic_geometry, reasonable_constraints)
    for t in tops:
        assert t.is_balanced


def test_all_meet_kw1_floor(synthetic_geometry, reasonable_constraints):
    floor = reasonable_constraints.enumeration_density.min_winding_factor_fundamental
    for t in enumerate_topologies(synthetic_geometry, reasonable_constraints):
        assert t.winding_factor_fundamental >= floor


def test_sorted_by_score_desc(synthetic_geometry, reasonable_constraints):
    tops = enumerate_topologies(synthetic_geometry, reasonable_constraints)
    scores = [t.topology_score for t in tops]
    assert scores == sorted(scores, reverse=True)


def test_q12_includes_classic_combos(synthetic_geometry, reasonable_constraints):
    """Q=12 should produce the classic 12s/10p, 12s/8p, 12s/4p combos."""
    tops = enumerate_topologies(synthetic_geometry, reasonable_constraints)
    poles_seen = {t.pole_count for t in tops}
    assert 10 in poles_seen   # 12s/10p (gold standard concentrated)
    assert 8 in poles_seen    # 12s/8p
    assert 4 in poles_seen    # 12s/4p


def test_max_topologies_respected(synthetic_geometry, reasonable_constraints):
    capped = reasonable_constraints.model_copy(
        update={
            "enumeration_density": reasonable_constraints.enumeration_density.model_copy(
                update={"max_topologies": 3}
            )
        }
    )
    tops = enumerate_topologies(synthetic_geometry, capped)
    assert len(tops) <= 3


def test_topology_id_format(synthetic_geometry, reasonable_constraints):
    tops = enumerate_topologies(synthetic_geometry, reasonable_constraints)
    for t in tops:
        tid = t.topology_id()
        assert tid.startswith("Q")
        assert "_p" in tid and "_L" in tid and "_y" in tid
