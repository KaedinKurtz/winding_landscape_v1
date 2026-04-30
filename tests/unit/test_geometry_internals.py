"""Tests for Stage 1 internal helpers: _measure_slots and _verify_periodicity.

Uses synthetic 2D boundary point clouds rather than real BREP cross-sections,
which lets these tests run without build123d. The synthetic data mimics what
build123d's section operation produces for known stator geometries.
"""

from __future__ import annotations

import math

import numpy as np

from winding_landscape.geometry.extraction import (
    _PlanarSection,
    _identify_bore_and_outer_radii,
    _measure_slots,
    _verify_periodicity,
)


def _synthetic_outer_boundary(
    Q: int,
    outer_r: float,
    slot_bottom_r: float,
    slot_opening_at_OD: float,
    n_points_per_slot_pitch: int = 200,
) -> np.ndarray:
    """Build an Nx2 array of points tracing the outer boundary of a slotted stator."""
    n_total = Q * n_points_per_slot_pitch
    theta = np.linspace(-math.pi, math.pi, n_total, endpoint=False)
    r = np.full(n_total, outer_r)

    slot_pitch_rad = 2 * math.pi / Q
    opening_half_angle_rad = (slot_opening_at_OD / outer_r) / 2.0
    chamfer_width = opening_half_angle_rad * 0.3

    for k in range(Q):
        theta_c = -math.pi + (k + 0.5) * slot_pitch_rad
        for i, t in enumerate(theta):
            dt = ((t - theta_c + math.pi) % (2 * math.pi)) - math.pi
            adt = abs(dt)
            if adt < opening_half_angle_rad:
                if adt > opening_half_angle_rad - chamfer_width:
                    frac = (opening_half_angle_rad - adt) / chamfer_width
                    r[i] = outer_r - frac * (outer_r - slot_bottom_r)
                else:
                    r[i] = slot_bottom_r

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])


def _synthetic_bore(bore_r: float, n_points: int = 360) -> np.ndarray:
    """Build a circular bore polygon."""
    theta = np.linspace(0, 2 * math.pi, n_points, endpoint=False)
    return np.column_stack([bore_r * np.cos(theta), bore_r * np.sin(theta)])


def _make_section(
    Q: int = 36,
    outer_r: float = 50.0,
    slot_bottom_r: float = 42.4,
    slot_opening_at_OD: float = 2.6,
    bore_r: float = 39.0,
    noise_mm: float = 0.0,
    n_points_per_slot_pitch: int = 200,
) -> _PlanarSection:
    outer = _synthetic_outer_boundary(
        Q, outer_r, slot_bottom_r, slot_opening_at_OD, n_points_per_slot_pitch
    )
    bore = _synthetic_bore(bore_r)
    if noise_mm > 0:
        rng = np.random.default_rng(42)
        outer = outer + rng.normal(0, noise_mm, outer.shape)
        bore = bore + rng.normal(0, noise_mm, bore.shape)
    return _PlanarSection(
        polygons=[bore, outer],
        plane_origin=np.zeros(3),
        u=np.array([1.0, 0.0, 0.0]),
        v=np.array([0.0, 1.0, 0.0]),
    )


# ---------- bore + OD identification ----------

def test_identify_bore_and_outer_clean():
    sec = _make_section(Q=36, outer_r=50.0, bore_r=39.0)
    bore_r, outer_r = _identify_bore_and_outer_radii(sec)
    assert abs(bore_r - 39.0) < 0.01
    assert abs(outer_r - 50.0) < 0.01


def test_identify_bore_takes_minimum_inner_radius():
    """If the section's inner side has features at multiple radii, we pick the smallest.

    Real stators sometimes have slot tooth-tips at one radius and the bore
    proper at a smaller radius. The dial-indicator answer is the minimum.
    """
    sec = _make_section(Q=36, outer_r=50.0, bore_r=39.0)
    # Add a polygon with mid-range radii (e.g., slot-tip arcs at r=40)
    extra = np.column_stack([
        40.0 * np.cos(np.linspace(0, 2 * math.pi, 25, endpoint=False)),
        40.0 * np.sin(np.linspace(0, 2 * math.pi, 25, endpoint=False)),
    ])
    sec.polygons.append(extra)
    bore_r, outer_r = _identify_bore_and_outer_radii(sec)
    assert abs(bore_r - 39.0) < 0.01, f"bore should be 39, got {bore_r}"


# ---------- slot counting ----------

def test_slot_count_36_clean():
    sec = _make_section(Q=36)
    bore_r, outer_r = _identify_bore_and_outer_radii(sec)
    Q, _ = _measure_slots(sec, bore_r, outer_r)
    assert Q == 36


def test_slot_count_36_with_tessellation_noise():
    """Resists the kind of arc-tessellation noise build123d produces."""
    sec = _make_section(Q=36, noise_mm=0.05)
    bore_r, outer_r = _identify_bore_and_outer_radii(sec)
    Q, _ = _measure_slots(sec, bore_r, outer_r)
    assert Q == 36


def test_slot_count_12():
    sec = _make_section(Q=12, outer_r=40.0, slot_bottom_r=32.0,
                        slot_opening_at_OD=3.0, bore_r=20.0)
    bore_r, outer_r = _identify_bore_and_outer_radii(sec)
    Q, _ = _measure_slots(sec, bore_r, outer_r)
    assert Q == 12


def test_slot_count_24():
    sec = _make_section(Q=24, outer_r=60.0, slot_bottom_r=50.0,
                        slot_opening_at_OD=2.5, bore_r=30.0)
    bore_r, outer_r = _identify_bore_and_outer_radii(sec)
    Q, _ = _measure_slots(sec, bore_r, outer_r)
    assert Q == 24


# ---------- slot dimensions ----------

def test_slot_dimensions_match_design():
    """For a 36-slot stator with known geometry, recovered dimensions match.

    Tolerance accounts for the discrete bin width of the detection algorithm
    and the linear chamfer approximation in the synthetic boundary.
    """
    sec = _make_section(
        Q=36, outer_r=50.0, slot_bottom_r=42.4,
        slot_opening_at_OD=2.6, bore_r=39.0,
    )
    bore_r, outer_r = _identify_bore_and_outer_radii(sec)
    Q, features = _measure_slots(sec, bore_r, outer_r)

    assert Q == 36
    assert abs(features["opening_width_mm"] - 2.6) < 0.1, (
        f"opening_width: {features['opening_width_mm']:.3f}, expected ~2.6"
    )
    assert abs(features["total_depth_mm"] - 7.6) < 0.2, (
        f"total_depth: {features['total_depth_mm']:.3f}, expected ~7.6"
    )
    # Yoke = bottom_r - bore_r = 42.4 - 39.0 = 3.4
    assert abs(features["yoke_thickness_mm"] - 3.4) < 0.1


# ---------- periodicity ----------

def test_periodicity_clean_stator():
    """A perfectly periodic stator has near-zero periodicity deviation."""
    sec = _make_section(Q=36)
    bore_r, outer_r = _identify_bore_and_outer_radii(sec)
    dev = _verify_periodicity(sec, slot_count=36, bore_r=bore_r, outer_r=outer_r)
    assert dev < 0.01, f"deviation should be near zero, got {dev:.3f}"


def test_periodicity_wrong_slot_count_gives_large_deviation():
    """If we claim Q=35 for an actually-36-slot stator, deviation is large."""
    sec = _make_section(Q=36)
    bore_r, outer_r = _identify_bore_and_outer_radii(sec)
    dev = _verify_periodicity(sec, slot_count=35, bore_r=bore_r, outer_r=outer_r)
    assert dev > 1.0, f"wrong Q should give large deviation, got {dev:.3f}"


def test_periodicity_with_tessellation_noise_stays_under_relative_tolerance():
    """Realistic CAD tessellation noise (~0.03 mm) shouldn't push past 50% of slot depth."""
    sec = _make_section(Q=36, noise_mm=0.03)
    bore_r, outer_r = _identify_bore_and_outer_radii(sec)
    Q, features = _measure_slots(sec, bore_r, outer_r)
    dev = _verify_periodicity(sec, slot_count=Q, bore_r=bore_r, outer_r=outer_r)
    # Fatal bound is 50% of slot depth, so deviation should comfortably stay below that
    assert dev < 0.5 * features["total_depth_mm"]
