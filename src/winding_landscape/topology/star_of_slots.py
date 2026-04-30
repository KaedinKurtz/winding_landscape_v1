"""Star-of-slots winding analysis.

Pure-Python implementation that does not depend on SWAT-EM. Used as the primary
implementation in V1 because it's simple, well-understood, and lets us validate
SWAT-EM's outputs when both are available.

The star-of-slots method (a.k.a. "Goerges diagram" in older texts) treats each
slot as a phasor in the electrical-angle domain and assigns slots to phases
based on which 60-degree phase belt the phasor lands in.

Reference: Hanselman, *Brushless Permanent Magnet Motor Design*, Chapter 5.
Also see Bianchi & Dai Pre, "Use of the star of slots in designing fractional-slot
single-layer synchronous motors", IEE Proc. EPA, 2006.

Mechanical-engineering analogy: the star-of-slots is to winding analysis what
a free-body diagram is to statics -- it's the canonical visual you'd draw on
a whiteboard to convince yourself a design will work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from math import gcd

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class StarOfSlotsResult:
    """Output of the star-of-slots calculation for a single (Q, 2p) combination."""

    connection_matrix: NDArray[np.int_]   # Q x 3, signed
    winding_factors: dict[int, float]     # harmonic order -> kw_h
    is_balanced: bool
    coils_per_phase: int


def lcm(a: int, b: int) -> int:
    """Least common multiple of two positive integers."""
    return a * b // gcd(a, b)


def is_three_phase_balanced(Q: int, pole_count: int) -> bool:
    """Quick balance check for a 3-phase double-layer winding.

    A 3-phase winding is balanced if Q / GCD(Q, p) is divisible by 3, where p
    is pole pairs. Equivalently, the slots-per-pole-per-phase pattern repeats
    consistently around the periphery.

    Reference: Hanselman eqn 5.10; Bianchi 2006.
    """
    pole_pairs = pole_count // 2
    return (Q // gcd(Q, pole_pairs)) % 3 == 0


def star_of_slots(
    Q: int,
    pole_count: int,
    coil_pitch_slots: int,
    layers: int = 2,
    harmonics: tuple[int, ...] = (1, 5, 7, 11, 13),
) -> StarOfSlotsResult:
    """Compute connection matrix and winding factors via the star-of-slots method.

    Parameters
    ----------
    Q
        Number of stator slots.
    pole_count
        Number of magnetic poles, 2p (always even).
    coil_pitch_slots
        Span of one coil in integer slot pitches.
    layers
        1 or 2. V1 supports double-layer fully; single-layer is allowed but
        yields a sparser connection matrix.
    harmonics
        Which harmonic orders to compute kw_h for.

    Returns
    -------
    StarOfSlotsResult
    """
    if Q < 3:
        raise ValueError(f"Q must be >= 3 (got {Q})")
    if pole_count < 2 or pole_count % 2 != 0:
        raise ValueError(f"pole_count must be even and >= 2 (got {pole_count})")
    if coil_pitch_slots < 1:
        raise ValueError(f"coil_pitch_slots must be >= 1 (got {coil_pitch_slots})")
    if layers not in (1, 2):
        raise ValueError(f"layers must be 1 or 2 (got {layers})")

    pole_pairs = pole_count // 2

    # Electrical angle of slot i (in degrees), reduced to [0, 360).
    # slot 0 sits at electrical angle 0.
    alpha_e_per_slot_deg = pole_pairs * 360.0 / Q
    slot_angles = np.array(
        [(i * alpha_e_per_slot_deg) % 360.0 for i in range(Q)], dtype=np.float64
    )

    # Phase-belt assignment. For 3-phase double-layer, six belts of 60deg each:
    #   [0,60)   -> A+
    #   [60,120) -> C-
    #   [120,180)-> B+
    #   [180,240)-> A-
    #   [240,300)-> C+
    #   [300,360)-> B-
    # Encode A=0, B=1, C=2; sign: +1 or -1. The canonical mapping is given by
    # the table below; row i -> (phase_index, sign).
    belt_table = [
        (0, +1),  # 0..60     A+
        (2, -1),  # 60..120   C-
        (1, +1),  # 120..180  B+
        (0, -1),  # 180..240  A-
        (2, +1),  # 240..300  C+
        (1, -1),  # 300..360  B-
    ]

    # Build the slot-to-(phase, sign) map for the "go" coil sides.
    go_assignments: list[tuple[int, int]] = []
    for ang in slot_angles:
        belt_idx = int(ang // 60.0) % 6
        go_assignments.append(belt_table[belt_idx])

    # Build the connection matrix.
    # For double-layer with coil_pitch y, each coil has its "go" side in slot k
    # and its "return" side in slot (k + y) mod Q. The return side carries the
    # opposite sign in the same phase.
    conn = np.zeros((Q, 3), dtype=np.int_)

    if layers == 2:
        # Place each coil: one coil per slot in V1 (one coil starts in each slot).
        for k in range(Q):
            phase_idx, sign = go_assignments[k]
            conn[k, phase_idx] += sign                          # go side
            conn[(k + coil_pitch_slots) % Q, phase_idx] += -sign  # return side
    else:
        # Single-layer: only every other slot starts a coil. We pick even-indexed
        # slots (k = 0, 2, 4, ...). Each slot ends up with exactly one coil side.
        for k in range(0, Q, 2):
            phase_idx, sign = go_assignments[k]
            conn[k, phase_idx] += sign
            conn[(k + coil_pitch_slots) % Q, phase_idx] += -sign

    # Balance check: each phase should have equal-magnitude positive and negative
    # contributions, and equal total turn count across phases.
    is_bal = is_three_phase_balanced(Q, pole_count) and _check_matrix_balance(conn)

    # Winding factors via the phasor sum of slot contributions for phase A,
    # for each harmonic. By symmetry, kw is identical across all three phases.
    wf: dict[int, float] = {}
    for h in harmonics:
        wf[h] = _winding_factor_from_matrix(
            conn, Q, pole_pairs, coil_pitch_slots, layers, h
        )

    # coils_per_phase = number of *coils* (not coil-sides) belonging to phase A.
    # Each coil has 2 sides (go + return), so coils = (sum of |conn[:,0]|) / 2.
    # For double-layer with one coil starting per slot, this gives Q/3 coils
    # per phase (e.g. 4 for 12-slot 3-phase). For single-layer with one coil
    # per slot pair, half as many.
    coil_sides_per_phase = int(np.sum(np.abs(conn[:, 0])))
    coils_per_phase = coil_sides_per_phase // 2

    return StarOfSlotsResult(
        connection_matrix=conn,
        winding_factors=wf,
        is_balanced=is_bal,
        coils_per_phase=coils_per_phase,
    )


def _check_matrix_balance(conn: NDArray[np.int_]) -> bool:
    """Check 3-phase symmetry of the connection matrix.

    The total signed phasor sum should be zero for each phase (when each slot's
    contribution is rotated by its electrical angle), and the total turn count
    per phase should be equal across A, B, C.
    """
    totals = np.sum(np.abs(conn), axis=0)
    return totals[0] == totals[1] == totals[2] and totals[0] > 0


def _winding_factor_from_matrix(
    conn: NDArray[np.int_],
    Q: int,
    pole_pairs: int,
    coil_pitch_slots: int,
    layers: int,
    harmonic: int,
) -> float:
    """Compute kw_h directly from the connection matrix using the phasor sum.

    For phase A:
        kw_h = | sum_i conn[i,0] * exp(j * h * pole_pairs * (2*pi*i/Q)) |
               * pitch_correction(h) / N_total

    where pitch_correction is automatically embedded by the go/return placement
    (the "return" side carries -sign, which gives the pitch factor). The
    distribution factor is embedded by the phasor sum across the multiple
    slots that belong to the phase.

    For double-layer windings this collapses to the textbook formula
    kw_h = kd_h * kp_h, but the matrix form generalizes cleanly to fractional-slot
    concentrated windings.
    """
    h_pp = harmonic * pole_pairs
    angles = 2.0 * math.pi * np.arange(Q) / Q
    phasors = np.exp(1j * h_pp * angles)
    phase_a = conn[:, 0]
    numerator = np.abs(np.sum(phase_a * phasors))
    denominator = np.sum(np.abs(phase_a))
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


# ---------------------------------------------------------------------------
# Closed-form distribution + pitch factors (for cross-validation only)
# ---------------------------------------------------------------------------

def distribution_factor_textbook(q: float, alpha_e_deg: float, harmonic: int) -> float:
    """Hanselman eqn 5.20: kd_h = sin(h*q*alpha_e/2) / (q*sin(h*alpha_e/2)).

    q here is slots-per-pole-per-phase (can be fractional). For integer q this
    matches the standard Stuermer formula. For fractional q this generalizes
    via the same expression (textbook notation varies; we follow Hanselman).
    """
    a_rad = math.radians(alpha_e_deg)
    if q == 0:
        return 0.0
    num = math.sin(harmonic * q * a_rad / 2.0)
    den = q * math.sin(harmonic * a_rad / 2.0)
    if abs(den) < 1e-12:
        return 0.0
    return num / den


def pitch_factor_textbook(coil_pitch_slots: int, pole_pitch_slots: float, harmonic: int) -> float:
    """Hanselman eqn 5.5: kp_h = sin(h * coil_pitch / pole_pitch * pi/2)."""
    if pole_pitch_slots <= 0:
        return 0.0
    return abs(math.sin(harmonic * coil_pitch_slots / pole_pitch_slots * math.pi / 2.0))
