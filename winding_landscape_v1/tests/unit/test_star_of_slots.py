"""Star-of-slots: canonical winding-factor cases.

These reference values come from textbook examples and SWAT-EM output. If any
of these regress, something is structurally wrong in the analyzer -- everything
downstream depends on these being correct.
"""

from __future__ import annotations

import pytest

from winding_landscape.topology.star_of_slots import (
    is_three_phase_balanced,
    lcm,
    star_of_slots,
)


# Canonical (Q, p, y, expected_kw1) values from Hanselman, Bianchi, and SWAT-EM.
@pytest.mark.parametrize(
    "Q, pole_count, coil_pitch, expected_kw1",
    [
        (12, 10, 1, 0.933),  # classic 12s/10p concentrated
        (24, 22, 1, 0.949),  # 24s/22p concentrated, very high kw1
        (36, 8, 4, 0.945),   # 36s/8p distributed, near-pole-pitch
        (36, 4, 9, 0.960),   # 36s/4p full-pitch (q=3)
        (12, 8, 1, 0.866),   # 12s/8p concentrated (q=1/2)
        (9, 6, 1, 0.866),    # 9s/6p (q=1/2 fractional)
        (6, 4, 1, 0.866),    # 6s/4p concentrated
    ],
)
def test_canonical_kw1(Q, pole_count, coil_pitch, expected_kw1):
    res = star_of_slots(Q=Q, pole_count=pole_count, coil_pitch_slots=coil_pitch, layers=2)
    assert res.is_balanced, (
        f"Q={Q} p={pole_count} y={coil_pitch} reported as unbalanced; "
        f"expected balanced based on textbook reference."
    )
    assert abs(res.winding_factors[1] - expected_kw1) < 0.002, (
        f"kw1 mismatch for Q={Q} p={pole_count} y={coil_pitch}: "
        f"got {res.winding_factors[1]:.4f}, expected ~{expected_kw1}"
    )


def test_kw_bounded():
    """All winding factors must be in [0, 1]."""
    for Q, p in [(12, 10), (24, 22), (36, 8), (12, 8), (15, 4), (18, 8)]:
        res = star_of_slots(Q=Q, pole_count=p, coil_pitch_slots=1, layers=2)
        for h, kw in res.winding_factors.items():
            assert 0.0 <= kw <= 1.0, (
                f"kw_{h} out of [0,1] for Q={Q} p={p}: {kw}"
            )


def test_concentrated_winding_q_half_has_kw_866():
    """Any q=1/2 fractional-slot concentrated winding has kw1 = sqrt(3)/2 ~ 0.866."""
    cases = [(6, 4), (9, 6), (12, 8), (15, 10), (18, 12)]
    for Q, p in cases:
        res = star_of_slots(Q=Q, pole_count=p, coil_pitch_slots=1, layers=2)
        if not res.is_balanced:
            continue
        assert abs(res.winding_factors[1] - 0.866) < 0.002, (
            f"q=1/2 case Q={Q} p={p} should have kw1=0.866, got {res.winding_factors[1]}"
        )


def test_balance_check():
    assert is_three_phase_balanced(12, 10) is True
    assert is_three_phase_balanced(24, 22) is True
    assert is_three_phase_balanced(12, 8) is True   # q=1/2 fractional
    assert is_three_phase_balanced(9, 8) is True    # q=3/8 fractional, balanced
    # Truly unbalanced: Q/gcd(Q,p) not divisible by 3.
    # Q=8, p=2 (pole_pairs=1): Q/gcd(8,1) = 8, 8 % 3 != 0 -> unbalanced.
    assert is_three_phase_balanced(8, 2) is False


def test_connection_matrix_signs_balance():
    """Sum of signed contributions must be zero per phase (no DC offset)."""
    res = star_of_slots(Q=12, pole_count=10, coil_pitch_slots=1, layers=2)
    for phase_col in range(3):
        column_sum = int(res.connection_matrix[:, phase_col].sum())
        assert column_sum == 0, (
            f"Phase {phase_col} signed sum is {column_sum}, expected 0 (no DC)"
        )


def test_lcm():
    assert lcm(12, 10) == 60
    assert lcm(24, 22) == 264
    assert lcm(36, 4) == 36


def test_input_validation():
    with pytest.raises(ValueError):
        star_of_slots(Q=2, pole_count=4, coil_pitch_slots=1)  # Q too small
    with pytest.raises(ValueError):
        star_of_slots(Q=12, pole_count=5, coil_pitch_slots=1)  # odd pole count
    with pytest.raises(ValueError):
        star_of_slots(Q=12, pole_count=10, coil_pitch_slots=0)  # zero pitch
    with pytest.raises(ValueError):
        star_of_slots(Q=12, pole_count=10, coil_pitch_slots=1, layers=3)  # bad layers
