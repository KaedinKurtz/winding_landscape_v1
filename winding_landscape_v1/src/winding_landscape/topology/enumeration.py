"""Stage 2: Topology enumeration.

Sweeps all (Q, 2p, layers, coil_pitch) combinations within the constraints,
filters out unbalanced or low-quality ones, and produces ranked
:class:`TopologyCandidate` objects.

Mechanical-engineering analogy: this is the "concept screening" step in a
gearbox design. You start with all kinematically valid ratio combinations,
prune the obvious losers (insufficient torque margin, unbalanced loading),
and rank the survivors by a heuristic before sizing them.
"""

from __future__ import annotations

import math
from fractions import Fraction
from math import gcd

from winding_landscape.config import Constraints
from winding_landscape.geometry.stator_geometry import StatorGeometry
from winding_landscape.topology.star_of_slots import lcm
from winding_landscape.topology.swat_em_wrapper import analyze_winding
from winding_landscape.topology.topology_candidate import TopologyCandidate
from winding_landscape.utils.logging_config import get_logger

logger = get_logger(__name__)


def enumerate_topologies(
    geometry: StatorGeometry,
    constraints: Constraints,
) -> list[TopologyCandidate]:
    """Enumerate viable topologies for the given stator + constraints.

    Returns
    -------
    list[TopologyCandidate]
        Sorted by topology_score descending. Truncated to
        ``constraints.enumeration_density.max_topologies`` entries.
    """
    Q = geometry.slot_count
    pole_lo, pole_hi = constraints.topology_constraints.pole_count_range

    # User-supplied list overrides the automatic sweep.
    if constraints.topology_constraints.slot_pole_combos_to_consider != "auto":
        explicit = constraints.topology_constraints.slot_pole_combos_to_consider
        assert isinstance(explicit, list)
        candidate_pole_counts = sorted(set(p for q, p in explicit if q == Q))
    else:
        candidate_pole_counts = list(range(max(2, pole_lo), pole_hi + 1, 2))

    candidates: list[TopologyCandidate] = []
    skipped_reasons: dict[str, int] = {}

    for pole_count in candidate_pole_counts:
        if pole_count > Q:
            _bump(skipped_reasons, "2p > Q")
            continue

        # Slots per pole per phase (can be fractional).
        q_slot = Fraction(Q, pole_count * 3)
        if q_slot.denominator > 8:
            _bump(skipped_reasons, "q denominator > 8")
            continue

        # Pole-pitch in slots (can be fractional).
        pole_pitch_slots_float = Q / pole_count

        # Coil-pitch sweep. For double-layer concentrated winding this is just 1;
        # for distributed windings we sweep around the integer-valued pole pitch.
        max_pitch = max(1, int(math.floor(pole_pitch_slots_float)) + 1)
        pitches_to_try = list(range(1, max_pitch + 1))

        for layers_str in constraints.topology_constraints.layer_options:
            layers = 2 if layers_str == "double" else 1

            for coil_pitch in pitches_to_try:
                # Coil pitch can't exceed slot count.
                if coil_pitch >= Q:
                    continue

                try:
                    result = analyze_winding(
                        Q=Q,
                        pole_count=pole_count,
                        coil_pitch_slots=coil_pitch,
                        layers=layers,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "analyze_winding failed for Q=%d p=%d y=%d L=%d: %s",
                        Q, pole_count, coil_pitch, layers, exc,
                    )
                    _bump(skipped_reasons, "analyzer error")
                    continue

                if not result.is_balanced:
                    _bump(skipped_reasons, "unbalanced")
                    continue

                kw1 = result.winding_factors[1]
                if kw1 < constraints.enumeration_density.min_winding_factor_fundamental:
                    _bump(skipped_reasons, f"kw1 < {constraints.enumeration_density.min_winding_factor_fundamental}")
                    continue

                # Determine valid parallel paths: a divides GCD(coils_per_phase,
                # number-of-pole-pairs) for typical wye connections. For V1 we
                # default to 1 path; the user can override downstream.
                pole_pairs = pole_count // 2
                parallel_paths = 1
                coils_per_phase = result.coils_per_phase
                # Number of coils in series per parallel path.
                if coils_per_phase % parallel_paths != 0:
                    parallel_paths = 1  # fall back
                coils_in_series_per_path = coils_per_phase // parallel_paths

                cogging_period = 360.0 / lcm(Q, pole_count)
                lcm_qp = lcm(Q, pole_count)
                # Heuristic ranking score (spec Section 5, Stage 2 step 3).
                score = (
                    kw1
                    - 0.3 * abs(result.winding_factors[5])
                    - 0.3 * abs(result.winding_factors[7])
                    - 0.1 * (abs(result.winding_factors[11]) + abs(result.winding_factors[13]))
                    + 0.5 * (1 - 1.0 / lcm_qp)
                )

                candidates.append(
                    TopologyCandidate(
                        Q=Q,
                        pole_count=pole_count,
                        layers=layers,
                        coil_pitch_slots=coil_pitch,
                        parallel_paths=parallel_paths,
                        connection_matrix=result.connection_matrix,
                        winding_factor_fundamental=kw1,
                        winding_factor_5th=result.winding_factors[5],
                        winding_factor_7th=result.winding_factors[7],
                        winding_factor_11th=result.winding_factors[11],
                        winding_factor_13th=result.winding_factors[13],
                        is_balanced=True,
                        cogging_period_mech_deg=cogging_period,
                        topology_score=float(score),
                        coils_per_phase=coils_per_phase,
                        coils_in_series_per_path=coils_in_series_per_path,
                    )
                )

    candidates.sort(key=lambda c: c.topology_score, reverse=True)

    # Deduplicate: prefer one (Q, p, layers) triple at a time -- if multiple
    # coil pitches produce candidates, keep them all (different y is a different
    # design), but ensure overall list is bounded.
    max_top = constraints.enumeration_density.max_topologies
    if len(candidates) > max_top:
        logger.info(
            "Trimming topology list from %d to %d (max_topologies cap).",
            len(candidates), max_top,
        )
        candidates = candidates[:max_top]

    if skipped_reasons:
        logger.info("Topology skip reasons: %s", dict(skipped_reasons))
    logger.info("Enumerated %d viable topologies for Q=%d.", len(candidates), Q)
    return candidates


def _bump(d: dict[str, int], key: str) -> None:
    d[key] = d.get(key, 0) + 1
