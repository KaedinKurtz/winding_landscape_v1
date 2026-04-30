"""Stage 3: Winding enumeration -- sweep (turns_per_coil, wire_gauge_AWG)
combinations and produce :class:`WindingCandidate` objects.

This stage is "constraint-aware": it computes the slot fit, opening fit, and
preliminary current density at each candidate point and uses these to prune
trivially-infeasible designs *before* spending time on Stage 4 EM analysis.
Infeasible designs are still emitted (with their fit_status flagged) because
they're useful downstream for "what-if-we-relaxed-this-constraint" analysis.
"""

from __future__ import annotations

import math

from winding_landscape.config import Constraints
from winding_landscape.geometry.stator_geometry import StatorGeometry
from winding_landscape.materials.database import MaterialsDatabase
from winding_landscape.topology.topology_candidate import TopologyCandidate
from winding_landscape.utils.logging_config import get_logger
from winding_landscape.winding.awg_table import filter_available_gauges, lookup_awg
from winding_landscape.winding.winding_candidate import WindingCandidate

logger = get_logger(__name__)

# Current density limits (A/mm^2) by cooling mode. Spec section 5, Stage 3.
_J_LIMIT_BY_COOLING = {
    "natural": 8.0,
    "forced_air": 15.0,
    "liquid": 25.0,
}

# Empirical end-turn length multiplier (spec section 5).
_END_TURN_FACTOR = 1.2

# Slot opening clearance: insulated wire OD must be < 0.95 * slot opening width
# to allow random winding through the opening.
_OPENING_CLEARANCE = 0.95


def enumerate_windings(
    topology: TopologyCandidate,
    geometry: StatorGeometry,
    constraints: Constraints,
    materials: MaterialsDatabase,
) -> list[WindingCandidate]:
    """Enumerate all (turns, gauge) candidates for a single topology.

    Returns
    -------
    list[WindingCandidate]
        Includes both feasible and infeasible designs (flagged via fit_status).
        Density-controlled per ``constraints.enumeration_density``.
    """
    mfg = constraints.manufacturing_constraints
    enum_cfg = constraints.enumeration_density

    # Available gauges, intersected with what's in the materials DB.
    gauges = filter_available_gauges(materials, mfg.available_wire_gauges_AWG)
    if not gauges:
        logger.warning("No AWG gauges available for topology %s", topology.topology_id())
        return []

    cooling_mode = constraints.thermal_envelope.cooling_mode
    j_limit = _J_LIMIT_BY_COOLING[cooling_mode]

    # Mean coil radius -- used for end-turn estimation.
    mean_coil_radius_mm = geometry.outer_radius_mm - geometry.slot_total_depth_mm / 2.0
    # End-turn arc length per coil side: 1.2 * (mean_radius * 2*pi * coil_pitch / Q).
    end_turn_per_side_mm = (
        _END_TURN_FACTOR
        * mean_coil_radius_mm
        * 2.0
        * math.pi
        * topology.coil_pitch_slots
        / topology.Q
    )

    # Conductors per slot. Double-layer: two coil sides per slot, each with
    # turns_per_coil conductors. Single-layer: one coil side per slot.
    coil_sides_per_slot = topology.layers

    # Fill-factor target.
    fill_limit = mfg.achievable_fill_factor_random
    available_copper_area = geometry.slot_useful_area_mm2 * fill_limit

    candidates: list[WindingCandidate] = []
    n_kept_per_gauge: dict[int, int] = {}

    # Outer loop: turns. Inner loop: gauge. This ordering means for any given
    # turn count we evaluate all gauges (we don't early-exit on fit, because
    # we want to capture the "no_fit_area" status for each).
    for turns_per_coil in range(
        mfg.min_turns_per_coil,
        mfg.max_turns_per_coil + 1,
        enum_cfg.turns_step,
    ):
        n_slot = coil_sides_per_slot * turns_per_coil

        for gauge in gauges:
            wire = lookup_awg(materials, gauge)
            d_ins = wire.diameter_insulated_mm

            # Required insulated copper area in the slot.
            a_required = n_slot * math.pi * (d_ins / 2.0) ** 2

            # Fit checks.
            fit_status = "fits"
            if a_required > available_copper_area:
                fit_status = "no_fit_area"
            elif d_ins > _OPENING_CLEARANCE * geometry.slot_opening_width_mm:
                fit_status = "no_fit_opening"

            slot_fill_actual = a_required / max(geometry.slot_useful_area_mm2, 1e-9)

            # Phase resistance at 20 C.
            R_phase = _phase_resistance_dc(
                turns_per_coil=turns_per_coil,
                coils_in_series_per_path=topology.coils_in_series_per_path,
                parallel_paths=topology.parallel_paths,
                stack_length_mm=geometry.stack_length_mm,
                end_turn_length_per_side_mm=end_turn_per_side_mm,
                wire_area_mm2=wire.area_bare_mm2,
                copper_resistivity_ohm_m=materials.copper_resistivity_at(20.0),
            )

            # Total copper mass per phase (single phase) summed over 3 phases.
            total_length_per_phase_m = (
                turns_per_coil
                * topology.coils_per_phase
                * 2.0
                * (geometry.stack_length_mm + end_turn_per_side_mm)
                / 1000.0
            )
            total_copper_mass_g = (
                3.0
                * total_length_per_phase_m
                * (wire.area_bare_mm2 * 1e-6)  # m^2
                * materials.copper_density_kg_per_m3
                * 1000.0  # to grams
            )

            # Provisional current density at the target continuous torque.
            # We estimate I_continuous via a back-of-envelope Kt that uses kw1
            # and a representative flux per pole. Stage 4 will recompute
            # everything precisely; here we just want a sanity flag.
            I_continuous_estimate = _estimate_continuous_current(
                turns_per_coil=turns_per_coil,
                coils_in_series_per_path=topology.coils_in_series_per_path,
                kw1=topology.winding_factor_fundamental,
                pole_pairs=topology.pole_count // 2,
                outer_radius_mm=geometry.outer_radius_mm,
                stack_length_mm=geometry.stack_length_mm,
                airgap_mm=geometry.airgap_mm,
                target_torque_Nm=constraints.operating_targets.target_continuous_torque_Nm,
                magnet_arc_fraction=constraints.topology_constraints.magnet_arc_fraction,
                magnet_thickness_mm=constraints.topology_constraints.magnet_thickness_mm,
            )
            j_value = I_continuous_estimate / max(wire.area_bare_mm2, 1e-9)

            candidates.append(
                WindingCandidate(
                    topology=topology,
                    turns_per_coil=turns_per_coil,
                    wire_gauge_AWG=gauge,
                    wire_diameter_bare_mm=wire.diameter_bare_mm,
                    wire_diameter_insulated_mm=wire.diameter_insulated_mm,
                    wire_area_mm2=wire.area_bare_mm2,
                    strands_per_conductor=1,
                    manufacturing_method="random_round",
                    slot_fill_factor_actual=slot_fill_actual,
                    slot_fill_factor_limit=fill_limit,
                    fit_status=fit_status,
                    end_turn_length_estimate_mm=end_turn_per_side_mm,
                    total_copper_mass_g=total_copper_mass_g,
                    R_phase_ohm_20C=R_phase,
                    current_density_at_target_continuous_A_per_mm2=j_value,
                )
            )
            if fit_status == "fits":
                n_kept_per_gauge[gauge] = n_kept_per_gauge.get(gauge, 0) + 1

    # Density control: if we have more than 3x target_designs_per_topology
    # feasible candidates, subsample. We keep all infeasibles (they're useful
    # for downstream what-if analysis) and only thin out the feasibles.
    target = enum_cfg.target_designs_per_topology
    feasibles = [c for c in candidates if c.fit_status == "fits"]
    if len(feasibles) > 3 * target:
        feasibles = _stratified_subsample(feasibles, target * 3)
        infeasibles = [c for c in candidates if c.fit_status != "fits"]
        candidates = feasibles + infeasibles

    logger.debug(
        "Topology %s: %d total candidates (%d feasible).",
        topology.topology_id(),
        len(candidates),
        sum(1 for c in candidates if c.fit_status == "fits"),
    )
    # Note: j_limit is logged only if we suspect mass infeasibility.
    if all(
        c.current_density_at_target_continuous_A_per_mm2 > j_limit
        for c in candidates if c.fit_status == "fits"
    ) and feasibles:
        logger.warning(
            "All %d feasible designs in %s exceed J limit %g A/mm^2 for %s cooling.",
            len(feasibles), topology.topology_id(), j_limit, cooling_mode,
        )

    return candidates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _phase_resistance_dc(
    turns_per_coil: int,
    coils_in_series_per_path: int,
    parallel_paths: int,
    stack_length_mm: float,
    end_turn_length_per_side_mm: float,
    wire_area_mm2: float,
    copper_resistivity_ohm_m: float,
) -> float:
    """DC phase resistance per spec section 5, Stage 3.

    R_phase = rho * (turns_per_coil * coils_in_series_per_path * 2 *
                     (stack_length + end_turn_per_side)) /
              (parallel_paths * area_mm^2 * 1000)

    The ``2 * (stack + end_turn)`` factor is one full conductor loop: into one
    end of the stack, across the back end-turn, back through the other side.
    """
    length_per_path_m = (
        turns_per_coil
        * coils_in_series_per_path
        * 2.0
        * (stack_length_mm + end_turn_length_per_side_mm)
        / 1000.0
    )
    area_m2 = wire_area_mm2 * 1e-6
    R_per_path = copper_resistivity_ohm_m * length_per_path_m / area_m2
    # Parallel paths -> resistance divides by (paths)^2 / paths = paths.
    return R_per_path / parallel_paths


def _estimate_continuous_current(
    turns_per_coil: int,
    coils_in_series_per_path: int,
    kw1: float,
    pole_pairs: int,
    outer_radius_mm: float,
    stack_length_mm: float,
    airgap_mm: float,
    target_torque_Nm: float,
    magnet_arc_fraction: float,
    magnet_thickness_mm: float,
) -> float:
    """Crude Kt estimate (back-of-envelope) for the Stage 3 J-density check.

    Stage 4 recomputes Kt properly; this is just for early flagging.
    """
    # Crude air-gap flux per pole using a representative B_g = 0.85 T.
    B_g_estimate = 0.85
    # Pole face area at the air-gap radius.
    A_pole_mm2 = (
        2.0 * math.pi * outer_radius_mm * stack_length_mm * magnet_arc_fraction
        / max(2 * pole_pairs, 1)
    )
    flux_per_pole_Wb = B_g_estimate * A_pole_mm2 * 1e-6

    N_series = turns_per_coil * coils_in_series_per_path
    Kt = 1.5 * pole_pairs * N_series * kw1 * flux_per_pole_Wb
    if Kt < 1e-6:
        return 1e6  # infeasible, will be flagged
    return target_torque_Nm / Kt


def _stratified_subsample(
    items: list[WindingCandidate], n_target: int
) -> list[WindingCandidate]:
    """Reduce ``items`` to ``n_target`` entries while preserving (turns, gauge) coverage.

    Strategy: sort by (turns, gauge) and take a uniform stride. This preserves
    corner cases (smallest turns, largest turns, finest wire, coarsest wire)
    and a uniform sampling of the interior.
    """
    if len(items) <= n_target:
        return items
    sorted_items = sorted(items, key=lambda c: (c.turns_per_coil, c.wire_gauge_AWG))
    stride = len(sorted_items) / n_target
    indices = sorted({int(i * stride) for i in range(n_target)})
    indices = [min(i, len(sorted_items) - 1) for i in indices]
    return [sorted_items[i] for i in indices]
