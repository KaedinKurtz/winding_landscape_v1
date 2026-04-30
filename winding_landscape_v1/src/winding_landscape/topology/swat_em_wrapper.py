"""Thin wrapper around SWAT-EM (https://swat-em.readthedocs.io/).

In V1 we use SWAT-EM if it imports cleanly; otherwise we fall back to the
in-house :mod:`star_of_slots` implementation. Both produce ``StarOfSlotsResult``
objects, so the rest of the pipeline doesn't care which engine was used.

If SWAT-EM is available we still cross-check kw1 against the in-house
implementation as a smoke-test: a >2% disagreement is logged at WARNING level.
"""

from __future__ import annotations

from winding_landscape.topology.star_of_slots import (
    StarOfSlotsResult,
    star_of_slots,
)
from winding_landscape.utils.logging_config import get_logger

logger = get_logger(__name__)


def swat_em_available() -> bool:
    """Return True if SWAT-EM is importable in this environment."""
    try:
        import swat_em  # noqa: F401
    except ImportError:
        return False
    return True


def analyze_winding(
    Q: int,
    pole_count: int,
    coil_pitch_slots: int,
    layers: int = 2,
    harmonics: tuple[int, ...] = (1, 5, 7, 11, 13),
    prefer_swat_em: bool = True,
) -> StarOfSlotsResult:
    """Analyze a (Q, 2p, layers, pitch) topology.

    Tries SWAT-EM first if available and ``prefer_swat_em`` is True; falls back
    to the in-house star-of-slots implementation. The two are cross-checked
    when both are available.
    """
    inhouse = star_of_slots(Q, pole_count, coil_pitch_slots, layers, harmonics)

    if not (prefer_swat_em and swat_em_available()):
        return inhouse

    try:
        swat_result = _swat_em_compute(Q, pole_count, coil_pitch_slots, layers, harmonics)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "SWAT-EM call failed for Q=%d, p=%d, y=%d (%s); using in-house fallback.",
            Q, pole_count, coil_pitch_slots, exc,
        )
        return inhouse

    # Cross-check fundamental winding factor.
    kw1_diff = abs(swat_result.winding_factors[1] - inhouse.winding_factors[1])
    if kw1_diff > 0.02:
        logger.warning(
            "SWAT-EM and in-house kw1 disagree by %.3f for Q=%d, p=%d, y=%d. "
            "Using SWAT-EM result.", kw1_diff, Q, pole_count, coil_pitch_slots,
        )
    return swat_result


def _swat_em_compute(
    Q: int,
    pole_count: int,
    coil_pitch_slots: int,
    layers: int,
    harmonics: tuple[int, ...],
) -> StarOfSlotsResult:
    """Compute via SWAT-EM. Format-converts to our StarOfSlotsResult."""
    import numpy as np
    from swat_em import datamodel as swat_dm

    wdg = swat_dm()
    wdg.genwdg(Q=Q, P=pole_count, m=3, w=coil_pitch_slots, layers=layers)

    # SWAT-EM stores the connection matrix in wdg.machinedata['phases'] as
    # a list of (slot_idx, sign, layer_idx) tuples per phase. We translate it
    # into our Q x 3 signed integer matrix.
    conn = np.zeros((Q, 3), dtype=np.int_)
    for phase_idx, phase_data in enumerate(wdg.get_phases()):
        for slot_entry in phase_data:
            # Format may be (slot, sign) or (slot, sign, layer) depending on version.
            slot = int(slot_entry[0])
            sign = int(slot_entry[1])
            conn[slot - 1, phase_idx] += sign

    # Winding factors per harmonic.
    wf: dict[int, float] = {}
    nu_array, kw_array = wdg.get_windingfactor()
    nu_to_kw = dict(zip(nu_array.tolist(), kw_array.tolist(), strict=False))
    for h in harmonics:
        wf[h] = float(abs(nu_to_kw.get(h, 0.0)))

    is_bal = bool(wdg.is_symmetric())
    coils_per_phase = int(np.sum(np.abs(conn[:, 0])))

    return StarOfSlotsResult(
        connection_matrix=conn,
        winding_factors=wf,
        is_balanced=is_bal,
        coils_per_phase=coils_per_phase,
    )
