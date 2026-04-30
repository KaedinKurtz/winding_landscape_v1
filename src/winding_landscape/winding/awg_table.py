"""Convenience accessors for AWG wire dimensions.

The actual table lives in :mod:`winding_landscape.materials.database` (loaded
from ``materials/data/awg_table.json``). This module just exposes a slim
typed interface for Stage 3.
"""

from __future__ import annotations

from winding_landscape.materials.database import AwgEntry, MaterialsDatabase


def lookup_awg(materials: MaterialsDatabase, gauge: int) -> AwgEntry:
    """Look up an AWG entry, raising KeyError with a helpful message if missing."""
    return materials.get_awg(gauge)


def filter_available_gauges(materials: MaterialsDatabase, gauges: list[int]) -> list[int]:
    """Return the subset of ``gauges`` that have entries in the AWG table."""
    return sorted(g for g in gauges if g in materials.awg)
