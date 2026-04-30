"""Stage 1: Stator geometry extraction from BREP.

Reads a `.step`/`.stp`/`.brep` file and produces a :class:`StatorGeometry`
dataclass with all the slot/tooth/yoke dimensions the downstream stages need.
"""
