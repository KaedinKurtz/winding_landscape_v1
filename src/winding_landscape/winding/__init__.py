"""Stage 3: Winding enumeration.

For each topology candidate, sweep (turns_per_coil, wire_gauge_AWG) combinations
and produce :class:`WindingCandidate` objects. Constraint-aware: rejects designs
that don't fit in the slot or pass through the slot opening.
"""
