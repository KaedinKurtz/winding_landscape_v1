"""PMSM outrunner stator winding landscape generator (V1).

This package enumerates and characterizes feasible winding designs for a given
stator BREP and set of operating constraints. The output is a structured dataset
(Parquet) suitable for downstream Pareto filtering and atlas integration.

The pipeline is a 7-stage process (see docs/architecture.md):
    1. Geometry extraction from BREP
    2. Topology enumeration (Q, 2p, layers, pitch)
    3. Winding enumeration (turns, gauge) -- constraint-aware
    4. Analytical performance (MEC-based EM)
    5. Steady-state thermal (lumped network)
    6. Feasibility classification
    7. Output serialization (Parquet + JSON summaries)
"""

__version__ = "1.0.0"
