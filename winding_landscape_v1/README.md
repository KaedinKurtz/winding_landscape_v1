# winding_landscape_v1

A programmatic winding-landscape generator for radial-flux PMSM outrunner stators.

Given a stator BREP file and a set of operating constraints, this tool enumerates
and characterizes the feasible winding designs, producing a structured Parquet
dataset suitable for downstream Pareto filtering.

This is the **enumerate-and-characterize** half of a larger design tool. There is
no optimizer here -- the tradeoff exploration happens downstream on the output
dataset. The system is deliberately scoped to V1 (single architecture, single
manufacturing method, analytical models only); see `docs/architecture.md` and
the V1 spec for what's deferred.

---

## Installation

```bash
git clone <this-repo>
cd winding_landscape_v1
pip install -e ".[dev]"
```

Python 3.11+ is required. The dependency tree is bigger than you might expect
because `build123d` (BREP parsing) brings OpenCascade with it; expect a
substantial download on first install.

If `swat-em` is unavailable for your platform, the tool transparently falls
back to an in-house star-of-slots implementation that produces the same
results for V1.

---

## Quickstart

```bash
python -m winding_landscape \
    --stator-brep example_stator.step \
    --constraints example_constraints.yaml \
    --output-dir results/ \
    --csv
```

Outputs land in `results/`:

| File | Purpose |
|---|---|
| `landscape.parquet` | The main dataset, one row per winding design |
| `landscape_summary.json` | Counts, ranges, runtime, version stamps |
| `geometry_extraction_report.json` | Extracted slot dimensions + warnings |
| `landscape.csv` | Optional human-readable mirror (`--csv` flag) |
| `run_log.txt` | Full DEBUG-level run log |

Exit codes:
| Code | Meaning |
|---|---|
| 0 | Success, at least one feasible design |
| 1 | Invalid input (BREP unparseable, constraints invalid) |
| 2 | Geometry extraction failed |
| 3 | No feasible designs found (output still written) |
| 4 | Internal error |

Use `--dry-run` to test geometry extraction without running the simulation
pipeline -- handy for validating that a new BREP parses correctly.

---

## Constraints file

YAML or JSON, with sensible defaults for every field. The minimal valid
constraints file is `{}`. See `example_constraints.yaml` for the full
schema with documentation. Defaults that get applied are recorded in
`landscape_summary.json` under `defaults_applied` so you know what
the run is actually using.

---

## Output schema

41 columns per design. The most-useful ones:

| Column | Description |
|---|---|
| `design_hash` | Deterministic 16-hex-char ID; same params -> same hash |
| `topology_id` | e.g. "Q12_p10_L2_y1" |
| `slot_count`, `pole_count`, `coil_pitch_slots`, `turns_per_coil`, `wire_gauge_AWG` | Winding parameters |
| `winding_factor_fundamental`, `winding_factor_5th`, ... `_13th` | Harmonic content from star-of-slots |
| `Kt_analytical_Nm_per_A`, `Ke_analytical_V_per_rad_per_s` | EM constants from MEC |
| `peak_torque_at_max_current_Nm` | Saturation-derated peak torque |
| `continuous_torque_thermal_limit_Nm` | Thermal-limited continuous torque |
| `max_speed_at_supply_voltage_rpm` | Voltage-limited top speed |
| `predicted_winding_temp_at_continuous_C` | Steady-state winding temp at the target continuous torque |
| `feasibility_status` | "feasible" or "infeasible_{fit,current,voltage,thermal}" |
| `slot_fill_factor_actual` | Achieved fill (>1 for designs that don't fit -- intentional, useful for downstream what-if analysis) |

Infeasible designs are **included** in the output, marked with their failure
mode. This is intentional: it lets downstream analysis answer "what if we
relaxed the voltage to 60V -- how many more designs become feasible?" without
having to re-run the pipeline.

---

## Architecture

A 7-stage pipeline; each stage is a separate module with a clean dataclass
contract to the next. See `docs/architecture.md` for the long version.

```
BREP file + Constraints YAML + Materials DB
                  |
                  v
   Stage 1: Geometry extraction (build123d)
                  |
                  v
   Stage 2: Topology enumeration (star-of-slots)
                  |
                  v
   Stage 3: Winding enumeration (turns x gauge sweep)
                  |
                  v
   Stage 4: Analytical EM (MEC + winding factors)
                  |
                  v
   Stage 5: Steady-state thermal (3-node lumped)
                  |
                  v
   Stage 6: Feasibility classification
                  |
                  v
   Stage 7: Output serialization (Parquet + JSON + CSV)
```

V1 explicitly does **not** include FEM simulation. All EM characterization is
analytical. The user should expect ~5-15% accuracy vs FEM and use V1 outputs for
landscape exploration, not for final design commitment.

---

## What's deliberately out of scope (V2+)

- FEM simulation (FEMM 2D, Elmer 3D)
- Cogging-torque waveform (only period + heuristic factor in V1)
- Litz wire and multi-strand conductors
- Hairpin / formed-coil manufacturing
- Single-layer windings (the code path exists in `star_of_slots` but Stage 2
  is restricted to double-layer in V1)
- Skewed stators, axial taper
- Axial-flux or transverse-flux topologies
- AC losses (skin and proximity effects)
- 3D end-turn modeling (only end-turn length is heuristically estimated)

Each is documented at the relevant code site. When V1 hits a design choice
that would require any of the above, it makes the simplest reasonable
assumption and documents it.

---

## Validation

The pipeline ships with 70+ unit tests covering:

- Star-of-slots winding factors against canonical textbook values
  (12s/10p, 24s/22p, 36s/8p, 36s/4p, 12s/8p, 9s/8p) -- all match within 0.001
- AWG table self-consistency against the standard formula
- Steinmetz iron-loss coefficients calibrated to give realistic W/kg at
  60 Hz / 1.5 T (M19 ~4 W/kg, M15 ~3 W/kg, Hiperco50 ~9 W/kg)
- Carter's coefficient sanity (returns 1.0 with no slots, monotonic in opening width)
- Thermal model convergence (under-relaxed fixed-point iteration with hard 500 C cap)
- Feasibility-priority ordering (fit > current > voltage > thermal)
- Parameter-scaling property tests (Ke linear in turns, R monotonic in turns, etc.)

Run with:
```bash
pytest tests/
```

---

## Known limitations and where the model is least accurate

| Quantity | Expected accuracy vs FEM | Why |
|---|---|---|
| Kt, Ke | ±5% | Solid 1D MEC; main error from magnet-arc-fraction heuristic |
| Ls (synchronous inductance) | ±30% | Crude formula; ignores leakage and saturation effects |
| B_tooth | ±15% | Flux-area ratio is a 1D approximation |
| Iron loss | ±20% | Steinmetz coefficients are typical-of-grade, not vendor-calibrated |
| Thermal | ±25% | Lumped 3-node ignores axial heat flow and end-turn effects |
| Cogging | n/a | Only period reported; amplitude requires FEM |

---

## Citation / references

Formulas in V1 follow:
- Hanselman, *Brushless Permanent Magnet Motor Design*, 2nd ed. (Chapters 4-6)
- Hendershot & Miller, *Design of Brushless Permanent-Magnet Machines*
- Bianchi & Dai Pre, "Use of the star of slots in designing fractional-slot
  single-layer synchronous motors," IEE Proc. EPA, 2006
- SWAT-EM documentation (https://swat-em.readthedocs.io/) for the topology
  analysis cross-check
