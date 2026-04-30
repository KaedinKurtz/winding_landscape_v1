# Architecture

This document describes how the V1 pipeline is structured. The V1 spec
(`winding_optimizer_v1_system_design.md`) is the authoritative source for
*what* gets computed; this document is about *how* the code is organized.

## The 7-stage pipeline

Each stage is a separate package. Inter-stage communication uses frozen
dataclasses (immutable record types) so that downstream stages can't
accidentally mutate upstream outputs.

```
                      Constraints YAML        Materials JSON DB
                            |                       |
                            v                       v
   BREP file -> [Stage 1: Geometry] -> StatorGeometry
                            |
                            v
                  [Stage 2: Topology]  -> List[TopologyCandidate]
                            |
                            v
                  [Stage 3: Winding]   -> List[WindingCandidate]
                            |
                            v
                  [Stage 4: EM (MEC)]  -> CharacterizedDesign (EM fields)
                            |
                            v
                  [Stage 5: Thermal]   -> CharacterizedDesign (+ thermal)
                            |
                            v
                  [Stage 6: Feasibility] -> CharacterizedDesign (+ status)
                            |
                            v
                  [Stage 7: Output]    -> Parquet + JSON files
```

### Why composition, not inheritance, for the dataclass progression

If you're coming from C++, the natural pattern is

```cpp
class WindingCandidate : public TopologyCandidate { ... };
class CharacterizedDesign : public WindingCandidate { ... };
```

Python supports inheritance just as well, but the *dataclass progression*
becomes hard to reason about when fields can come from anywhere up the chain.
Instead, each stage's dataclass holds the previous stage's dataclass as a
**field** (composition):

```python
@dataclass
class WindingCandidate:
    topology: TopologyCandidate     # composition, not inheritance
    turns_per_coil: int
    ...

@dataclass
class CharacterizedDesign:
    winding: WindingCandidate       # composition
    Ke_V_per_rad_per_s: float = 0.0
    ...
```

This keeps each stage's contribution visually scoped to its own dataclass and
lets you grep for "where does Kt come from?" by searching for `Kt_Nm_per_A` in
exactly one place.

## Stage 1: Geometry extraction (`geometry/extraction.py`)

Loads the BREP, identifies the rotation axis, slices at mid-stack, and walks
the 2D cross-section to extract every dimension downstream stages need.

The mechanical-engineering analogy: this is the inspection step. Treat the
BREP as the part-as-built; this stage's job is to put it on a virtual CMM and
measure all the features the design spec called out.

Key tolerances:
- **Periodicity**: rotate the OD profile by 360°/Q and check it overlaps
  itself. Warn at 0.05 mm deviation, fatal at 1.0 mm.
- **Slot detection**: a "slot" is a contiguous angular segment where the OD
  radius drops below `outer_radius - 0.1 mm`. Run-length-encode to count.
- **Bore identification**: the smallest cylindrical face concentric with the
  bounding-box centroid.

Returns a `StatorGeometry` frozen dataclass with all dimensions in mm and
mm² (areas).

## Stage 2: Topology enumeration (`topology/enumeration.py`)

Sweeps all `(Q, 2p, layers, coil_pitch)` combinations subject to the
constraints, runs star-of-slots on each, and returns the survivors ranked
by a composite score.

Two implementations of the star-of-slots are provided:
- `star_of_slots.py`: in-house pure-Python; the canonical V1 implementation.
- `swat_em_wrapper.py`: thin wrapper around the SWAT-EM library.

If both are available, the wrapper calls SWAT-EM and cross-checks the kw1
result against the in-house implementation; >2% disagreement is logged.

The in-house implementation is validated against canonical textbook values
(Hanselman, Bianchi) for 12s/10p, 24s/22p, 36s/8p, 36s/4p, 12s/8p, 9s/8p.
Tests pass within ±0.001.

The composite topology score:

    score = kw1
          - 0.3 * (|kw5| + |kw7|)
          - 0.1 * (|kw11| + |kw13|)
          + 0.5 * (1 - 1 / LCM(Q, 2p))

The first three terms penalize the lowest-order torque-ripple harmonics; the
last term rewards low cogging (high LCM = low cogging period frequency).

Returns ranked `List[TopologyCandidate]`.

## Stage 3: Winding enumeration (`winding/enumeration.py`)

For each topology, sweeps `(turns_per_coil, wire_gauge_AWG)` and emits a
`WindingCandidate` for every combination -- including infeasible ones, with
their `fit_status` flagged. Keeping infeasibles is intentional: the downstream
landscape is meant to support "what if we relaxed X by Y%?" questions without
re-running the pipeline.

Fit checks:
1. **Area**: required insulated copper area <= achievable fill x slot useful area.
2. **Opening**: insulated wire OD < 0.95 x slot opening width (so it can be
   wound through the opening for random round wire).

Density control: if a topology produces >3x the target candidate count, the
list is stratified-subsampled to preserve corner cases (extremes of turns,
extremes of gauges) and uniform interior coverage.

## Stage 4: Electromagnetic (`performance/electromagnetic.py`)

The Magnetic Equivalent Circuit (MEC) model. Mechanical analogy: hydraulic
network. Magnet is a pressure source; airgap and recoil permeability are
flow resistances; Carter's coefficient corrects for slot openings.

Key formulas (all from Hanselman):

- **Carter's coefficient** (eqn 4.31): corrects effective airgap for slotting.
- **Air-gap flux density**: `B_g = B_r / (1 + mu_rec * g_eff / l_m)`
- **Pole flux**: `Phi = B_g * A_pole_face`
- **Back-EMF constant**: `Ke = N_series * kw1 * Phi * pole_pairs`
- **Torque constant**: `Kt = 1.5 * Ke` (sinusoidal back-EMF, vector control)
- **Synchronous inductance**: `Ls ~ mu0 * L * N^2 * kw1^2 / (p^2 * g_eff * 2pi)`

Saturation derating: `B_tooth = B_g * (slot_pitch / tooth_width) * (1 + 0.2)`
where the 1.2 factor accounts for armature reaction at peak load. If
`B_tooth > 1.8 T`, peak torque is linearly derated to 0.7 at `B_tooth = 2.2 T`.
For high-saturation steels (Hiperco50, B_sat = 2.35 T) the knee shifts up.

Iron loss uses the Steinmetz formula, with the average B clamped to the
steel's saturation B_T to prevent unbounded loss in deeply saturated regions.

## Stage 5: Thermal (`performance/thermal.py`)

A 3-node lumped-parameter network:

```
   Winding -- R_w_to_iron -- Iron -- R_iron_to_housing -- Housing -- R_housing_to_amb -- Ambient
       |
   P_copper                  P_iron flows in here
```

The thermal-resistance estimates:
- `R_w_to_iron`: slot-liner conduction + half-winding bulk.
  Heat-transfer area = (slot perimeter * stack length) summed over Q slots.
- `R_iron_to_housing`: cylindrical-wall conduction `ln(r_out/r_in) / (4*pi*k*L)`
  plus a press-fit contact resistance of 0.005 K-m^2/W.
- `R_housing_to_amb`: from the constraints YAML; user knows their housing best.

The iteration is non-trivial because copper resistivity rises with temperature,
which raises copper loss, which raises temperature. We use under-relaxation
(damping factor 0.5) and a hard 500 C ceiling to prevent runaway. If the
solver hits the ceiling, the design will fail thermal feasibility -- which
is the right answer.

For the continuous-torque thermal limit, we run a Brent-method root-find on
the residual `T_w(I_phase) - T_w_max` -- find the current that puts winding
temp exactly at the insulation ceiling.

## Stage 6: Feasibility (`feasibility/checker.py`)

Per V1 spec: priority ordering is **fit > current > voltage > thermal**. A
design with multiple violations is reported with the highest-priority status,
and all violations are listed in the notes field.

## Stage 7: Output (`output/serialization.py`)

Writes:
- `landscape.parquet` (or `landscape.jsonl` if pyarrow is unavailable)
- `landscape_summary.json` (counts, ranges, runtime, version stamps)
- `geometry_extraction_report.json` (extracted dimensions + warnings)
- `landscape.csv` (optional, `--csv` flag)

The Parquet schema is locked at 41 columns in a fixed order
(`_PARQUET_COLUMNS` in `serialization.py`). Adding a column is a
schema bump and should be reflected in `code_version`.

## Determinism and reproducibility

Every design gets a 16-hex-char `design_hash` computed deterministically from
its parameter dict (sorted-key canonical JSON, SHA-256, truncated). Same
parameters -> same hash, always.

The full code-version stamp embedded in every row is `{__version__}+g{git_short}`.
This lets you tell which commit produced which row of which dataset --
critical when you start running multiple sweeps and combining them downstream.

## Where to put V2 extension points

The V1 architecture deliberately leaves room for V2 to slot in without
re-architecting. The expected extensions:

- **FEM as a strategic verification layer**: add a new column family
  `fem_validated_*` alongside the analytical results. Don't replace; both
  should coexist so we can train a correction surrogate on the difference.
- **Architecture discriminator**: V1 always produces "PMSM_radial_outrunner";
  add an `architecture` column and dispatch geometry extraction + topology
  enumeration on its value.
- **Manufacturing method**: V1 hardcodes "random_round". Hairpin and
  formed-coil are V2 work; the `manufacturing_method` column already exists
  as a discriminator.

The V1 architecture *is* extensible for these but is **not** preemptively
generalized -- which is why the codebase is small and grokkable for V1.
Build for V1; document extension points; do not engineer for hypothetical
needs that may never arrive.
