"""Integration test: full pipeline from synthetic geometry to written outputs.

Stage 1 (BREP extraction) is bypassed -- we feed in a synthetic StatorGeometry
directly. Everything else (Stages 2-7) runs end to end.
"""

from __future__ import annotations

import json
from pathlib import Path

from winding_landscape.feasibility.checker import classify_feasibility
from winding_landscape.output.serialization import write_landscape
from winding_landscape.performance.characterized_design import CharacterizedDesign
from winding_landscape.performance.electromagnetic import characterize_electromagnetic
from winding_landscape.performance.thermal import characterize_thermal
from winding_landscape.topology.enumeration import enumerate_topologies
from winding_landscape.winding.enumeration import enumerate_windings


def test_full_pipeline_writes_valid_outputs(
    synthetic_geometry, reasonable_constraints, materials_db, tmp_path: Path
):
    designs: list[CharacterizedDesign] = []
    for top in enumerate_topologies(synthetic_geometry, reasonable_constraints):
        for w in enumerate_windings(
            top, synthetic_geometry, reasonable_constraints, materials_db
        ):
            d = characterize_electromagnetic(
                w, synthetic_geometry, reasonable_constraints, materials_db
            )
            if w.is_geometrically_feasible():
                d = characterize_thermal(
                    d, synthetic_geometry, reasonable_constraints, materials_db
                )
            d = classify_feasibility(d, reasonable_constraints)
            designs.append(d)

    assert len(designs) > 0

    output_dir = tmp_path / "pipeline_out"
    paths = write_landscape(
        designs=designs,
        output_dir=output_dir,
        geometry=synthetic_geometry,
        constraints=reasonable_constraints,
        runtime_seconds=1.0,
        defaults_used=[],
        write_csv=True,
    )

    # Summary file is well-formed and contains expected keys.
    summary = json.loads(paths["summary"].read_text())
    assert summary["total_designs"] == len(designs)
    assert "feasibility_counts" in summary
    assert sum(summary["feasibility_counts"].values()) == len(designs)
    assert "constraints_used" in summary

    # Geometry report exists and contains the slot count.
    geo_report = json.loads(paths["geometry_report"].read_text())
    assert geo_report["geometry"]["slot_count"] == synthetic_geometry.slot_count

    # CSV exists, has a header row plus N data rows.
    csv_lines = paths["csv"].read_text().strip().split("\n")
    assert len(csv_lines) == len(designs) + 1


def test_at_least_one_feasible_design(
    synthetic_geometry, reasonable_constraints, materials_db
):
    """For the chosen synthetic geometry + reasonable constraints, the pipeline
    must produce at least one fully feasible design."""
    feasible_count = 0
    for top in enumerate_topologies(synthetic_geometry, reasonable_constraints):
        for w in enumerate_windings(
            top, synthetic_geometry, reasonable_constraints, materials_db
        ):
            d = characterize_electromagnetic(
                w, synthetic_geometry, reasonable_constraints, materials_db
            )
            if w.is_geometrically_feasible():
                d = characterize_thermal(
                    d, synthetic_geometry, reasonable_constraints, materials_db
                )
            d = classify_feasibility(d, reasonable_constraints)
            if d.feasibility_status == "feasible":
                feasible_count += 1
    assert feasible_count > 0


def test_design_hashes_unique(
    synthetic_geometry, reasonable_constraints, materials_db, tmp_path: Path
):
    """Each (topology, turns, gauge) tuple should yield a unique design_hash."""
    import csv

    designs: list[CharacterizedDesign] = []
    for top in enumerate_topologies(synthetic_geometry, reasonable_constraints):
        for w in enumerate_windings(
            top, synthetic_geometry, reasonable_constraints, materials_db
        ):
            d = characterize_electromagnetic(
                w, synthetic_geometry, reasonable_constraints, materials_db
            )
            if w.is_geometrically_feasible():
                d = characterize_thermal(
                    d, synthetic_geometry, reasonable_constraints, materials_db
                )
            d = classify_feasibility(d, reasonable_constraints)
            designs.append(d)

    paths = write_landscape(
        designs=designs,
        output_dir=tmp_path / "hash_check",
        geometry=synthetic_geometry,
        constraints=reasonable_constraints,
        runtime_seconds=1.0,
        defaults_used=[],
        write_csv=True,
    )
    rows = list(csv.DictReader(paths["csv"].open()))
    hashes = [r["design_hash"] for r in rows]
    assert len(hashes) == len(set(hashes)), "design_hash collisions detected"
