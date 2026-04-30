"""Smoke test: run the pipeline (Stages 2-7) end-to-end on a synthetic geometry.

This bypasses Stage 1 (which requires build123d and a real BREP file). Useful
for confirming the math pipeline works without setting up a CAD environment.

Run from project root:
    PYTHONPATH=src python tests/smoke_test_no_brep.py
"""

from __future__ import annotations

import time
from pathlib import Path

from winding_landscape.config import Constraints
from winding_landscape.feasibility.checker import classify_feasibility
from winding_landscape.geometry.stator_geometry import StatorGeometry
from winding_landscape.materials.database import load_materials
from winding_landscape.output.serialization import write_landscape
from winding_landscape.performance.electromagnetic import characterize_electromagnetic
from winding_landscape.performance.thermal import characterize_thermal
from winding_landscape.topology.enumeration import enumerate_topologies
from winding_landscape.winding.enumeration import enumerate_windings


def main() -> None:
    start = time.perf_counter()

    # A 60 mm OD outrunner stator -- representative of a hub motor for a small
    # actuator or scooter.
    geometry = StatorGeometry(
        bore_radius_mm=25.0,
        outer_radius_mm=60.0,
        stack_length_mm=40.0,
        slot_count=12,
        slot_opening_width_mm=2.5,
        slot_opening_depth_mm=1.5,
        slot_total_depth_mm=18.0,
        slot_bottom_width_mm=14.0,
        slot_area_mm2=180.0,
        slot_useful_area_mm2=170.0,
        tooth_width_min_mm=14.0,
        yoke_thickness_mm=17.0,
        airgap_mm=0.7,
        periodicity_tolerance_mm=0.001,
    )

    constraints = Constraints(
        operating_targets={
            "target_continuous_torque_Nm": 1.0,
            "target_peak_torque_Nm": 3.0,
            "target_max_speed_rpm": 2000.0,
        },
        electrical_envelope={
            "supply_voltage_V": 96.0,
            "max_phase_current_A_rms": 30.0,
            "modulation_margin": 0.9,
        },
        thermal_envelope={
            "cooling_mode": "forced_air",
            "housing_thermal_resistance_K_per_W": 0.3,
        },
    )

    materials = load_materials()

    print("Stage 2: enumerating topologies...")
    topologies = enumerate_topologies(geometry, constraints)
    print(f"  -> {len(topologies)} topology candidates")
    for t in topologies[:5]:
        print(
            f"     {t.topology_id():>20s} kw1={t.winding_factor_fundamental:.3f} "
            f"score={t.topology_score:.3f}"
        )

    print("Stages 3-6: enumerating + characterizing windings...")
    all_designs = []
    for topology in topologies:
        windings = enumerate_windings(topology, geometry, constraints, materials)
        for w in windings:
            d = characterize_electromagnetic(w, geometry, constraints, materials)
            if w.is_geometrically_feasible():
                d = characterize_thermal(d, geometry, constraints, materials)
            d = classify_feasibility(d, constraints)
            all_designs.append(d)

    runtime = time.perf_counter() - start
    print(f"  -> {len(all_designs)} designs total in {runtime:.2f} s")
    from collections import Counter
    print(f"     status counts: {dict(Counter(d.feasibility_status for d in all_designs))}")

    print("Stage 7: writing outputs...")
    out_dir = Path("smoke_test_output")
    written = write_landscape(
        designs=all_designs,
        output_dir=out_dir,
        geometry=geometry,
        constraints=constraints,
        runtime_seconds=runtime,
        defaults_used=[],
        write_csv=True,
    )
    for label, path in written.items():
        print(f"     {label}: {path}")

    print(f"\nDone. Total runtime: {time.perf_counter() - start:.2f} s")


if __name__ == "__main__":
    main()
