"""Command-line entry point.

Usage:
    python -m winding_landscape \\
        --stator-brep path/to/stator.step \\
        --constraints path/to/constraints.yaml \\
        --output-dir path/to/output/ \\
        [--csv] [--verbose] [--max-topologies 10] [--dry-run]

Exit codes (per spec section 8):
    0 : success
    1 : invalid input (BREP unparseable, constraints invalid)
    2 : extraction failure
    3 : no feasible designs found (warning, output still written)
    4 : internal error
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from winding_landscape.config import Constraints, load_constraints
from winding_landscape.feasibility.checker import classify_feasibility
from winding_landscape.geometry.extraction import (
    GeometryExtractionError,
    extract_geometry,
)
from winding_landscape.materials.database import load_materials
from winding_landscape.output.serialization import write_landscape
from winding_landscape.performance.characterized_design import CharacterizedDesign
from winding_landscape.performance.electromagnetic import characterize_electromagnetic
from winding_landscape.performance.thermal import characterize_thermal
from winding_landscape.topology.enumeration import enumerate_topologies
from winding_landscape.utils.logging_config import configure_logging, get_logger
from winding_landscape.winding.enumeration import enumerate_windings


def main(argv: list[str] | None = None) -> int:
    """Run the full pipeline. Returns the appropriate exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir / "run_log.txt", verbose=args.verbose)
    logger = get_logger(__name__)

    start = time.perf_counter()

    # ---- Load inputs ----
    try:
        constraints, defaults_used = load_constraints(args.constraints)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load constraints: %s", exc)
        return 1

    if defaults_used:
        logger.info("Defaults applied for: %s", defaults_used)

    if args.max_topologies is not None:
        # CLI override takes effect by mutating the loaded constraints.
        constraints = constraints.model_copy(
            update={
                "enumeration_density": constraints.enumeration_density.model_copy(
                    update={"max_topologies": args.max_topologies}
                )
            }
        )

    try:
        materials = load_materials()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load materials database: %s", exc)
        return 4

    # Validate that the chosen grades exist.
    if constraints.materials.steel_grade not in materials.steels:
        logger.error(
            "Steel grade '%s' not in materials DB. Available: %s",
            constraints.materials.steel_grade, sorted(materials.steels.keys()),
        )
        return 1
    if constraints.materials.magnet_grade not in materials.magnets:
        logger.error(
            "Magnet grade '%s' not in materials DB. Available: %s",
            constraints.materials.magnet_grade, sorted(materials.magnets.keys()),
        )
        return 1

    # ---- Stage 1: Geometry extraction ----
    logger.info("Stage 1: Geometry extraction")
    try:
        geometry = extract_geometry(args.stator_brep, constraints)
    except GeometryExtractionError as exc:
        logger.error("Geometry extraction failed: %s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during geometry extraction: %s", exc)
        return 4

    if args.dry_run:
        logger.info("Dry run: stopping after geometry extraction.")
        runtime = time.perf_counter() - start
        write_landscape(
            designs=[], output_dir=output_dir,
            geometry=geometry, constraints=constraints,
            runtime_seconds=runtime, defaults_used=defaults_used,
            write_csv=args.csv,
        )
        return 0

    # ---- Stage 2: Topology enumeration ----
    logger.info("Stage 2: Topology enumeration")
    topologies = enumerate_topologies(geometry, constraints)
    if not topologies:
        logger.warning("No viable topologies found.")
        runtime = time.perf_counter() - start
        write_landscape(
            designs=[], output_dir=output_dir,
            geometry=geometry, constraints=constraints,
            runtime_seconds=runtime, defaults_used=defaults_used,
            write_csv=args.csv,
        )
        return 3

    # ---- Stages 3-6: per-topology winding enumeration + characterization ----
    all_designs: list[CharacterizedDesign] = []
    for topology in topologies:
        logger.info("Processing topology %s", topology.topology_id())
        windings = enumerate_windings(topology, geometry, constraints, materials)

        for w in windings:
            # Stage 4
            try:
                design = characterize_electromagnetic(w, geometry, constraints, materials)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "EM characterization failed for %s, N=%d, AWG=%d: %s",
                    topology.topology_id(), w.turns_per_coil, w.wire_gauge_AWG, exc,
                )
                continue
            # Stage 5 (only if the design fit -- thermal is meaningless if it doesn't fit)
            if w.is_geometrically_feasible():
                try:
                    design = characterize_thermal(design, geometry, constraints, materials)
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "Thermal characterization failed for %s: %s",
                        topology.topology_id(), exc,
                    )
            # Stage 6
            design = classify_feasibility(design, constraints)
            all_designs.append(design)

    runtime = time.perf_counter() - start
    logger.info(
        "Processed %d designs in %.2f seconds.", len(all_designs), runtime,
    )

    # ---- Stage 7: serialize ----
    feasible_count = sum(1 for d in all_designs if d.feasibility_status == "feasible")
    logger.info("%d / %d designs are feasible.", feasible_count, len(all_designs))

    write_landscape(
        designs=all_designs,
        output_dir=output_dir,
        geometry=geometry,
        constraints=constraints,
        runtime_seconds=runtime,
        defaults_used=defaults_used,
        write_csv=args.csv,
    )

    return 0 if feasible_count > 0 else 3


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="winding-landscape",
        description="Generate a winding-landscape dataset for a stator BREP.",
    )
    p.add_argument(
        "--stator-brep", required=True,
        help="Path to the stator BREP file (.step, .stp, or .brep).",
    )
    p.add_argument(
        "--constraints", required=True,
        help="Path to the constraints YAML or JSON file.",
    )
    p.add_argument(
        "--output-dir", required=True,
        help="Directory to write outputs into. Created if it doesn't exist.",
    )
    p.add_argument(
        "--csv", action="store_true",
        help="Also emit a CSV mirror of the Parquet for human inspection.",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging on the console.",
    )
    p.add_argument(
        "--max-topologies", type=int, default=None,
        help="Override the topology candidate cap.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Run extraction only and write a summary; skip simulation.",
    )
    return p


if __name__ == "__main__":
    sys.exit(main())
