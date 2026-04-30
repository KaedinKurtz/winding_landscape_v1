"""Stage 7: Output serialization.

Produces:
    landscape.parquet                -- the main dataset
    landscape_summary.json           -- counts, ranges, runtime, version info
    geometry_extraction_report.json  -- extracted dimensions + warnings
    landscape.csv                    -- optional human-readable mirror

The Parquet schema follows spec section 3.1 exactly.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from winding_landscape.config import Constraints
from winding_landscape.geometry.stator_geometry import StatorGeometry
from winding_landscape.materials.database import MATERIALS_DB_VERSION
from winding_landscape.performance.characterized_design import CharacterizedDesign
from winding_landscape.utils.hashing import (
    GEOMETRY_EXTRACTION_VERSION,
    design_hash,
    get_code_version,
)
from winding_landscape.utils.logging_config import get_logger

logger = get_logger(__name__)


# Spec section 3.1 column order. Used to enforce a stable schema.
_PARQUET_COLUMNS: list[str] = [
    "design_hash",
    "topology_id",
    "slot_count",
    "pole_count",
    "layers",
    "coil_pitch_slots",
    "phase_count",
    "parallel_paths",
    "turns_per_coil",
    "wire_gauge_AWG",
    "wire_diameter_bare_mm",
    "wire_diameter_insulated_mm",
    "strands_per_conductor",
    "manufacturing_method",
    "connection_matrix_json",
    "winding_factor_fundamental",
    "winding_factor_5th",
    "winding_factor_7th",
    "winding_factor_11th",
    "winding_factor_13th",
    "cogging_period_mech_deg",
    "is_balanced",
    "slot_fill_factor_actual",
    "slot_fill_factor_limit",
    "current_density_A_per_mm2",
    "phase_resistance_ohm_at_20C",
    "phase_inductance_synchronous_mH",
    "Kt_analytical_Nm_per_A",
    "Ke_analytical_V_per_rad_per_s",
    "peak_torque_at_max_current_Nm",
    "continuous_torque_thermal_limit_Nm",
    "max_speed_at_supply_voltage_rpm",
    "predicted_winding_temp_at_continuous_C",
    "predicted_iron_loss_at_max_speed_W",
    "predicted_copper_loss_at_continuous_W",
    "feasibility_status",
    "feasibility_notes",
    "end_turn_length_estimate_mm",
    "total_copper_mass_g",
    "geometry_extraction_version",
    "materials_db_version",
    "code_version",
]


def write_landscape(
    designs: list[CharacterizedDesign],
    output_dir: Path,
    geometry: StatorGeometry,
    constraints: Constraints,
    runtime_seconds: float,
    defaults_used: list[str],
    write_csv: bool,
) -> dict[str, Path]:
    """Write all output artifacts. Returns a dict of written file paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    code_ver = get_code_version()

    # -------- Build the dataframe row-by-row --------
    rows = [_design_to_row(d, code_version=code_ver) for d in designs]
    df = pd.DataFrame(rows, columns=_PARQUET_COLUMNS)

    # Enforce dtypes for stable Parquet schema.
    df = _enforce_dtypes(df)

    parquet_path = output_dir / "landscape.parquet"
    try:
        df.to_parquet(parquet_path, index=False, engine="pyarrow")
    except ImportError:
        # Fall back to JSON-Lines if pyarrow is unavailable; warn the user.
        logger.warning(
            "pyarrow not available; falling back to landscape.jsonl. "
            "Install pyarrow for the canonical Parquet output."
        )
        parquet_path = output_dir / "landscape.jsonl"
        df.to_json(parquet_path, orient="records", lines=True)

    written: dict[str, Path] = {"landscape": parquet_path}

    # -------- Summary JSON --------
    summary_path = output_dir / "landscape_summary.json"
    summary = _build_summary(
        df=df,
        runtime_seconds=runtime_seconds,
        defaults_used=defaults_used,
        constraints=constraints,
        code_version=code_ver,
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    written["summary"] = summary_path

    # -------- Geometry extraction report --------
    geo_path = output_dir / "geometry_extraction_report.json"
    geo_report = {
        "extraction_version": GEOMETRY_EXTRACTION_VERSION,
        "geometry": asdict(geometry),
        "warnings": geometry.extraction_warnings,
    }
    geo_path.write_text(json.dumps(geo_report, indent=2), encoding="utf-8")
    written["geometry_report"] = geo_path

    # -------- Optional CSV mirror --------
    if write_csv:
        csv_path = output_dir / "landscape.csv"
        df.to_csv(csv_path, index=False)
        written["csv"] = csv_path

    logger.info("Wrote outputs to %s: %s", output_dir, list(written.keys()))
    return written


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------

def _design_to_row(design: CharacterizedDesign, code_version: str) -> dict[str, Any]:
    """Convert one CharacterizedDesign into a dict matching the Parquet schema."""
    w = design.winding
    t = w.topology

    params_for_hash = {
        "Q": t.Q,
        "p": t.pole_count,
        "L": t.layers,
        "y": t.coil_pitch_slots,
        "paths": t.parallel_paths,
        "N": w.turns_per_coil,
        "AWG": w.wire_gauge_AWG,
        "strands": w.strands_per_conductor,
        "method": w.manufacturing_method,
    }
    h = design_hash(params_for_hash)

    return {
        "design_hash": h,
        "topology_id": t.topology_id(),
        "slot_count": int(t.Q),
        "pole_count": int(t.pole_count),
        "layers": int(t.layers),
        "coil_pitch_slots": int(t.coil_pitch_slots),
        "phase_count": 3,
        "parallel_paths": int(t.parallel_paths),
        "turns_per_coil": int(w.turns_per_coil),
        "wire_gauge_AWG": int(w.wire_gauge_AWG),
        "wire_diameter_bare_mm": float(w.wire_diameter_bare_mm),
        "wire_diameter_insulated_mm": float(w.wire_diameter_insulated_mm),
        "strands_per_conductor": int(w.strands_per_conductor),
        "manufacturing_method": w.manufacturing_method,
        "connection_matrix_json": json.dumps(t.connection_matrix.tolist()),
        "winding_factor_fundamental": float(t.winding_factor_fundamental),
        "winding_factor_5th": float(t.winding_factor_5th),
        "winding_factor_7th": float(t.winding_factor_7th),
        "winding_factor_11th": float(t.winding_factor_11th),
        "winding_factor_13th": float(t.winding_factor_13th),
        "cogging_period_mech_deg": float(t.cogging_period_mech_deg),
        "is_balanced": bool(t.is_balanced),
        "slot_fill_factor_actual": float(w.slot_fill_factor_actual),
        "slot_fill_factor_limit": float(w.slot_fill_factor_limit),
        "current_density_A_per_mm2": float(w.current_density_at_target_continuous_A_per_mm2),
        "phase_resistance_ohm_at_20C": float(w.R_phase_ohm_20C),
        "phase_inductance_synchronous_mH": float(design.Ls_synchronous_mH),
        "Kt_analytical_Nm_per_A": float(design.Kt_Nm_per_A),
        "Ke_analytical_V_per_rad_per_s": float(design.Ke_V_per_rad_per_s),
        "peak_torque_at_max_current_Nm": float(design.peak_torque_at_max_current_Nm),
        "continuous_torque_thermal_limit_Nm": float(design.continuous_torque_thermal_limit_Nm),
        "max_speed_at_supply_voltage_rpm": float(design.max_speed_at_supply_voltage_rpm),
        "predicted_winding_temp_at_continuous_C": float(
            design.predicted_winding_temp_at_continuous_C
        ),
        "predicted_iron_loss_at_max_speed_W": float(design.iron_loss_at_max_speed_W),
        "predicted_copper_loss_at_continuous_W": float(design.copper_loss_at_continuous_W),
        "feasibility_status": design.feasibility_status,
        "feasibility_notes": design.feasibility_notes,
        "end_turn_length_estimate_mm": float(w.end_turn_length_estimate_mm),
        "total_copper_mass_g": float(w.total_copper_mass_g),
        "geometry_extraction_version": GEOMETRY_EXTRACTION_VERSION,
        "materials_db_version": MATERIALS_DB_VERSION,
        "code_version": code_version,
    }


def _enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to deterministic dtypes to keep Parquet schema stable across runs."""
    int_cols = [
        "slot_count", "pole_count", "layers", "coil_pitch_slots", "phase_count",
        "parallel_paths", "turns_per_coil", "wire_gauge_AWG", "strands_per_conductor",
    ]
    bool_cols = ["is_balanced"]
    str_cols = [
        "design_hash", "topology_id", "manufacturing_method", "connection_matrix_json",
        "feasibility_status", "feasibility_notes",
        "geometry_extraction_version", "materials_db_version", "code_version",
    ]

    for c in int_cols:
        df[c] = df[c].astype("int64")
    for c in bool_cols:
        df[c] = df[c].astype("bool")
    for c in str_cols:
        df[c] = df[c].astype("string")
    # Everything else float64.
    for c in df.columns:
        if c not in int_cols + bool_cols + str_cols:
            df[c] = df[c].astype("float64")
    return df


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _build_summary(
    df: pd.DataFrame,
    runtime_seconds: float,
    defaults_used: list[str],
    constraints: Constraints,
    code_version: str,
) -> dict[str, Any]:
    """Build the landscape_summary.json contents."""
    status_counts = dict(Counter(df["feasibility_status"].tolist()))
    topology_counts = dict(Counter(df["topology_id"].tolist()))

    if len(df) > 0:
        ranges = {
            "turns_per_coil": [int(df["turns_per_coil"].min()), int(df["turns_per_coil"].max())],
            "wire_gauge_AWG": [int(df["wire_gauge_AWG"].min()), int(df["wire_gauge_AWG"].max())],
            "winding_factor_fundamental": [
                float(df["winding_factor_fundamental"].min()),
                float(df["winding_factor_fundamental"].max()),
            ],
            "slot_fill_factor_actual": [
                float(df["slot_fill_factor_actual"].min()),
                float(df["slot_fill_factor_actual"].max()),
            ],
        }
    else:
        ranges = {}

    return {
        "total_designs": int(len(df)),
        "feasibility_counts": status_counts,
        "topology_counts": topology_counts,
        "parameter_ranges": ranges,
        "runtime_seconds": round(runtime_seconds, 2),
        "code_version": code_version,
        "materials_db_version": MATERIALS_DB_VERSION,
        "geometry_extraction_version": GEOMETRY_EXTRACTION_VERSION,
        "defaults_applied": defaults_used,
        "constraints_used": json.loads(constraints.model_dump_json()),
    }
