"""Configuration schema for the winding landscape generator.

Uses pydantic v2 for validation. All fields have sensible defaults so the user can
provide a partial YAML/JSON file. Defaults that are applied are recorded in the
run log via the ``defaults_used`` machinery in ``load_constraints``.

Mechanical-engineering analogy: think of this as the "design requirements"
document a customer hands you before you start sizing a gearbox. Each field is
a constraint the final design must satisfy or stay within.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator
except ImportError:  # pragma: no cover -- exercised in CI envs without pydantic
    from winding_landscape._pydantic_shim import (
        BaseModel,
        ConfigDict,
        Field,
        field_validator,
    )


# ---------------------------------------------------------------------------
# Sub-sections
# ---------------------------------------------------------------------------

class OperatingTargets(BaseModel):
    """Performance targets the design must hit."""

    model_config = ConfigDict(extra="forbid")

    target_continuous_torque_Nm: float = Field(default=5.0, gt=0)
    target_peak_torque_Nm: float = Field(default=15.0, gt=0)
    target_max_speed_rpm: float = Field(default=6000.0, gt=0)


class ElectricalEnvelope(BaseModel):
    """Inverter and bus limits."""

    model_config = ConfigDict(extra="forbid")

    supply_voltage_V: float = Field(default=48.0, gt=0)
    max_phase_current_A_rms: float = Field(default=30.0, gt=0)
    modulation_margin: float = Field(default=0.90, gt=0, le=1.0)


class ThermalEnvelope(BaseModel):
    """Thermal limits and cooling configuration."""

    model_config = ConfigDict(extra="forbid")

    ambient_temp_C: float = 40.0
    max_winding_temp_C: float = 155.0
    max_magnet_temp_C: float = 100.0
    cooling_mode: Literal["natural", "forced_air", "liquid"] = "natural"
    housing_thermal_resistance_K_per_W: float = Field(default=1.5, gt=0)
    # Optional overrides for internal thermal resistances (W/m/K). If None, defaults are used.
    k_slot_liner_W_per_mK: float = Field(default=0.2, gt=0)
    k_winding_eff_W_per_mK: float = Field(default=0.5, gt=0)
    k_iron_radial_W_per_mK: float = Field(default=25.0, gt=0)


class MaterialsSelection(BaseModel):
    """Lookup keys into the materials database."""

    model_config = ConfigDict(extra="forbid")

    steel_grade: str = "M19_29Ga"
    magnet_grade: str = "N42SH"
    wire_insulation_class: str = "F"


class ManufacturingConstraints(BaseModel):
    """Practical limits on what can actually be built."""

    model_config = ConfigDict(extra="forbid")

    available_wire_gauges_AWG: list[int] = Field(
        default_factory=lambda: [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    )
    min_slot_liner_thickness_mm: float = Field(default=0.25, gt=0)
    achievable_fill_factor_random: float = Field(default=0.42, gt=0, le=0.78)
    min_turns_per_coil: int = Field(default=5, ge=1)
    max_turns_per_coil: int = Field(default=200, ge=1)

    @field_validator("available_wire_gauges_AWG")
    @classmethod
    def _validate_gauges(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("available_wire_gauges_AWG must not be empty")
        if any(g < 0 or g > 50 for g in v):
            raise ValueError("AWG values must be in [0, 50]")
        return sorted(set(v))


class TopologyConstraints(BaseModel):
    """Bounds on the (Q, 2p, layers, pitch) search space."""

    model_config = ConfigDict(extra="forbid")

    pole_count_range: tuple[int, int] = (4, 28)
    slot_pole_combos_to_consider: str | list[tuple[int, int]] = "auto"
    layer_options: list[Literal["single", "double"]] = Field(default_factory=lambda: ["double"])
    # Optional override for assumed mechanical airgap if not present in BREP.
    airgap_mm: float = Field(default=0.5, gt=0)
    # Magnet thickness (BREP is stator-only).
    magnet_thickness_mm: float = Field(default=3.0, gt=0)
    # Magnet arc fraction (fraction of pole pitch the magnet covers).
    magnet_arc_fraction: float = Field(default=0.83, gt=0, le=1.0)

    @field_validator("pole_count_range")
    @classmethod
    def _validate_pole_range(cls, v: tuple[int, int]) -> tuple[int, int]:
        lo, hi = v
        if lo < 2 or hi < lo:
            raise ValueError("pole_count_range must have 2 <= lo <= hi")
        if lo % 2 != 0 or hi % 2 != 0:
            raise ValueError("pole_count_range bounds must be even (poles come in pairs)")
        return v


class EnumerationDensity(BaseModel):
    """Knobs controlling how aggressively we sweep the design space."""

    model_config = ConfigDict(extra="forbid")

    target_designs_per_topology: int = Field(default=50, ge=1)
    turns_step: int = Field(default=1, ge=1)
    max_topologies: int = Field(default=20, ge=1)
    min_winding_factor_fundamental: float = Field(default=0.85, gt=0, lt=1.0)


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

class Constraints(BaseModel):
    """Full constraints specification.

    All sub-sections are optional in the YAML; missing sections fall back to defaults.
    """

    model_config = ConfigDict(extra="forbid")

    operating_targets: OperatingTargets = Field(default_factory=OperatingTargets)
    electrical_envelope: ElectricalEnvelope = Field(default_factory=ElectricalEnvelope)
    thermal_envelope: ThermalEnvelope = Field(default_factory=ThermalEnvelope)
    materials: MaterialsSelection = Field(default_factory=MaterialsSelection)
    manufacturing_constraints: ManufacturingConstraints = Field(
        default_factory=ManufacturingConstraints
    )
    topology_constraints: TopologyConstraints = Field(default_factory=TopologyConstraints)
    enumeration_density: EnumerationDensity = Field(default_factory=EnumerationDensity)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_constraints(path: Path | str) -> tuple[Constraints, list[str]]:
    """Load a constraints YAML/JSON file and return the validated model.

    Returns
    -------
    constraints : Constraints
        Validated constraints object.
    defaults_used : list[str]
        Dotted paths into the constraints tree where the user did not supply a
        value and the default was applied. Useful for the run log.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() in {".yaml", ".yml"}:
        raw: dict[str, Any] = yaml.safe_load(text) or {}
    elif path.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        raise ValueError(f"Unsupported constraints file extension: {path.suffix}")

    if not isinstance(raw, dict):
        raise ValueError("Constraints file must parse to a top-level mapping/object")

    defaults_used = _diff_against_defaults(raw, Constraints())
    return Constraints.model_validate(raw), defaults_used


def _diff_against_defaults(
    user_dict: dict[str, Any],
    defaults: Constraints,
    prefix: str = "",
) -> list[str]:
    """Return the list of dotted paths where the user omitted a field."""
    result: list[str] = []
    default_dump = defaults.model_dump()
    for key, default_val in default_dump.items():
        full_key = f"{prefix}{key}"
        if key not in user_dict:
            result.append(full_key)
            continue
        # Recurse into nested mappings.
        user_val = user_dict[key]
        if isinstance(default_val, dict) and isinstance(user_val, dict):
            sub_defaults = getattr(defaults, key)
            for sub_key, sub_default in default_val.items():
                if sub_key not in user_val:
                    result.append(f"{full_key}.{sub_key}")
    return result
