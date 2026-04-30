"""Materials database loader.

Loads the bundled JSON files into typed dataclasses and exposes a single
:class:`MaterialsDatabase` object.

Mechanical-engineering analogy: this is the materials handbook on the shelf.
You don't recompute B_r from first principles every time you size a magnet --
you look it up. Same idea here.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_DATA_DIR = Path(__file__).parent / "data"

# Materials DB version. Bump when JSON schemas or canonical values change.
MATERIALS_DB_VERSION = "v1.0.0"


# ---------------------------------------------------------------------------
# Typed records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SteelGrade:
    name: str
    description: str
    lamination_thickness_mm: float
    stacking_factor: float
    density_kg_per_m3: float
    BH_curve: NDArray[np.float64]  # shape (N, 2): columns are H[A/m], B[T]
    steinmetz_kh: float
    steinmetz_alpha: float
    steinmetz_beta: float
    eddy_ke: float
    saturation_B_T: float

    def H_at_B(self, B: float) -> float:
        """Inverse BH lookup: given a target B in T, return H in A/m via interpolation."""
        Bs = self.BH_curve[:, 1]
        Hs = self.BH_curve[:, 0]
        if B <= Bs[0]:
            return float(Hs[0])
        if B >= Bs[-1]:
            # Beyond saturation: extrapolate linearly with steep slope (deep saturation).
            slope = (Hs[-1] - Hs[-2]) / max(Bs[-1] - Bs[-2], 1e-9)
            return float(Hs[-1] + slope * (B - Bs[-1]))
        return float(np.interp(B, Bs, Hs))


@dataclass(frozen=True)
class MagnetGrade:
    name: str
    description: str
    B_r_at_20C_T: float
    B_r_temp_coefficient_per_C: float       # per-C fraction (negative)
    demagnetization_knee_at_20C_T: float
    demagnetization_temp_coefficient_per_C: float
    max_operating_temp_C: float
    density_kg_per_m3: float
    mu_recoil: float

    def B_r_at_temp(self, temp_C: float) -> float:
        """Temperature-corrected remanence."""
        return self.B_r_at_20C_T * (1.0 + self.B_r_temp_coefficient_per_C * (temp_C - 20.0))


@dataclass(frozen=True)
class InsulationClass:
    name: str
    max_temp_C: float
    description: str


@dataclass(frozen=True)
class AwgEntry:
    awg: int
    diameter_bare_mm: float
    diameter_insulated_mm: float
    area_bare_mm2: float


# ---------------------------------------------------------------------------
# Top-level container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MaterialsDatabase:
    steels: dict[str, SteelGrade]
    magnets: dict[str, MagnetGrade]
    insulation: dict[str, InsulationClass]
    awg: dict[int, AwgEntry]
    version: str = MATERIALS_DB_VERSION

    # Copper properties (constants, not loaded from JSON in V1).
    copper_resistivity_20C_ohm_m: float = field(default=1.724e-8)
    copper_temp_coeff_per_C: float = field(default=0.00393)
    copper_density_kg_per_m3: float = field(default=8960.0)

    def copper_resistivity_at(self, temp_C: float) -> float:
        """Copper resistivity at a given temperature (Ohm-m)."""
        return self.copper_resistivity_20C_ohm_m * (
            1.0 + self.copper_temp_coeff_per_C * (temp_C - 20.0)
        )

    def get_awg(self, gauge: int) -> AwgEntry:
        if gauge not in self.awg:
            raise KeyError(
                f"AWG {gauge} not in database. Available: {sorted(self.awg.keys())}"
            )
        return self.awg[gauge]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_materials(data_dir: Path | None = None) -> MaterialsDatabase:
    """Load all JSON files from the materials data directory."""
    base = Path(data_dir) if data_dir else _DATA_DIR

    steels_raw = _load_json(base / "steels.json")
    magnets_raw = _load_json(base / "magnets.json")
    insulation_raw = _load_json(base / "insulation.json")
    awg_raw = _load_json(base / "awg_table.json")

    steels = {
        name: SteelGrade(
            name=name,
            description=blob["description"],
            lamination_thickness_mm=float(blob["lamination_thickness_mm"]),
            stacking_factor=float(blob["stacking_factor"]),
            density_kg_per_m3=float(blob["density_kg_per_m3"]),
            BH_curve=np.asarray(blob["BH_curve"], dtype=np.float64),
            steinmetz_kh=float(blob["steinmetz_kh"]),
            steinmetz_alpha=float(blob["steinmetz_alpha"]),
            steinmetz_beta=float(blob["steinmetz_beta"]),
            eddy_ke=float(blob["eddy_ke"]),
            saturation_B_T=float(blob["saturation_B_T"]),
        )
        for name, blob in steels_raw.items()
        if not name.startswith("_")
    }

    magnets = {
        name: MagnetGrade(
            name=name,
            description=blob["description"],
            B_r_at_20C_T=float(blob["B_r_at_20C_T"]),
            B_r_temp_coefficient_per_C=float(blob["B_r_temp_coefficient_per_C"]),
            demagnetization_knee_at_20C_T=float(blob["demagnetization_knee_at_20C_T"]),
            demagnetization_temp_coefficient_per_C=float(
                blob["demagnetization_temp_coefficient_per_C"]
            ),
            max_operating_temp_C=float(blob["max_operating_temp_C"]),
            density_kg_per_m3=float(blob["density_kg_per_m3"]),
            mu_recoil=float(blob["mu_recoil"]),
        )
        for name, blob in magnets_raw.items()
        if not name.startswith("_")
    }

    insulation = {
        name: InsulationClass(
            name=name,
            max_temp_C=float(blob["max_temp_C"]),
            description=blob["description"],
        )
        for name, blob in insulation_raw.items()
        if not name.startswith("_")
    }

    awg = {
        int(awg_str): AwgEntry(
            awg=int(awg_str),
            diameter_bare_mm=float(blob["diameter_bare_mm"]),
            diameter_insulated_mm=float(blob["diameter_insulated_mm"]),
            area_bare_mm2=float(blob["area_bare_mm2"]),
        )
        for awg_str, blob in awg_raw.items()
        if not awg_str.startswith("_")
    }

    # Sanity-check the AWG entries against the standard formula. Tolerance 1%.
    for entry in awg.values():
        expected = 0.127 * math.pow(92.0, (36.0 - entry.awg) / 39.0)
        rel_err = abs(entry.diameter_bare_mm - expected) / expected
        if rel_err > 0.01:
            raise ValueError(
                f"AWG {entry.awg} bare diameter {entry.diameter_bare_mm} disagrees with "
                f"formula prediction {expected:.4f} by {rel_err * 100:.2f}%"
            )

    return MaterialsDatabase(
        steels=steels, magnets=magnets, insulation=insulation, awg=awg
    )


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object at top level")
    return data
