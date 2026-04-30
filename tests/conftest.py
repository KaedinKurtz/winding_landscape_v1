"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from winding_landscape.config import Constraints
from winding_landscape.geometry.stator_geometry import StatorGeometry
from winding_landscape.materials.database import load_materials


@pytest.fixture(scope="session")
def materials_db():
    """Load the materials database once per test session."""
    return load_materials()


@pytest.fixture
def synthetic_geometry() -> StatorGeometry:
    """A 12-slot, 50 mm OD outrunner stator. Matches a typical hobby BLDC."""
    return StatorGeometry(
        bore_radius_mm=10.0,
        outer_radius_mm=25.0,
        stack_length_mm=20.0,
        slot_count=12,
        slot_opening_width_mm=2.0,
        slot_opening_depth_mm=1.0,
        slot_total_depth_mm=8.0,
        slot_bottom_width_mm=4.5,
        slot_area_mm2=30.0,
        slot_useful_area_mm2=27.0,
        tooth_width_min_mm=4.0,
        yoke_thickness_mm=7.0,
        airgap_mm=0.5,
        periodicity_tolerance_mm=0.01,
    )


@pytest.fixture
def reasonable_constraints() -> Constraints:
    """A constraint set sized for the synthetic_geometry fixture."""
    return Constraints(
        operating_targets={
            "target_continuous_torque_Nm": 0.15,
            "target_peak_torque_Nm": 0.5,
            "target_max_speed_rpm": 2000,
        },
        electrical_envelope={
            "supply_voltage_V": 36.0,
            "max_phase_current_A_rms": 10.0,
        },
    )
