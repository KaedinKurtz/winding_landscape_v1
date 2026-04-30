"""StatorGeometry dataclass: the contract between Stage 1 and the rest of the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StatorGeometry:
    """Extracted geometric description of a single radial-flux outrunner stator.

    All linear dimensions are in millimetres, areas in mm^2, angles in degrees.

    Mechanical-engineering analogy: this is the equivalent of the bill-of-dimensions
    you'd write at the top of a stress calculation -- everything the analytical
    pipeline downstream needs, sitting in one place. After this, no module touches
    the BREP file directly; they all read from this struct.
    """

    bore_radius_mm: float
    """Inner shaft-bore radius (the bore the rotor shaft passes through)."""

    outer_radius_mm: float
    """Outer envelope radius -- the OD against which the rotor magnets fly."""

    stack_length_mm: float
    """Axial length of the stator iron stack."""

    slot_count: int
    """Q -- number of slots."""

    slot_opening_width_mm: float
    """Tangential width of the slot opening at the OD (the gap between tooth tips)."""

    slot_opening_depth_mm: float
    """Radial depth of the parallel-sided opening before the slot widens out."""

    slot_total_depth_mm: float
    """Radial extent of the slot, OD inward to the slot bottom."""

    slot_bottom_width_mm: float
    """Tangential chord at the radial bottom of the slot (the wider end)."""

    slot_area_mm2: float
    """Per-slot planimetric area of the cavity (before subtracting the liner)."""

    slot_useful_area_mm2: float
    """Slot area minus the allowance for a slot liner of the configured thickness."""

    tooth_width_min_mm: float
    """Minimum tangential width of a tooth (typically at the slot-opening level)."""

    yoke_thickness_mm: float
    """Radial distance from slot bottom to the bore."""

    airgap_mm: float
    """Mechanical airgap. Taken from constraints YAML since BREP is stator-only."""

    periodicity_tolerance_mm: float
    """Largest deviation found when verifying that all Q slots are congruent."""

    extraction_warnings: list[str] = field(default_factory=list)
    """Non-fatal issues that occurred during extraction."""

    def slot_pitch_mech_deg(self) -> float:
        """Mechanical angle subtended by one slot pitch (360/Q)."""
        return 360.0 / self.slot_count

    def slot_pitch_arclength_mm(self) -> float:
        """Arclength at the airgap of one slot pitch (2*pi*r/Q at OD)."""
        import math
        return 2.0 * math.pi * self.outer_radius_mm / self.slot_count
