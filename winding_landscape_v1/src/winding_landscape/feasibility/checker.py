"""Stage 6: Apply all constraints and classify each design.

Per spec section 5, Stage 6, the priority order for reporting infeasibility is:
    fit > current > voltage > thermal

A design with multiple violations is reported with the highest-priority status,
and all violations are listed in the notes field.
"""

from __future__ import annotations

from winding_landscape.config import Constraints
from winding_landscape.performance.characterized_design import CharacterizedDesign


def classify_feasibility(
    design: CharacterizedDesign, constraints: Constraints
) -> CharacterizedDesign:
    """Populate ``feasibility_status`` and ``feasibility_notes`` on the design.

    Modifies the design in place (and returns it).
    """
    targets = constraints.operating_targets
    notes: list[str] = []

    # 1. Fit (highest priority)
    fit_failure = design.winding.fit_status != "fits"
    if fit_failure:
        notes.append(f"fit: {design.winding.fit_status}")

    # 2. Current/torque (peak torque limited)
    current_failure = design.peak_torque_at_max_current_Nm < targets.target_peak_torque_Nm
    if current_failure:
        notes.append(
            f"peak torque {design.peak_torque_at_max_current_Nm:.2f} Nm < "
            f"target {targets.target_peak_torque_Nm} Nm"
        )

    # 3. Voltage / max speed
    voltage_failure = (
        design.max_speed_at_supply_voltage_rpm < targets.target_max_speed_rpm
    )
    if voltage_failure:
        notes.append(
            f"max speed {design.max_speed_at_supply_voltage_rpm:.0f} rpm < "
            f"target {targets.target_max_speed_rpm} rpm"
        )

    # 4. Thermal / continuous torque
    thermal_failure = (
        design.continuous_torque_thermal_limit_Nm < targets.target_continuous_torque_Nm
    )
    if thermal_failure:
        notes.append(
            f"thermal-limited continuous torque {design.continuous_torque_thermal_limit_Nm:.2f} Nm < "
            f"target {targets.target_continuous_torque_Nm} Nm"
        )

    # Set status by priority.
    if fit_failure:
        design.feasibility_status = "infeasible_fit"
    elif current_failure:
        design.feasibility_status = "infeasible_current"
    elif voltage_failure:
        design.feasibility_status = "infeasible_voltage"
    elif thermal_failure:
        design.feasibility_status = "infeasible_thermal"
    else:
        design.feasibility_status = "feasible"

    design.feasibility_notes = "; ".join(notes) if notes else "all constraints satisfied"
    return design
