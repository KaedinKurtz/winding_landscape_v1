"""CharacterizedDesign dataclass: contract from Stages 4-5 to Stage 6/7.

Wraps a :class:`WindingCandidate` and adds all the EM and thermal performance
fields. We use composition (a winding-candidate sub-object) rather than
inheritance because the dataclass progression is deep and we want each stage's
output to be inspectable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass

from winding_landscape.winding.winding_candidate import WindingCandidate


@dataclass
class CharacterizedDesign:
    """Fully-characterized design with EM and thermal performance fields populated."""

    winding: WindingCandidate

    # --- Stage 4 EM outputs ---
    Ke_V_per_rad_per_s: float = 0.0
    Kt_Nm_per_A: float = 0.0
    Ls_synchronous_mH: float = 0.0
    peak_torque_at_max_current_Nm: float = 0.0
    max_speed_at_supply_voltage_rpm: float = 0.0
    iron_loss_at_max_speed_W: float = 0.0
    tooth_flux_density_peak_T: float = 0.0
    yoke_flux_density_peak_T: float = 0.0
    saturation_factor: float = 1.0   # 1.0 = no saturation, <1.0 = derated
    flux_per_pole_Wb: float = 0.0    # exposed for downstream sanity checks

    # --- Stage 5 thermal outputs ---
    R_phase_ohm_at_operating_temp: float = 0.0
    predicted_winding_temp_at_continuous_C: float = 0.0
    predicted_iron_temp_at_continuous_C: float = 0.0
    copper_loss_at_continuous_W: float = 0.0
    continuous_torque_thermal_limit_Nm: float = 0.0
    thermal_iterations_used: int = 0

    # --- Stage 6 feasibility ---
    feasibility_status: str = "unknown"
    feasibility_notes: str = ""
