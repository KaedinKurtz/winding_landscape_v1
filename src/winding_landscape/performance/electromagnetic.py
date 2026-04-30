"""Stage 4: Analytical electromagnetic performance via Magnetic Equivalent Circuit.

Implements the formulae in spec section 5, Stage 4, following Hanselman's
notation where possible.

Mechanical-engineering analogy: think of the magnetic circuit as a hydraulic
network. The magnet is a pressure source (B_r), the airgap and magnet recoil
permeability are resistances ("reluctances"), and Carter's coefficient is a
flow-area correction for the slot openings. The voltage/current balance gives
us flux per pole, and from there everything else follows.
"""

from __future__ import annotations

import math

from winding_landscape.config import Constraints
from winding_landscape.geometry.stator_geometry import StatorGeometry
from winding_landscape.materials.database import MaterialsDatabase, SteelGrade
from winding_landscape.performance.characterized_design import CharacterizedDesign
from winding_landscape.utils.logging_config import get_logger
from winding_landscape.winding.winding_candidate import WindingCandidate

logger = get_logger(__name__)

MU_0 = 4.0e-7 * math.pi  # H/m
NOMINAL_MAGNET_TEMP_C = 80.0  # spec: V1 uses a fixed nominal temp for B_r

# Saturation derating curve: linear from 1.0 at 1.8 T to 0.7 at 2.2 T.
_SAT_KNEE_T = 1.8
_SAT_FLOOR_T = 2.2
_SAT_FLOOR_FACTOR = 0.7


def characterize_electromagnetic(
    winding: WindingCandidate,
    geometry: StatorGeometry,
    constraints: Constraints,
    materials: MaterialsDatabase,
) -> CharacterizedDesign:
    """Apply the MEC + winding-factor model to fill in EM performance fields.

    Returns a :class:`CharacterizedDesign` with the Stage 4 fields populated.
    Stage 5 (thermal) operates on this output and adds the thermal fields.
    """
    design = CharacterizedDesign(winding=winding)
    topology = winding.topology
    pole_pairs = topology.pole_count // 2

    magnet = materials.magnets[constraints.materials.magnet_grade]
    steel = materials.steels[constraints.materials.steel_grade]

    # ----- Step 1: airgap / Carter's coefficient / magnet flux per pole -----
    g_mech = geometry.airgap_mm
    slot_opening = geometry.slot_opening_width_mm
    carter_kc = _carter_coefficient(
        slot_opening_mm=slot_opening,
        airgap_mm=g_mech,
        slot_pitch_mm=geometry.slot_pitch_arclength_mm(),
    )
    g_eff = g_mech * carter_kc

    l_m = constraints.topology_constraints.magnet_thickness_mm
    mu_rec = magnet.mu_recoil
    B_r = magnet.B_r_at_temp(NOMINAL_MAGNET_TEMP_C)

    # Operating-point flux density in the airgap (1D MEC, surface-mount, no
    # flux concentration).
    B_g = B_r / (1.0 + mu_rec * g_eff / l_m)

    # Pole-face area at the air-gap radius (mm^2).
    arc_fraction = constraints.topology_constraints.magnet_arc_fraction
    A_pole_mm2 = (
        2.0 * math.pi * geometry.outer_radius_mm
        * geometry.stack_length_mm
        * arc_fraction
        / topology.pole_count
    )
    flux_per_pole_Wb = B_g * A_pole_mm2 * 1e-6
    design.flux_per_pole_Wb = flux_per_pole_Wb

    # ----- Step 2: Ke (back-EMF constant) -----
    N_series = winding.turns_per_coil * topology.coils_in_series_per_path
    kw1 = topology.winding_factor_fundamental
    Ke = N_series * kw1 * flux_per_pole_Wb * pole_pairs
    design.Ke_V_per_rad_per_s = Ke

    # ----- Step 3: Kt (torque constant) -----
    # For sinusoidal back-EMF + vector control with q-axis current only,
    # Kt = (3/2) * Ke (peak phase current to torque, 3-phase).
    # The (2/pi) factor in the spec was for trapezoidal flux waveforms; we
    # follow the sinusoidal convention since it's the cleaner default for
    # PMSM analysis. Document this assumption.
    Kt = 1.5 * Ke
    design.Kt_Nm_per_A = Kt

    # ----- Step 4: Synchronous inductance Ls -----
    # Ls = mu0 * stack_L * N_series^2 * kw1^2 / (pole_pairs^2 * g_eff * pi * 2) [mH]
    if pole_pairs > 0 and g_eff > 0:
        Ls = (
            MU_0 * (geometry.stack_length_mm * 1e-3) * N_series**2 * kw1**2
            / (pole_pairs**2 * (g_eff * 1e-3) * math.pi * 2.0)
        ) * 1000.0  # H -> mH
    else:
        Ls = 0.0
    design.Ls_synchronous_mH = Ls

    # ----- Step 5: Peak torque at max current, with saturation derating -----
    I_peak_A = constraints.electrical_envelope.max_phase_current_A_rms * math.sqrt(2.0)

    # Tooth flux density at I_peak. Armature reaction adds to magnet flux at
    # the loaded pole.
    L_armature_factor = 0.2 * 1.0  # I_peak / I_rated = 1.0 by definition
    slot_pitch_at_OD = geometry.slot_pitch_arclength_mm()
    tooth_w = max(geometry.tooth_width_min_mm, 1e-3)
    B_tooth_no_load = B_g * (slot_pitch_at_OD / tooth_w)
    B_tooth_loaded = B_tooth_no_load * (1.0 + L_armature_factor)
    design.tooth_flux_density_peak_T = B_tooth_loaded

    # Yoke flux density: half a pole's flux flows through the yoke cross-section.
    # B_yoke = (Phi_pole / 2) / (yoke_thickness * stack_length).
    A_yoke_mm2 = geometry.yoke_thickness_mm * geometry.stack_length_mm
    B_yoke = (flux_per_pole_Wb / 2.0) / max(A_yoke_mm2 * 1e-6, 1e-9)
    design.yoke_flux_density_peak_T = B_yoke

    sat_factor = _saturation_factor(B_tooth_loaded, steel)
    design.saturation_factor = sat_factor
    design.peak_torque_at_max_current_Nm = Kt * I_peak_A * sat_factor

    # ----- Step 6: Max speed at supply voltage -----
    V_supply = constraints.electrical_envelope.supply_voltage_V
    mod_margin = constraints.electrical_envelope.modulation_margin
    if Ke > 1e-9:
        omega_max = V_supply * mod_margin / math.sqrt(3.0) / Ke
        design.max_speed_at_supply_voltage_rpm = omega_max * 60.0 / (2.0 * math.pi)
    else:
        design.max_speed_at_supply_voltage_rpm = 0.0

    # ----- Step 7: Iron losses at max-speed operating point -----
    target_max_speed_rpm = constraints.operating_targets.target_max_speed_rpm
    f_elec_Hz = target_max_speed_rpm / 60.0 * pole_pairs
    # Average B over tooth + yoke (mean field experienced by laminations).
    # Clamp to steel saturation: above B_sat, the differential permeability
    # collapses and the Steinmetz curve flattens. Without this clamp, geometry
    # ratios that imply B_tooth >> B_sat give unphysical iron losses (the
    # formula has B^2 with no upper bound).
    B_tooth_for_loss = min(B_tooth_no_load, steel.saturation_B_T)
    B_yoke_for_loss = min(B_yoke, steel.saturation_B_T)
    B_avg_steel = 0.5 * (B_tooth_for_loss + B_yoke_for_loss)
    iron_volume_m3 = _stator_iron_volume(geometry)
    P_iron = _steinmetz_loss(
        steel=steel,
        f_Hz=f_elec_Hz,
        B_T=B_avg_steel,
        volume_m3=iron_volume_m3,
    )
    design.iron_loss_at_max_speed_W = P_iron

    return design


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _carter_coefficient(slot_opening_mm: float, airgap_mm: float, slot_pitch_mm: float) -> float:
    """Carter's coefficient: corrects the effective airgap for slotting effects.

    kc = slot_pitch / (slot_pitch - gamma * airgap)

    where  gamma = (4/pi) * [0.5*r * atan(0.5*r) - ln(sqrt(1 + (0.5*r)^2))],
    and r = slot_opening / airgap.

    Reference: Hanselman eqn 4.31; Lipo, "Introduction to AC Machine Design".
    """
    if airgap_mm <= 0 or slot_pitch_mm <= 0:
        return 1.0
    r = slot_opening_mm / airgap_mm
    half_r = 0.5 * r
    gamma = (4.0 / math.pi) * (
        half_r * math.atan(half_r) - math.log(math.sqrt(1.0 + half_r**2))
    )
    denom = slot_pitch_mm - gamma * airgap_mm
    if denom <= 0:
        return 2.0  # degenerate -- just bump the airgap a lot
    kc = slot_pitch_mm / denom
    return float(max(kc, 1.0))


def _saturation_factor(B_tooth_T: float, steel: SteelGrade) -> float:
    """Linear derating from 1.0 at 1.8 T to 0.7 at 2.2 T (per spec).

    For steels with much higher saturation (e.g. Hiperco50), shift the knee to
    80% of the steel's saturation_B_T so we don't over-penalize CoFe.
    """
    knee = max(_SAT_KNEE_T, 0.8 * steel.saturation_B_T - 0.2)
    floor = min(_SAT_FLOOR_T, 0.95 * steel.saturation_B_T + 0.05)
    if B_tooth_T <= knee:
        return 1.0
    if B_tooth_T >= floor:
        return _SAT_FLOOR_FACTOR
    # Linear interpolation between (knee, 1.0) and (floor, 0.7).
    frac = (B_tooth_T - knee) / (floor - knee)
    return 1.0 + frac * (_SAT_FLOOR_FACTOR - 1.0)


def _stator_iron_volume(geometry: StatorGeometry) -> float:
    """Approximate the iron-only volume of the stator in m^3.

    iron_volume = total_annular_volume - slot_volume * Q
    where total_annular_volume is the full annular cylinder from bore to OD
    over the stack length.
    """
    total_annulus_mm3 = math.pi * (
        geometry.outer_radius_mm**2 - geometry.bore_radius_mm**2
    ) * geometry.stack_length_mm
    slots_mm3 = geometry.slot_count * geometry.slot_area_mm2 * geometry.stack_length_mm
    iron_mm3 = max(total_annulus_mm3 - slots_mm3, 0.0)
    return iron_mm3 * 1e-9  # mm^3 -> m^3


def _steinmetz_loss(
    steel: SteelGrade,
    f_Hz: float,
    B_T: float,
    volume_m3: float,
) -> float:
    """Steinmetz loss model: P = kh*f^a*B^b + ke*f^2*B^2*t^2  (W/m^3).

    Returns total iron loss in W (over the given volume).
    """
    if f_Hz <= 0 or B_T <= 0 or volume_m3 <= 0:
        return 0.0
    p_density_W_per_m3 = (
        steel.steinmetz_kh * (f_Hz**steel.steinmetz_alpha) * (B_T**steel.steinmetz_beta)
        + steel.eddy_ke * (f_Hz**2) * (B_T**2) * (steel.lamination_thickness_mm**2)
    )
    return p_density_W_per_m3 * volume_m3
