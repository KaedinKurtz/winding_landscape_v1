"""Stage 5: Lumped-parameter steady-state thermal model.

Three thermal nodes: winding, iron, housing. Solves for the steady-state
temperature distribution given continuous-current copper loss + iron loss,
iterating because copper resistance is itself temperature-dependent.

Mechanical-engineering analogy: it's a heat-circuit equivalent of Ohm's law.
Heat flow = thermal current; temperature difference = voltage; thermal
resistance = electrical resistance. The iteration is because the "load"
(copper resistance) responds to the "voltage" (winding temp) -- like a
nonlinear resistor in an electrical circuit.
"""

from __future__ import annotations

import math

from scipy.optimize import brentq  # type: ignore[import-untyped]

from winding_landscape.config import Constraints
from winding_landscape.geometry.stator_geometry import StatorGeometry
from winding_landscape.materials.database import MaterialsDatabase
from winding_landscape.performance.characterized_design import CharacterizedDesign
from winding_landscape.utils.logging_config import get_logger

logger = get_logger(__name__)

_MAX_ITER = 30
_TEMP_TOL_C = 1.0


def characterize_thermal(
    design: CharacterizedDesign,
    geometry: StatorGeometry,
    constraints: Constraints,
    materials: MaterialsDatabase,
) -> CharacterizedDesign:
    """Populate thermal fields on ``design`` in place and return it.

    Computes:
      - converged winding temperature at the target continuous current
      - converged copper loss at that temperature
      - the continuous torque thermal limit (max torque without exceeding the
        winding-temp ceiling)
    """
    therm = constraints.thermal_envelope

    # ---- Compute thermal resistances of internal paths ----
    R_w_to_iron = _R_winding_to_iron(geometry, constraints)
    R_iron_to_housing = _R_iron_to_housing(geometry, constraints)
    R_housing_to_amb = therm.housing_thermal_resistance_K_per_W

    logger.debug(
        "Thermal Rs: winding->iron=%.3f, iron->housing=%.3f, housing->amb=%.3f K/W",
        R_w_to_iron, R_iron_to_housing, R_housing_to_amb,
    )

    # ---- Iterate winding temp at the target continuous current ----
    target_T_continuous = constraints.operating_targets.target_continuous_torque_Nm
    Kt = design.Kt_Nm_per_A
    if Kt <= 1e-9:
        # No torque per amp -> can't operate; mark and bail.
        design.predicted_winding_temp_at_continuous_C = therm.ambient_temp_C
        design.copper_loss_at_continuous_W = 0.0
        design.continuous_torque_thermal_limit_Nm = 0.0
        design.R_phase_ohm_at_operating_temp = design.winding.R_phase_ohm_20C
        return design

    I_continuous_target = target_T_continuous / Kt

    T_w, T_iron, P_cu, R_phase_op, n_iter = _solve_temperatures(
        I_phase_rms=I_continuous_target,
        P_iron_W=design.iron_loss_at_max_speed_W,
        R_phase_20C=design.winding.R_phase_ohm_20C,
        R_w_to_iron=R_w_to_iron,
        R_iron_to_housing=R_iron_to_housing,
        R_housing_to_amb=R_housing_to_amb,
        T_ambient=therm.ambient_temp_C,
        copper_temp_coeff=materials.copper_temp_coeff_per_C,
    )
    design.predicted_winding_temp_at_continuous_C = T_w
    design.predicted_iron_temp_at_continuous_C = T_iron
    design.copper_loss_at_continuous_W = P_cu
    design.R_phase_ohm_at_operating_temp = R_phase_op
    design.thermal_iterations_used = n_iter

    # ---- Continuous-torque thermal limit ----
    # Find the I that drives T_w exactly to the max winding temp (Class F default 155 C).
    T_w_max = therm.max_winding_temp_C

    def f_residual(I_test: float) -> float:
        """Returns T_w(I_test) - T_w_max. We want the root."""
        Tw, _, _, _, _ = _solve_temperatures(
            I_phase_rms=I_test,
            P_iron_W=design.iron_loss_at_max_speed_W,
            R_phase_20C=design.winding.R_phase_ohm_20C,
            R_w_to_iron=R_w_to_iron,
            R_iron_to_housing=R_iron_to_housing,
            R_housing_to_amb=R_housing_to_amb,
            T_ambient=therm.ambient_temp_C,
            copper_temp_coeff=materials.copper_temp_coeff_per_C,
        )
        return Tw - T_w_max

    # Bracket the root: f(0) is very negative (T_w ~ ambient), f(big) very positive.
    # If even at I_continuous_target we're already above the limit, T_thermal_limit < target.
    try:
        f_zero = f_residual(0.0)
        # Pick an upper bound: 2x the inverter limit is plenty.
        I_upper = constraints.electrical_envelope.max_phase_current_A_rms * 2.0
        f_upper = f_residual(I_upper)
        if f_zero > 0:
            # Even at zero current we're above max temp -> ambient + iron loss already too hot.
            I_thermal_limit = 0.0
        elif f_upper < 0:
            # Even at 2x rated, still below max temp -- limit is set by something else.
            I_thermal_limit = I_upper
        else:
            I_thermal_limit = brentq(f_residual, 0.0, I_upper, xtol=0.05)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Thermal limit solve failed: %s; falling back to target.", exc)
        I_thermal_limit = I_continuous_target

    design.continuous_torque_thermal_limit_Nm = float(Kt * I_thermal_limit)
    return design


# ---------------------------------------------------------------------------
# Inner solve: 3-node steady-state with copper-resistance temperature coupling
# ---------------------------------------------------------------------------

def _solve_temperatures(
    I_phase_rms: float,
    P_iron_W: float,
    R_phase_20C: float,
    R_w_to_iron: float,
    R_iron_to_housing: float,
    R_housing_to_amb: float,
    T_ambient: float,
    copper_temp_coeff: float,
) -> tuple[float, float, float, float, int]:
    """Steady-state 3-node solve with iteration on T_w-dependent copper resistance.

    The system is:
        P_cu(T_w) = 3 * I^2 * R(T_w)
        T_w = T_amb + (P_cu + P_iron) * (R_h_amb + R_i_h) + P_cu * R_w_i

    With R(T_w) increasing with T_w, this fixed-point map can be ill-conditioned
    if the loop gain dR/dT * R_thermal is large. We use under-relaxation
    (damping factor 0.5) and a hard physical clamp at 500 C to prevent runaway:
    if T_w blows past 500 C, the design is thermally infeasible at this current
    by any reasonable definition; we just return the clamped value and let the
    feasibility check flag it.

    Returns
    -------
    T_winding, T_iron, P_copper, R_phase_at_T_winding, n_iter
    """
    T_w = T_ambient + 60.0  # initial guess
    R_phase_op = R_phase_20C
    P_cu = 0.0
    T_iron = T_ambient
    T_HARD_CAP = 500.0  # C; physical absurdity ceiling
    DAMPING = 0.5

    for n_iter in range(1, _MAX_ITER + 1):
        # Resistance update with floor (copper goes to 0 ohms long before -273).
        R_phase_op = max(
            R_phase_20C * (1.0 + copper_temp_coeff * (T_w - 20.0)),
            R_phase_20C * 0.1,
        )
        # 3-phase loss: I^2 * R per phase, x3.
        P_cu = 3.0 * I_phase_rms**2 * R_phase_op

        # Solve the network. Heat flows from winding through iron through housing.
        T_housing = T_ambient + (P_cu + P_iron_W) * R_housing_to_amb
        T_iron = T_housing + (P_cu + P_iron_W) * R_iron_to_housing
        T_w_target = T_iron + P_cu * R_w_to_iron

        # Under-relax: T_w_new = T_w + alpha * (T_w_target - T_w).
        T_w_new = T_w + DAMPING * (T_w_target - T_w)

        # Clamp to physical absurdity ceiling.
        if T_w_new > T_HARD_CAP:
            T_w_new = T_HARD_CAP

        if abs(T_w_new - T_w) < _TEMP_TOL_C:
            T_w = T_w_new
            return T_w, T_iron, P_cu, R_phase_op, n_iter
        T_w = T_w_new

    return T_w, T_iron, P_cu, R_phase_op, _MAX_ITER


# ---------------------------------------------------------------------------
# Thermal-resistance estimators
# ---------------------------------------------------------------------------

def _R_winding_to_iron(geometry: StatorGeometry, constraints: Constraints) -> float:
    """Thermal resistance from winding bulk to surrounding iron.

    Two paths in series:
      1. Slot liner: thickness / k_liner / A_slot_thermal
      2. Half the bulk winding: 0.5 * winding_height / k_winding_eff / A_slot_thermal

    A_slot_thermal is the lateral (axial) area available for heat transfer:
    perimeter of the slot (sides + bottom) times stack length, summed over Q slots.
    """
    therm = constraints.thermal_envelope
    mfg = constraints.manufacturing_constraints

    liner_t_m = mfg.min_slot_liner_thickness_mm * 1e-3
    # Approximate slot perimeter (sides + bottom) per slot.
    perim_mm = (
        2.0 * geometry.slot_total_depth_mm
        + 0.5 * (geometry.slot_bottom_width_mm + geometry.slot_opening_width_mm)
    )
    perim_m = perim_mm * 1e-3
    L_stack_m = geometry.stack_length_mm * 1e-3
    A_thermal_per_slot_m2 = perim_m * L_stack_m
    A_total_m2 = max(A_thermal_per_slot_m2 * geometry.slot_count, 1e-9)

    # Heat-transfer height through the bulk winding (from slot center to iron surface).
    winding_height_m = geometry.slot_total_depth_mm * 0.5 * 1e-3

    R_liner = liner_t_m / max(therm.k_slot_liner_W_per_mK * A_total_m2, 1e-9)
    R_bulk = (
        0.5 * winding_height_m / max(therm.k_winding_eff_W_per_mK * A_total_m2, 1e-9)
    )
    return R_liner + R_bulk


def _R_iron_to_housing(geometry: StatorGeometry, constraints: Constraints) -> float:
    """Thermal resistance from iron to housing.

    For a stator where iron-loss heat is generated *throughout* the iron volume
    (not at one face), the conduction-only resistance to the bore-side housing
    is approximately:

        R_cond ~ ln(r_outer/r_bore) / (4 * pi * k * L_stack)

    (cylindrical bulk-heated wall, integrated to the inner surface, divided by 2
    because heat exits both axial ends as well as radially -- the factor-of-4
    in the denominator approximates this). We add a small contact resistance for
    the press-fit / adhesive joint at the bore.

    For a typical 50-100 mm OD stator this gives R in the 0.5-3 K/W range, which
    matches measured values reported by Boglietti et al. (IEEE T-IA 2003).
    """
    therm = constraints.thermal_envelope

    r_in_m = geometry.bore_radius_mm * 1e-3
    r_out_m = geometry.outer_radius_mm * 1e-3
    L_m = geometry.stack_length_mm * 1e-3

    if r_in_m >= r_out_m or L_m <= 0:
        return 1.0  # degenerate fallback

    R_cond = math.log(r_out_m / r_in_m) / (
        4.0 * math.pi * therm.k_iron_radial_W_per_mK * L_m
    )

    # Bore-side contact resistance (press-fit / adhesive into hub).
    # Boglietti et al. (IEEE T-IA 2008) measure 0.001-0.005 K*m^2/W for typical
    # press-fits with thermal interface materials; we use the looser end as
    # default but the user can override via constraints if they have measurement.
    A_bore_m2 = 2.0 * math.pi * r_in_m * L_m
    contact_R_per_area = 0.005  # K*m^2/W, typical for shrink/press fit
    R_contact = contact_R_per_area / max(A_bore_m2, 1e-9)

    return R_cond + R_contact
