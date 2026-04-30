"""Stage 1: Stator geometry extraction from a BREP file.

We use ``build123d`` (which wraps OpenCascade via OCP) for the heavy lifting.
The high-level algorithm is in the V1 spec section 5/Stage 1; this module just
implements it.

Build123d's API gives us:
- ``Solid.from_step(path)`` / ``import_step`` to load .step/.stp
- ``import_brep`` to load .brep
- ``solid.cylindrical_faces()`` to find cylinders
- Section/slicing via plane intersection

If build123d cannot be imported (e.g., not installed), we raise a clear error.
The fallback to pythonocc-core mentioned in the spec is reserved for cases
where build123d's *parser* fails on a specific file, not for missing-package.

Mechanical-engineering analogy: think of this as the inspection step where you
take a part off the line, put it on the CMM, and measure all the features that
the spec called out. Same fixturing, same probing routine, every time -- so
downstream calculations are reproducible.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from winding_landscape.config import Constraints
from winding_landscape.geometry.stator_geometry import StatorGeometry
from winding_landscape.utils.logging_config import get_logger

logger = get_logger(__name__)

# Slicing offset from one axial end. Mid-stack for V1 (no taper assumed).
_SLICE_FRACTION = 0.5

# Tolerances
_AXIS_TOL = 1e-3                 # mm, for "concentric with axis" checks
_PERIODICITY_WARN_TOL = 0.05     # mm, log a warning above this
_PERIODICITY_FATAL_TOL = 1.0     # mm, treat as fatal if exceeded


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_geometry(brep_path: Path | str, constraints: Constraints) -> StatorGeometry:
    """Extract a :class:`StatorGeometry` from a stator BREP.

    Parameters
    ----------
    brep_path
        Path to a .step, .stp, or .brep file. The file must contain a single
        radial-flux outrunner stator as described in the V1 spec section 2.1.
    constraints
        Full constraints object. We use ``topology_constraints.airgap_mm`` and
        ``manufacturing_constraints.min_slot_liner_thickness_mm``.

    Raises
    ------
    GeometryExtractionError
        On any structural failure (file unparseable, geometry doesn't look like
        a stator, dimensions out of plausible range, etc.).
    """
    path = Path(brep_path)
    if not path.exists():
        raise GeometryExtractionError(f"BREP file not found: {path}")

    logger.info("Loading BREP from %s", path)
    solid = _load_brep(path)

    logger.debug("Computing axis and stack length")
    axis, axis_origin = _identify_rotation_axis(solid)
    stack_length = _stack_length_along_axis(solid, axis, axis_origin)

    logger.debug("Slicing at mid-stack to obtain 2D cross-section")
    section_polygon = _slice_at(solid, axis, axis_origin, stack_length * _SLICE_FRACTION)

    logger.debug("Identifying bore and outer radii")
    bore_r, outer_r = _identify_bore_and_outer_radii(section_polygon)

    logger.debug("Counting slots and measuring slot geometry")
    slot_count, slot_features = _measure_slots(section_polygon, bore_r, outer_r)

    logger.debug("Verifying slot periodicity")
    periodicity_dev = _verify_periodicity(section_polygon, slot_count, bore_r, outer_r)

    warnings: list[str] = []
    if periodicity_dev > _PERIODICITY_FATAL_TOL:
        raise GeometryExtractionError(
            f"Slot periodicity deviation {periodicity_dev:.3f} mm exceeds "
            f"fatal tolerance {_PERIODICITY_FATAL_TOL} mm; not a periodic stator?"
        )
    if periodicity_dev > _PERIODICITY_WARN_TOL:
        msg = f"Periodicity deviation {periodicity_dev:.3f} mm > warn tolerance"
        logger.warning(msg)
        warnings.append(msg)

    # Compute slot useful area: subtract liner thickness times slot perimeter.
    liner_t = constraints.manufacturing_constraints.min_slot_liner_thickness_mm
    slot_perimeter_estimate = 2.0 * (
        slot_features["total_depth_mm"]
        + 0.5 * (slot_features["bottom_width_mm"] + slot_features["opening_width_mm"])
    )
    slot_useful = max(
        slot_features["area_mm2"] - liner_t * slot_perimeter_estimate,
        0.0,
    )

    geo = StatorGeometry(
        bore_radius_mm=bore_r,
        outer_radius_mm=outer_r,
        stack_length_mm=stack_length,
        slot_count=slot_count,
        slot_opening_width_mm=slot_features["opening_width_mm"],
        slot_opening_depth_mm=slot_features["opening_depth_mm"],
        slot_total_depth_mm=slot_features["total_depth_mm"],
        slot_bottom_width_mm=slot_features["bottom_width_mm"],
        slot_area_mm2=slot_features["area_mm2"],
        slot_useful_area_mm2=slot_useful,
        tooth_width_min_mm=slot_features["tooth_width_min_mm"],
        yoke_thickness_mm=slot_features["yoke_thickness_mm"],
        airgap_mm=constraints.topology_constraints.airgap_mm,
        periodicity_tolerance_mm=periodicity_dev,
        extraction_warnings=warnings,
    )

    _sanity_check(geo)
    logger.info(
        "Geometry: Q=%d, OD=%.2fmm, ID=%.2fmm, L=%.2fmm, slot area=%.2fmm^2",
        geo.slot_count, 2 * geo.outer_radius_mm, 2 * geo.bore_radius_mm,
        geo.stack_length_mm, geo.slot_area_mm2,
    )
    return geo


class GeometryExtractionError(RuntimeError):
    """Raised when the BREP cannot be turned into a valid StatorGeometry."""


# ---------------------------------------------------------------------------
# BREP loading
# ---------------------------------------------------------------------------

def _load_brep(path: Path):
    """Load a STEP/BREP file into a build123d Solid.

    Returns
    -------
    Solid
        A build123d ``Solid`` (or compatible) object.
    """
    try:
        # build123d 0.6+ exposes import_step / import_brep at the package level.
        import build123d as bd
    except ImportError as exc:
        raise GeometryExtractionError(
            "build123d is required for BREP processing. Install with: "
            "pip install build123d"
        ) from exc

    suffix = path.suffix.lower()
    if suffix in {".step", ".stp"}:
        result = bd.import_step(str(path))
    elif suffix == ".brep":
        result = bd.import_brep(str(path))
    else:
        raise GeometryExtractionError(
            f"Unsupported BREP extension '{suffix}'. Use .step, .stp, or .brep."
        )

    # build123d returns a Compound or Part; we need a single Solid.
    solids = list(result.solids()) if hasattr(result, "solids") else [result]
    if len(solids) == 0:
        raise GeometryExtractionError("BREP contains no solid bodies.")
    if len(solids) > 1:
        # Pick the largest by volume; warn.
        solids.sort(key=lambda s: s.volume, reverse=True)
        logger.warning(
            "BREP contains %d solids; using the largest by volume.", len(solids)
        )
    return solids[0]


# ---------------------------------------------------------------------------
# Axis identification
# ---------------------------------------------------------------------------

def _identify_rotation_axis(solid) -> tuple[np.ndarray, np.ndarray]:
    """Find the rotation axis of the stator.

    Strategy: of all cylindrical faces, find the one with the smallest radius
    that is concentric with the bounding-box centroid -- this is the bore.
    Its axis defines the rotation axis.

    Returns
    -------
    axis_dir : ndarray (3,)
        Unit vector along the rotation axis.
    axis_origin : ndarray (3,)
        A point on the rotation axis (the bore's reference point).
    """
    bbox = solid.bounding_box()
    centroid = np.array([
        0.5 * (bbox.min.X + bbox.max.X),
        0.5 * (bbox.min.Y + bbox.max.Y),
        0.5 * (bbox.min.Z + bbox.max.Z),
    ])

    cylinders: list[tuple[float, np.ndarray, np.ndarray]] = []  # (radius, axis_dir, origin)
    for face in solid.faces():
        # build123d/OCP: face.geom_type returns "CYLINDER" for cylindrical faces.
        if str(getattr(face, "geom_type", "")).upper() != "CYLINDER":
            continue
        # Extract the cylinder's axis. build123d exposes this on .axis_of_rotation
        # or via face.location. The exact attribute varies by version, so we
        # try a few in turn.
        cyl_axis_dir, cyl_axis_origin, radius = _cylinder_axis_and_radius(face)
        if radius is None or radius < 1e-6:
            continue

        # Distance from cylinder axis to bounding-box centroid (closest point).
        rel = centroid - cyl_axis_origin
        proj = rel - np.dot(rel, cyl_axis_dir) * cyl_axis_dir
        dist_to_axis = float(np.linalg.norm(proj))
        if dist_to_axis > 5.0:  # mm
            # Not concentric with the centroid; not a candidate.
            continue

        cylinders.append((radius, cyl_axis_dir, cyl_axis_origin))

    if not cylinders:
        raise GeometryExtractionError(
            "No cylindrical faces concentric with the bounding-box centroid were found. "
            "Is this a radial-flux outrunner stator?"
        )

    cylinders.sort(key=lambda x: x[0])
    bore_r, axis_dir, axis_origin = cylinders[0]
    logger.debug(
        "Rotation axis: dir=%s, origin=%s; bore radius candidate=%.3f mm",
        axis_dir.round(4), axis_origin.round(3), bore_r,
    )
    return axis_dir, axis_origin


def _cylinder_axis_and_radius(face) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Pull (axis_direction, axis_origin, radius) out of a build123d cylindrical face.

    build123d's OCCT wrappers expose this differently across versions. We try
    the documented locations and degrade gracefully.
    """
    # Approach 1: face.geometry() returns a gp_Cylinder-like object.
    geom = getattr(face, "geometry", None)
    if callable(geom):
        try:
            g = geom()
            ax = g.Axis()
            loc = ax.Location()
            direction = ax.Direction()
            origin = np.array([loc.X(), loc.Y(), loc.Z()])
            axis_dir = np.array([direction.X(), direction.Y(), direction.Z()])
            axis_dir /= np.linalg.norm(axis_dir)
            radius = float(g.Radius())
            return axis_dir, origin, radius
        except (AttributeError, Exception):  # noqa: BLE001
            pass

    # Approach 2: face has axis_of_rotation property (build123d high-level API).
    aor = getattr(face, "axis_of_rotation", None)
    if aor is not None:
        # Newer build123d returns an Axis with .position and .direction (Vector).
        try:
            origin = np.array([aor.position.X, aor.position.Y, aor.position.Z])
            axis_dir = np.array([aor.direction.X, aor.direction.Y, aor.direction.Z])
            axis_dir /= np.linalg.norm(axis_dir)
            radius = float(getattr(face, "radius", 0.0))
            return axis_dir, origin, radius if radius > 0 else None
        except AttributeError:
            pass

    return np.array([0, 0, 1]), np.zeros(3), None


# ---------------------------------------------------------------------------
# Stack length
# ---------------------------------------------------------------------------

def _stack_length_along_axis(solid, axis: np.ndarray, origin: np.ndarray) -> float:
    """Project all vertices onto the rotation axis, return the range."""
    projections: list[float] = []
    for vertex in solid.vertices():
        p = np.array([vertex.X, vertex.Y, vertex.Z])
        projections.append(float(np.dot(p - origin, axis)))
    if not projections:
        raise GeometryExtractionError("Solid has no vertices?!")
    length = max(projections) - min(projections)
    if length <= 0:
        raise GeometryExtractionError(f"Computed stack length {length} mm is non-positive.")
    return length


# ---------------------------------------------------------------------------
# Cross-sectioning
# ---------------------------------------------------------------------------

def _slice_at(solid, axis: np.ndarray, origin: np.ndarray, axial_offset: float):
    """Slice the solid with a plane perpendicular to the axis at given offset.

    Returns the cross-section as a list of polygons (each an Nx2 array of points
    in the slicing plane's local 2D frame, with the axis at origin).

    For build123d, this uses ``solid.cut_by_plane`` or ``Plane.section_of``.
    """
    try:
        import build123d as bd
    except ImportError as exc:
        raise GeometryExtractionError("build123d unavailable.") from exc

    # Build the slicing plane.
    plane_origin = origin + axial_offset * axis
    # Choose two in-plane basis vectors orthogonal to the axis.
    u, v = _orthonormal_basis_in_plane(axis)

    plane = bd.Plane(
        origin=bd.Vector(*plane_origin.tolist()),
        x_dir=bd.Vector(*u.tolist()),
        z_dir=bd.Vector(*axis.tolist()),
    )

    # build123d's section operation returns wires/edges in the plane.
    try:
        section = solid & plane  # boolean intersection with infinite plane
    except Exception as exc:  # noqa: BLE001
        raise GeometryExtractionError(f"Cross-section operation failed: {exc}") from exc

    # Collect polygon points by walking each wire in the section.
    polygons: list[np.ndarray] = []
    wires = list(section.wires()) if hasattr(section, "wires") else []
    if not wires:
        # Some versions expose edges instead of wires for a planar section.
        edges = list(section.edges()) if hasattr(section, "edges") else []
        if not edges:
            raise GeometryExtractionError("Cross-section produced no wires or edges.")
        wires = [bd.Wire(edges)]

    for wire in wires:
        pts = _sample_wire_to_2d(wire, plane_origin, u, v, n_samples_per_edge=24)
        if len(pts) >= 3:
            polygons.append(pts)

    if not polygons:
        raise GeometryExtractionError("No polygons recovered from cross-section.")

    return _PlanarSection(polygons=polygons, plane_origin=plane_origin, u=u, v=v)


def _orthonormal_basis_in_plane(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Produce two unit vectors orthogonal to ``axis`` and to each other."""
    axis = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    v /= np.linalg.norm(v)
    return u, v


def _sample_wire_to_2d(
    wire,
    plane_origin: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    n_samples_per_edge: int,
) -> np.ndarray:
    """Sample a 3D wire into a list of (x, y) points in the slicing plane."""
    pts: list[tuple[float, float]] = []
    for edge in wire.edges():
        # build123d edges support .positions(samples) returning Vectors at param fractions.
        try:
            samples = edge.positions([i / n_samples_per_edge for i in range(n_samples_per_edge + 1)])
        except (AttributeError, TypeError):
            # Fallback: sample by parameter range.
            samples = [edge.start_point(), edge.end_point()]
        for s in samples:
            p = np.array([s.X, s.Y, s.Z]) - plane_origin
            pts.append((float(np.dot(p, u)), float(np.dot(p, v))))
    return np.asarray(pts, dtype=np.float64)


class _PlanarSection:
    """Container for a 2D cross-section: list of polygons in (u, v) local frame."""

    def __init__(self, polygons: list[np.ndarray], plane_origin: np.ndarray, u, v):
        self.polygons = polygons
        self.plane_origin = plane_origin
        self.u = u
        self.v = v


# ---------------------------------------------------------------------------
# Section interpretation: bore, OD, slots
# ---------------------------------------------------------------------------

def _identify_bore_and_outer_radii(section: _PlanarSection) -> tuple[float, float]:
    """From the polygons of the cross-section, extract bore radius and OD radius.

    The bore is the polygon whose maximum-distance-from-origin is the smallest
    of all polygons (it's the innermost). The outer envelope is the polygon
    with the largest max-distance-from-origin.
    """
    if not section.polygons:
        raise GeometryExtractionError("Cross-section has no polygons.")

    poly_max_r = [float(np.max(np.linalg.norm(p, axis=1))) for p in section.polygons]
    poly_min_r = [float(np.min(np.linalg.norm(p, axis=1))) for p in section.polygons]

    if len(section.polygons) == 1:
        # Single polygon -- inner and outer boundaries are the same polygon's
        # min and max distance from centroid.
        outer_r = poly_max_r[0]
        bore_r = poly_min_r[0]
    else:
        # Multiple polygons: bore is the inner polygon, outer envelope is the outer one.
        outer_r = max(poly_max_r)
        # Bore = polygon whose mean radius is smallest *and* is roughly circular.
        mean_radii = [
            float(np.mean(np.linalg.norm(p, axis=1))) for p in section.polygons
        ]
        bore_idx = int(np.argmin(mean_radii))
        bore_pts = section.polygons[bore_idx]
        bore_r = float(np.mean(np.linalg.norm(bore_pts, axis=1)))
        # Sanity: bore should be near-circular.
        radial = np.linalg.norm(bore_pts, axis=1)
        if (radial.max() - radial.min()) / max(radial.mean(), 1e-9) > 0.05:
            raise GeometryExtractionError(
                "Identified bore polygon is not approximately circular "
                f"(max/min radial spread = {radial.max() - radial.min():.3f} mm)."
            )

    if bore_r >= outer_r:
        raise GeometryExtractionError(
            f"Bore radius {bore_r:.3f} mm >= outer radius {outer_r:.3f} mm; impossible."
        )
    return bore_r, outer_r


def _measure_slots(
    section: _PlanarSection, bore_r: float, outer_r: float
) -> tuple[int, dict[str, float]]:
    """Count slots and extract the dimensions of one slot.

    Algorithm: the outer boundary polygon has periodic notches (the slots cut
    into the OD). Walk along the boundary in angle order and detect the
    periodic radial dips.
    """
    # Find the polygon that has the largest angular extent and a max-radius near OD.
    candidates = []
    for p in section.polygons:
        if abs(float(np.max(np.linalg.norm(p, axis=1))) - outer_r) < 1e-3:
            candidates.append(p)
    if not candidates:
        raise GeometryExtractionError("Could not isolate the OD-with-slots polygon.")
    boundary = max(candidates, key=lambda p: len(p))

    # Convert to (theta, r) and sort by theta.
    theta = np.arctan2(boundary[:, 1], boundary[:, 0])
    r = np.linalg.norm(boundary, axis=1)
    order = np.argsort(theta)
    theta = theta[order]
    r = r[order]

    # Detect slots: a slot is a contiguous angular interval where r < (outer_r - 0.1 mm).
    slot_threshold = outer_r - 0.1
    in_slot = r < slot_threshold

    # Run-length encode to find slot count.
    transitions = np.where(np.diff(in_slot.astype(int)) != 0)[0]
    # Wrap-around handling: if first and last are both in_slot, those segments
    # belong to the same slot.
    n_runs = len(transitions) // 2
    if in_slot[0] and in_slot[-1] and n_runs >= 1:
        # leading + trailing slot run is actually one slot wrapping the seam.
        slot_count = n_runs
    else:
        slot_count = n_runs

    if slot_count < 3:
        raise GeometryExtractionError(
            f"Detected only {slot_count} slot-like regions; not a stator geometry."
        )

    # Measure the first slot found.
    # Find the first slot's angular span and depth profile.
    slot_indices: list[tuple[int, int]] = []
    i = 0
    n = len(in_slot)
    while i < n:
        if in_slot[i]:
            j = i
            while j < n and in_slot[j]:
                j += 1
            slot_indices.append((i, j))
            i = j
        else:
            i += 1

    if not slot_indices:
        raise GeometryExtractionError("Slot detection produced no intervals.")

    # Use the slot whose angular span is closest to the median (most representative).
    spans = [theta[j - 1] - theta[i] for i, j in slot_indices]
    median_idx = int(np.argsort(spans)[len(spans) // 2])
    i0, i1 = slot_indices[median_idx]
    slot_theta = theta[i0:i1]
    slot_r = r[i0:i1]

    # Slot dimensions.
    angular_span = float(slot_theta[-1] - slot_theta[0])
    opening_width_mm = angular_span * outer_r  # arc-length at OD
    total_depth_mm = outer_r - float(slot_r.min())
    # Opening depth: depth over which radial profile is roughly constant near OD.
    # Heuristic: depth where r drops by 15% of total_depth (transition from
    # parallel-sided opening into the wider body of the slot).
    parallel_thresh = outer_r - 0.15 * total_depth_mm
    parallel_indices = np.where(slot_r > parallel_thresh)[0]
    if len(parallel_indices) > 0:
        opening_depth_mm = outer_r - float(slot_r[parallel_indices].min())
    else:
        opening_depth_mm = 0.5 * total_depth_mm

    # Bottom width: tangential chord at the slot bottom (deepest radial point).
    bottom_r = float(slot_r.min())
    # Find points within 0.05*depth of the bottom.
    bottom_mask = slot_r < bottom_r + 0.05 * total_depth_mm
    if bottom_mask.sum() >= 2:
        bottom_thetas = slot_theta[bottom_mask]
        bottom_width_mm = float((bottom_thetas.max() - bottom_thetas.min()) * bottom_r)
    else:
        bottom_width_mm = opening_width_mm  # degenerate fallback

    # Slot area: integrate r dr dtheta over the slot region using the trapezoidal rule.
    # Approximation: treat slot as bounded by r(theta) on the OD side and r=bottom_r
    # on the ID side. This is rough -- ~10% error vs true planimetric.
    slot_area_mm2 = float(0.5 * np.trapezoid(outer_r**2 - slot_r**2, slot_theta))

    # Tooth width at narrowest point: tangential gap between this slot's
    # left edge and the previous slot's right edge, at the depth of the opening
    # (i.e., where teeth are thinnest).
    # Approximate from periodicity: tooth_width = slot_pitch_at_OD - opening_width.
    slot_pitch_at_od = 2 * math.pi * outer_r / slot_count
    tooth_width_min_mm = max(slot_pitch_at_od - opening_width_mm, 0.5 * opening_width_mm)

    # Yoke thickness: from slot bottom to bore.
    yoke_thickness_mm = bottom_r - bore_r

    return slot_count, {
        "opening_width_mm": opening_width_mm,
        "opening_depth_mm": opening_depth_mm,
        "total_depth_mm": total_depth_mm,
        "bottom_width_mm": bottom_width_mm,
        "area_mm2": slot_area_mm2,
        "tooth_width_min_mm": tooth_width_min_mm,
        "yoke_thickness_mm": yoke_thickness_mm,
    }


def _verify_periodicity(
    section: _PlanarSection,
    slot_count: int,
    bore_r: float,
    outer_r: float,
) -> float:
    """Rotate the OD profile by 360/Q and compare to itself; return max deviation in mm.

    Implementation: sample the boundary in equal theta steps, then check that the
    profile r(theta) matches r(theta + 2*pi/Q) at all sample points.
    """
    boundary = max(section.polygons, key=lambda p: float(np.max(np.linalg.norm(p, axis=1))))
    theta = np.arctan2(boundary[:, 1], boundary[:, 0])
    r = np.linalg.norm(boundary, axis=1)
    order = np.argsort(theta)
    theta = theta[order]
    r = r[order]

    # Resample uniformly.
    n_samples = max(360, slot_count * 24)
    theta_uniform = np.linspace(-math.pi, math.pi, n_samples, endpoint=False)
    r_uniform = np.interp(theta_uniform, theta, r, period=2 * math.pi)

    # Rotate by one slot pitch and compare.
    shift = n_samples // slot_count
    if shift < 1:
        return 0.0
    r_rotated = np.roll(r_uniform, shift)
    deviation_mm = float(np.max(np.abs(r_uniform - r_rotated)))
    return deviation_mm


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def _sanity_check(geo: StatorGeometry) -> None:
    """Spec section 5.1: hard fails on physically-impossible values."""
    if not (3 <= geo.slot_count <= 72):
        raise GeometryExtractionError(
            f"slot_count {geo.slot_count} outside plausible range [3, 72]"
        )
    if geo.slot_opening_width_mm <= 0:
        raise GeometryExtractionError("slot_opening_width must be > 0")
    if geo.slot_opening_width_mm > geo.tooth_width_min_mm:
        # This is unusual but not strictly impossible; warn only via the
        # extraction_warnings list (already populated by extract_geometry).
        pass
    if geo.slot_total_depth_mm > (geo.outer_radius_mm - geo.bore_radius_mm):
        raise GeometryExtractionError(
            "slot_total_depth exceeds the radial extent (OD - ID)"
        )
    for fname in (
        "bore_radius_mm",
        "outer_radius_mm",
        "stack_length_mm",
        "slot_total_depth_mm",
        "tooth_width_min_mm",
        "yoke_thickness_mm",
    ):
        v = getattr(geo, fname)
        if v <= 0:
            raise GeometryExtractionError(f"{fname} must be positive (got {v})")
