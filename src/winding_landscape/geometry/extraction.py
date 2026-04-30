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
# Periodicity tolerance is interpreted as a fraction of the slot depth at runtime
# rather than as a fixed mm value. Real CAD tessellation can produce sub-mm
# residuals on perfectly periodic parts; what we care about is whether the
# residual is small relative to feature size.
_PERIODICITY_WARN_FRAC = 0.10    # 10% of slot depth
_PERIODICITY_FATAL_FRAC = 0.50   # 50% of slot depth

# Periodicity check resamples the boundary onto a uniform theta grid so that
# np.roll by one slot pitch can compare the profile to itself. Density is a
# constant linear count per mm of OD circumference -- a 10 mm part and a
# 200 mm part both end up with the same arc resolution (so a small slot
# feature spans roughly the same number of samples either way).
_PERIODICITY_DENSITY_PER_MM = 10.0
_PERIODICITY_MIN_SAMPLES_PER_SLOT = 36   # floor for tiny stators


def periodicity_samples_per_slot(slot_count: int, outer_r: float) -> int:
    """Samples per slot for the uniform periodicity grid.

    The grid must be a multiple of ``slot_count`` so that ``np.roll`` rotates
    by exactly one slot pitch. We pick the smallest multiple that gives at
    least ``_PERIODICITY_DENSITY_PER_MM * circumference`` total samples, with
    a floor at ``_PERIODICITY_MIN_SAMPLES_PER_SLOT`` for very small parts.

    Exposed (no leading underscore) so visualization tools can mirror the
    extraction's sampling exactly.
    """
    circumference_mm = 2.0 * math.pi * outer_r
    desired_total = circumference_mm * _PERIODICITY_DENSITY_PER_MM
    return max(_PERIODICITY_MIN_SAMPLES_PER_SLOT, math.ceil(desired_total / slot_count))


def radial_min_envelope(
    theta: np.ndarray,
    r: np.ndarray,
    n_uniform: int,
) -> np.ndarray:
    """Single-valued radial profile r(θ) on a uniform angular grid.

    Returns ``r_uniform`` aligned to ``np.linspace(-π, π, n_uniform, endpoint=False)``.

    Algorithm: dedupe-then-interp. Sort the boundary points by θ; collapse any
    consecutive group of points sharing the same θ (within float ULPs) to a
    single point at (θ, min r). Then linearly interpolate onto the uniform
    grid with ``period=2π``.

    Why this rather than plain ``np.interp``: stator slot side walls are
    typically radial in CAD, so many boundary points share an exact θ with r
    values ranging from the slot bottom up to the OD corner. ``np.interp`` is
    undefined on multi-valued inputs and produces wildly oscillating r at
    those angles. Collapsing each cluster to its smallest r (the closest
    reach to the axis at that angle) preserves the slot-bottom envelope
    that defines the airgap-facing profile.

    Why this rather than min-per-bin: the synthetic test case has 7200
    angular points that don't divide evenly into the periodicity grid, so
    different bins pick up different counts of chamfer samples (2 vs 3) and
    the per-bin min jitters across slots even though the underlying signal
    is exactly periodic. Interp between collapsed unique-θ pairs is grid-
    invariant and avoids that artifact.
    """
    order = np.argsort(theta)
    theta_s = theta[order]
    r_s = r[order]

    eps = 1e-9  # rad; tight enough to only collapse exact radial walls
    is_first = np.empty(len(theta_s), dtype=bool)
    is_first[0] = True
    is_first[1:] = np.diff(theta_s) > eps

    cluster_id = np.cumsum(is_first) - 1
    n_clusters = int(cluster_id[-1]) + 1
    theta_unique = theta_s[is_first]
    r_min = np.full(n_clusters, np.inf)
    np.minimum.at(r_min, cluster_id, r_s)

    theta_uniform = np.linspace(-np.pi, np.pi, n_uniform, endpoint=False)
    return np.interp(theta_uniform, theta_unique, r_min, period=2.0 * np.pi)


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
    axis, axis_origin, axis_radii = _identify_rotation_axis(solid)
    stack_length, axial_min, axial_max = _stack_length_along_axis(solid, axis, axis_origin)

    # Bore and OD come from the consensus-bucket cylinder radii — robust against
    # OCCT's section-of-Face edge-dropping bug, which has been observed to lose
    # the bore-side wire of an annular section on certain BREPs. The radii in
    # the winner bucket are by construction concentric with the rotation axis.
    bore_r = float(min(axis_radii))
    outer_r = float(max(axis_radii))
    logger.debug("Bore radius=%.3f mm, OD=%.3f mm (from cylinder consensus)", bore_r, outer_r)

    # Slice at the true axial midpoint of the part, not at axis_origin + 0.5*L.
    # axis_origin is the closest point on the axis to the world origin, which
    # is unrelated to where the part sits along its own axis.
    midstack_offset = 0.5 * (axial_min + axial_max)
    logger.debug("Slicing at axial offset %.3f mm to obtain 2D cross-section", midstack_offset)
    section_polygon = _slice_at(solid, axis, axis_origin, midstack_offset)

    logger.debug("Counting slots and measuring slot geometry")
    slot_count, slot_features = _measure_slots(section_polygon, bore_r, outer_r)

    logger.debug("Verifying slot periodicity")
    periodicity_dev = _verify_periodicity(section_polygon, slot_count, bore_r, outer_r)

    slot_depth = slot_features["total_depth_mm"]
    warn_tol = max(0.5, _PERIODICITY_WARN_FRAC * slot_depth)   # at least 0.5 mm
    fatal_tol = max(2.0, _PERIODICITY_FATAL_FRAC * slot_depth) # at least 2.0 mm

    warnings: list[str] = []
    if periodicity_dev > fatal_tol:
        raise GeometryExtractionError(
            f"Slot periodicity deviation {periodicity_dev:.3f} mm exceeds "
            f"fatal tolerance {fatal_tol:.3f} mm "
            f"({100 * _PERIODICITY_FATAL_FRAC:.0f}% of slot depth {slot_depth:.2f} mm); "
            "not a periodic stator?"
        )
    if periodicity_dev > warn_tol:
        msg = (
            f"Periodicity deviation {periodicity_dev:.3f} mm > warn tolerance "
            f"{warn_tol:.3f} mm ({100 * _PERIODICITY_WARN_FRAC:.0f}% of slot depth)"
        )
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

def _identify_rotation_axis(solid) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Find the rotation axis by consensus voting across cylindrical faces.

    A radial-flux stator is a bundle of nested cylindrical features (bore, OD,
    slot tips, slot bottoms) all sharing one axis. Auxiliary features
    (mounting holes, fillet patches) sit on other axes but always in the
    minority. We bucket cylindrical faces by their axis line and return the
    bucket with the most members.

    Mechanical analogy: imagine looking at the stator from many angles -- the
    "true" axis is the one along which the most surface features line up. A
    democracy of face normals.

    Returns
    -------
    axis_dir : ndarray (3,)  unit vector along the rotation axis
    axis_origin : ndarray (3,)  the closest point on the axis line to the world origin
    radii : list[float]  every cylindrical face radius concentric with the axis,
        in the order they were discovered. Downstream code uses min(radii) as the
        bore radius and max(radii) as the OD -- bypasses the section operation,
        which has been observed to silently drop bore-side edges in OCCT.
    """
    try:
        import build123d as bd
    except ImportError as exc:
        raise GeometryExtractionError("build123d unavailable.") from exc

    cylinders: list[tuple[np.ndarray, np.ndarray, float]] = []
    for face in solid.faces():
        # Use enum comparison rather than string parsing -- in build123d 0.10+
        # repr(face.geom_type) is "<GeomType.CYLINDER>", which doesn't equal
        # "CYLINDER" under any string transformation.
        if face.geom_type != bd.GeomType.CYLINDER:
            continue
        axis_dir, origin, radius = _cylinder_axis_and_radius(face)
        if radius is None or radius < 1e-6:
            continue
        cylinders.append((axis_dir, origin, radius))

    if not cylinders:
        raise GeometryExtractionError(
            "No cylindrical faces found in the BREP. Cannot identify the "
            "rotation axis."
        )

    # Bucket by axis line. Two cylinders share an axis line if:
    #   (a) their direction vectors are parallel or anti-parallel, AND
    #   (b) the perpendicular distance between their axis lines is < 0.1 mm.
    # We canonicalize the axis direction (flip so the largest-magnitude
    # component is positive) so anti-parallel cylinders bucket together.
    AXIS_DIST_TOL = 0.1   # mm
    DIR_DOT_TOL = 0.999   # cos(~2.5 deg)

    def canonicalize(v: np.ndarray) -> np.ndarray:
        idx = int(np.argmax(np.abs(v)))
        return v if v[idx] >= 0 else -v

    def axis_line_distance(d1, p1, d2, p2) -> float:
        """Minimum distance between two infinite lines (d, p)."""
        cross = np.cross(d1, d2)
        n = np.linalg.norm(cross)
        if n < 1e-9:
            # Lines are parallel; distance = perpendicular component of (p2 - p1).
            diff = p2 - p1
            proj = diff - np.dot(diff, d1) * d1
            return float(np.linalg.norm(proj))
        return float(abs(np.dot(p2 - p1, cross)) / n)

    buckets: list[dict] = []
    for axis_dir, origin, radius in cylinders:
        d_canon = canonicalize(axis_dir)
        placed = False
        for bucket in buckets:
            if abs(np.dot(d_canon, bucket["dir"])) < DIR_DOT_TOL:
                continue
            if axis_line_distance(d_canon, origin, bucket["dir"], bucket["origin"]) > AXIS_DIST_TOL:
                continue
            bucket["count"] += 1
            bucket["radii"].append(radius)
            placed = True
            break
        if not placed:
            buckets.append({
                "dir": d_canon,
                "origin": origin,
                "count": 1,
                "radii": [radius],
            })

    # Sort by member count, descending. Ties broken by radius spread (the part
    # axis should have a wide spread of radii: bore, OD, slot features).
    buckets.sort(key=lambda b: (b["count"], max(b["radii"]) - min(b["radii"])), reverse=True)
    winner = buckets[0]

    if winner["count"] < 2:
        raise GeometryExtractionError(
            f"No axis has 2+ concentric cylindrical faces. Top bucket: "
            f"{winner['count']} cylinder, dir={winner['dir']}. Is this really "
            "a radial-flux stator BREP?"
        )

    # Snap the axis origin to the closest point on the line to the world origin.
    d = winner["dir"]
    p = winner["origin"]
    t = -np.dot(p, d)
    canonical_origin = p + t * d

    logger.debug(
        "Rotation axis: dir=%s, origin=%s, %d concentric cylinders, "
        "radii range %.2f .. %.2f mm",
        d.round(4), canonical_origin.round(3), winner["count"],
        min(winner["radii"]), max(winner["radii"]),
    )
    return d, canonical_origin, list(winner["radii"])


def _cylinder_axis_and_radius(face) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Pull (axis_direction_unit, axis_origin, radius) out of a build123d
    cylindrical face. Targets build123d 0.7+ API (radius is an attribute,
    axis_of_rotation has .position and .direction Vectors).

    Mechanical analogy: this is the equivalent of clamping a part in a fixture
    and measuring (1) which way the cylinder runs, (2) where its centerline is,
    and (3) what diameter it has -- everything you'd need to put it on a
    rotational fixture.
    """
    # Strategy 1: high-level attributes (build123d 0.7+).
    try:
        radius = float(face.radius)
        aor = face.axis_of_rotation
        origin = np.array([aor.position.X, aor.position.Y, aor.position.Z])
        axis_dir = np.array([aor.direction.X, aor.direction.Y, aor.direction.Z])
        n = np.linalg.norm(axis_dir)
        if n < 1e-9 or radius < 1e-9:
            return np.array([0.0, 0.0, 1.0]), np.zeros(3), None
        return axis_dir / n, origin, radius
    except (AttributeError, ValueError, TypeError):
        pass

    # Strategy 2: BRepAdaptor on the wrapped TopoDS_Face. Used if a future
    # build123d release moves these attributes again.
    try:
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        adaptor = BRepAdaptor_Surface(face.wrapped)
        cyl = adaptor.Cylinder()
        ax = cyl.Axis()
        loc = ax.Location()
        direction = ax.Direction()
        origin = np.array([loc.X(), loc.Y(), loc.Z()])
        axis_dir = np.array([direction.X(), direction.Y(), direction.Z()])
        n = np.linalg.norm(axis_dir)
        if n < 1e-9:
            return np.array([0.0, 0.0, 1.0]), np.zeros(3), None
        return axis_dir / n, origin, float(cyl.Radius())
    except Exception:  # noqa: BLE001
        return np.array([0.0, 0.0, 1.0]), np.zeros(3), None


# ---------------------------------------------------------------------------
# Stack length
# ---------------------------------------------------------------------------

def _stack_length_along_axis(
    solid, axis: np.ndarray, origin: np.ndarray
) -> tuple[float, float, float]:
    """Project all vertices onto the rotation axis.

    Returns
    -------
    length : float  the axial extent of the solid (max - min projection).
    axial_min : float  smallest projection value, in axis-coordinates relative to ``origin``.
    axial_max : float  largest projection value, in axis-coordinates relative to ``origin``.

    The min/max are needed because ``origin`` is the closest point on the axis
    to the world origin, not the geometric center of the part. Slicing at
    ``origin + 0.5 * length * axis`` would put the cut at one of the axial end
    faces if the part is centered on the world origin (a common Solidworks
    default). Use ``0.5 * (axial_min + axial_max)`` for true mid-stack.
    """
    projections: list[float] = []
    for vertex in solid.vertices():
        p = np.array([vertex.X, vertex.Y, vertex.Z])
        projections.append(float(np.dot(p - origin, axis)))
    if not projections:
        raise GeometryExtractionError("Solid has no vertices?!")
    axial_min = min(projections)
    axial_max = max(projections)
    length = axial_max - axial_min
    if length <= 0:
        raise GeometryExtractionError(f"Computed stack length {length} mm is non-positive.")
    return length, axial_min, axial_max


# ---------------------------------------------------------------------------
# Cross-sectioning
# ---------------------------------------------------------------------------
def _slice_at(solid, axis: np.ndarray, origin: np.ndarray, axial_offset: float):
    """Slice the solid with a plane perpendicular to the axis at given offset.

    Returns the cross-section as a :class:`_PlanarSection` with up to two
    polygons: the bore-side (inner) boundary and the OD-side (outer) boundary.

    Algorithm:
      1. Build a slicing plane perpendicular to ``axis`` at ``origin + offset*axis``.
         Use a world-aligned x_dir to avoid OpenCascade clipping artifacts that
         have been observed when the plane's local frame is oblique.
      2. Boolean-intersect the solid with the plane to obtain a planar section.
      3. Iterate over ``section.edges()`` directly (don't trust ``section.wires()``
         -- in some build123d builds the wire grouping silently drops edges).
      4. Sample each edge to a 2D point list, tracking each edge's max radius.
      5. Classify edges as bore-side or OD-side by max-radius gap. Bore arcs
         and bore-edge chamfers cluster at small max_r; slot bottoms, slot
         walls, and OD arcs all reach to (or near) the OD radius.
      6. Concatenate each class's points into a single polygon for downstream
         processing.
    """
    try:
        import build123d as bd
    except ImportError as exc:
        raise GeometryExtractionError("build123d unavailable.") from exc

    # ----- Build the slicing plane -----
    plane_origin = origin + axial_offset * axis
    # Pick a world axis for the plane's local x_dir. We've observed cases where
    # an oblique x_dir causes OpenCascade to clip section edges; a world-aligned
    # frame avoids that. Choose the world axis least parallel to the rotation axis.
    candidate_x_dirs = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    best_x = max(
        candidate_x_dirs,
        key=lambda d: float(np.linalg.norm(np.cross(axis, d))),
    )
    # Project out any axis component to ensure exact perpendicularity.
    best_x = best_x - np.dot(best_x, axis) * axis
    best_x /= np.linalg.norm(best_x)
    u = best_x
    v = np.cross(axis, u)
    v /= np.linalg.norm(v)

    # Use OCCT's BRepAlgoAPI_Section directly rather than build123d's high-level
    # `solid & plane` operator. The high-level operator returns a Face, and
    # OCCT's Face-classification step has been observed to drop edges that
    # belong to "inner hole" wires (notably bore arcs in annular sections) when
    # the plane's bounded extent doesn't quite cover them. BRepAlgoAPI_Section
    # produces a Compound of free edges with no boundary classification, so
    # nothing gets dropped.
    try:
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
        from OCP.gp import gp_Pln, gp_Pnt, gp_Dir, gp_Ax3
    except ImportError as exc:
        raise GeometryExtractionError(
            "OCP (OpenCascade Python bindings) unavailable; required for robust "
            "cross-sectioning."
        ) from exc

    # Build a gp_Pln (infinite plane) directly. The local frame is the same as
    # what we'd give build123d, but as a primitive there's no Face wrapping.
    ax_system = gp_Ax3(
        gp_Pnt(*plane_origin.tolist()),
        gp_Dir(*axis.tolist()),
        gp_Dir(*u.tolist()),
    )
    gp_plane = gp_Pln(ax_system)

    try:
        sectioner = BRepAlgoAPI_Section(solid.wrapped, gp_plane, False)
        sectioner.ComputePCurveOn1(True)
        sectioner.Approximation(False)
        sectioner.Build()
        if not sectioner.IsDone():
            raise GeometryExtractionError("BRepAlgoAPI_Section did not complete.")
        section_shape = sectioner.Shape()
    except Exception as exc:  # noqa: BLE001
        raise GeometryExtractionError(f"Cross-section operation failed: {exc}") from exc

    # The result is a TopoDS_Compound of TopoDS_Edge. Walk it and wrap each
    # edge as a build123d Edge so our existing _sample_edge_robust works.
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_EDGE

    edges: list = []
    explorer = TopExp_Explorer(section_shape, TopAbs_EDGE)
    while explorer.More():
        topods_edge = explorer.Current()
        # Wrap as a build123d Edge for API uniformity.
        bd_edge = bd.Edge(topods_edge)
        edges.append(bd_edge)
        explorer.Next()
    if not edges:
        raise GeometryExtractionError("Cross-section produced no edges.")

    # ----- Sample each edge, recording the points and the edge's max radius -----
    sample_params = [i / 24 for i in range(25)]
    edge_points_and_max: list[tuple[list[tuple[float, float]], float]] = []

    for edge in edges:
        try:
            samples = _sample_edge_robust(edge, sample_params)
        except Exception:  # noqa: BLE001
            samples = []
        if not samples:
            continue
        pts: list[tuple[float, float]] = []
        rs: list[float] = []
        for s in samples:
            try:
                p = np.array([s.X, s.Y, s.Z]) - plane_origin
                x2 = float(np.dot(p, u))
                y2 = float(np.dot(p, v))
                pts.append((x2, y2))
                rs.append(float(np.hypot(x2, y2)))
            except Exception:  # noqa: BLE001
                continue
        if pts:
            edge_points_and_max.append((pts, max(rs)))

    if not edge_points_and_max:
        raise GeometryExtractionError("Cross-section produced no sample points.")

    # ----- Classify edges by their max-radius -----
    # An annular stator section has two distinct families of features:
    #   - Bore-side: bore arcs + chamfer transitions, clustered at small max_r.
    #   - Outer-side: slot bottom arcs, slot side-walls, OD arcs. All of these
    #     have at least one point at or near the OD, so their max_r is high.
    # We separate them with a relative threshold sized to bundle the bore arcs
    # with their chamfer transitions (typically ~1mm apart in radius) but not
    # incorporate slot bottoms (typically further out).
    edge_max_rs = [m for _, m in edge_points_and_max]
    smallest_max_r = min(edge_max_rs)
    largest_max_r = max(edge_max_rs)
    bore_tol = max(1.5, 0.10 * (largest_max_r - smallest_max_r))
    bore_threshold = smallest_max_r + bore_tol

    inner_pts: list[tuple[float, float]] = []
    outer_pts: list[tuple[float, float]] = []
    for pts, max_r in edge_points_and_max:
        if max_r <= bore_threshold:
            inner_pts.extend(pts)
        else:
            outer_pts.extend(pts)

    polygons: list[np.ndarray] = []
    if len(inner_pts) >= 3:
        polygons.append(np.array(inner_pts))
    if len(outer_pts) >= 3:
        polygons.append(np.array(outer_pts))
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
    pts: list[tuple[float, float]] = []
    sample_params = [i / n_samples_per_edge for i in range(n_samples_per_edge + 1)]

    for edge in wire.edges():
        for s in _sample_edge_robust(edge, sample_params):
            try:
                p = np.array([s.X, s.Y, s.Z]) - plane_origin
                pts.append((float(np.dot(p, u)), float(np.dot(p, v))))
            except Exception:  # noqa: BLE001
                continue

    return np.asarray(pts, dtype=np.float64)


def _sample_edge_robust(edge, sample_params: list[float]) -> list:
    """Sample an edge at the given normalized parameters [0, 1].

    Tries multiple build123d API shapes in turn so one bad edge can't poison
    the whole wire's sampling.
    """
    # Strategy 1: edge.positions(params) -- bulk version (build123d 0.6+).
    try:
        result = edge.positions(sample_params)
        if result and len(result) > 0:
            return list(result)
    except Exception:  # noqa: BLE001
        pass

    # Strategy 2: edge.position_at(p) called per-parameter (build123d 0.7+).
    try:
        return [edge.position_at(p) for p in sample_params]
    except Exception:  # noqa: BLE001
        pass

    # Strategy 3: edge @ p operator overload (build123d sugar).
    try:
        return [edge @ p for p in sample_params]
    except Exception:  # noqa: BLE001
        pass

    # Strategy 4: walk vertices (only the endpoints, but better than nothing).
    try:
        verts = list(edge.vertices())
        if verts:
            return [v.center() if callable(getattr(v, "center", None)) else v for v in verts]
    except Exception:  # noqa: BLE001
        pass

    # Last resort: this edge contributes nothing, but the rest of the wire continues.
    logger.debug("Could not sample edge %r with any known API", edge)
    return []


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

    Strategy: pool all points across all polygons in the section. The OD radius
    is the maximum distance any point reaches from the axis. The bore radius is
    the minimum distance, with the caveat that real stators sometimes have
    slot tooth-tips that extend inward beyond the bore proper (the bore as
    drawn in the part is at one radius, but slot openings cut through at a
    slightly different radius). We use the *minimum* of all inner-side points
    since that's what the rotor's airgap will see.

    Mechanical analogy: this is the equivalent of running a dial indicator
    around the inside of the bore -- the smallest reading is what matters
    for assembly, regardless of whether the surface there is technically
    "bore" or "tooth-tip".
    """
    if not section.polygons:
        raise GeometryExtractionError("Cross-section has no polygons.")

    all_r: list[float] = []
    for p in section.polygons:
        radii = np.linalg.norm(p, axis=1)
        all_r.extend(radii.tolist())

    if not all_r:
        raise GeometryExtractionError("Cross-section polygons contain no points.")

    outer_r = float(max(all_r))
    bore_r = float(min(all_r))

    if bore_r >= outer_r:
        raise GeometryExtractionError(
            f"Bore radius {bore_r:.3f} mm >= outer radius {outer_r:.3f} mm; impossible."
        )
    if bore_r < 1e-3:
        raise GeometryExtractionError(
            f"Bore radius {bore_r:.3f} mm too close to axis; not a stator geometry."
        )
    return bore_r, outer_r


_SLOT_DETECTION_BINS = 1440          # 0.25 degree angular resolution
_SLOT_DETECTION_DEPTH_FRACS = (0.30, 0.40, 0.50, 0.60, 0.70)
_SLOT_BOTTOM_PERCENTILE = 0.5        # robust slot-bottom estimate (ignores the deepest 0.5%)


def _count_slots_at_depth(
    bin_idx: np.ndarray,
    r: np.ndarray,
    r_test: float,
    n_bins: int,
) -> tuple[int, list[tuple[int, int]], np.ndarray]:
    """Count angular regions where the boundary reaches below ``r_test``.

    A "deep region" is a contiguous run of angular bins that contain at least
    one boundary point with r < r_test. Teeth produce no such bins (no
    section material between OD and the rotor at tooth angles); slots do.
    Returns ``(n_slots, runs, has_deep_closed)`` where ``runs`` are the
    (start, end) bin indices and ``has_deep_closed`` is the morphologically
    bridged boolean array.
    """
    deep_pts = r < r_test
    has_deep = np.zeros(n_bins, dtype=bool)
    if deep_pts.any():
        has_deep[bin_idx[deep_pts]] = True
    # Bridge single-bin tessellation gaps so a slot whose section happens to
    # miss one bin doesn't fragment into two.
    closed = has_deep | np.roll(has_deep, 1) | np.roll(has_deep, -1)
    runs = _find_runs_circular(closed)
    return len(runs), runs, closed


def _detect_slot_count_and_regions(
    theta: np.ndarray,
    r: np.ndarray,
    outer_r: float,
) -> tuple[int, list[tuple[int, int]], np.ndarray, int]:
    """Robust slot-count detection by counting deep regions at multiple depths.

    The previous algorithm classified bins by *median r* with a threshold near
    the OD. That works on clean stators but breaks when slots have internal
    sub-features (lock cuts, fillets, draft) -- the median can drift across
    sub-slot boundaries and fragment one physical slot into many. The deep-
    region count is more robust: at any radius significantly below OD, only
    real slot interiors are populated, regardless of what's above.

    Voting across several depths (30%, 40%, 50%, 60%, 70% of the way from OD
    to the slot bottom) makes the count immune to a single bad depth: even
    if one frac picks up an artifact, the mode is the true slot count.

    Returns ``(slot_count, runs, bin_idx, n_bins)`` where ``runs`` are the
    deep-region (start, end) bin indices at the depth that produced the
    consensus count, and ``bin_idx`` maps each input point to its bin.
    """
    from collections import Counter

    n_bins = _SLOT_DETECTION_BINS
    bin_edges = np.linspace(-math.pi, math.pi, n_bins + 1)
    bin_idx = np.clip(np.digitize(theta, bin_edges) - 1, 0, n_bins - 1)

    r_min = float(np.percentile(r, _SLOT_BOTTOM_PERCENTILE))
    if r_min >= outer_r - 0.05:
        raise GeometryExtractionError(
            "No radial dips detected in outer boundary; not a slotted stator?"
        )

    candidates: list[tuple[float, int, list, np.ndarray]] = []
    for frac in _SLOT_DETECTION_DEPTH_FRACS:
        r_test = outer_r - frac * (outer_r - r_min)
        n, runs, _closed = _count_slots_at_depth(bin_idx, r, r_test, n_bins)
        candidates.append((frac, n, runs, _closed))

    counts = [c[1] for c in candidates]
    consensus, votes = Counter(counts).most_common(1)[0]

    if consensus < 3:
        raise GeometryExtractionError(
            f"Slot count consensus is {consensus} across test depths {counts}; "
            "not a slotted stator?"
        )
    if votes < 3:
        logger.warning(
            "Slot-count voting weak: counts=%s, consensus=%d (%d/5 votes). "
            "Geometry may have unusual sub-slot features.",
            counts, consensus, votes,
        )

    # Pick the candidate at the depth nearest 50% that produced the consensus.
    chosen = min(
        (c for c in candidates if c[1] == consensus),
        key=lambda c: abs(c[0] - 0.5),
    )
    _frac, _n, runs, _closed = chosen

    return consensus, runs, bin_idx, n_bins


def _measure_slots(
    section: _PlanarSection, bore_r: float, outer_r: float
) -> tuple[int, dict[str, float]]:
    """Count slots and extract dimensions of one slot.

    Slot count comes from :func:`_detect_slot_count_and_regions` (multi-depth
    over-under voting). Slot dimensions come from the deep-region of the
    representative slot at the consensus depth.
    """
    # Pool boundary points from all polygons that reach near the OD.
    # In real STEP exports the outer boundary may be split across multiple wires.
    boundary_points: list[np.ndarray] = []
    for p in section.polygons:
        max_r = float(np.max(np.linalg.norm(p, axis=1)))
        if max_r >= outer_r - 0.5:  # within 0.5 mm of OD = part of the outer boundary
            boundary_points.append(p)
    if not boundary_points:
        raise GeometryExtractionError("Could not isolate the OD-with-slots boundary.")
    boundary = np.vstack(boundary_points)

    # Convert to (theta, r), sort by theta.
    theta = np.arctan2(boundary[:, 1], boundary[:, 0])
    r = np.linalg.norm(boundary, axis=1)
    order = np.argsort(theta)
    theta = theta[order]
    r = r[order]

    # Robust slot count + slot regions.
    slot_count, runs, bin_idx, n_bins = _detect_slot_count_and_regions(theta, r, outer_r)

    # Pick a representative slot by median width in bins.
    bin_width_rad = 2 * math.pi / n_bins
    widths_bins = [_run_width(start, end, n_bins) for start, end in runs]
    median_width = sorted(widths_bins)[len(widths_bins) // 2]
    rep_idx = min(range(len(runs)), key=lambda i: abs(widths_bins[i] - median_width))
    rep_start, rep_end = runs[rep_idx]
    rep_width_bins = widths_bins[rep_idx]

    # Collect actual data points within this slot (inclusive of the slot's
    # angular extent, in original sample points -- not bins).
    slot_mask = _circular_bin_mask(bin_idx, rep_start, rep_end, n_bins)
    slot_r = r[slot_mask]
    slot_theta = theta[slot_mask]
    if len(slot_r) < 3:
        raise GeometryExtractionError("Representative slot has too few points to measure.")

    # ----- Slot dimensions -----
    angular_span = rep_width_bins * bin_width_rad
    body_arc_at_od_mm = float(angular_span * outer_r)  # body arc length, used as fallback

    # Slot opening at the airgap is the angular gap between adjacent OD arcs --
    # i.e. the stretch of theta where there is no material at radius=outer_r.
    # Measuring this directly is more accurate than measuring the slot-body
    # width, since semi-closed slots have tooth-tip overhangs that widen the
    # body well below the actual airgap-side opening.
    od_tol = 0.1  # mm; tessellation noise on OD arc samples is well below this
    od_mask = r >= outer_r - od_tol
    if od_mask.sum() >= slot_count * 4:
        od_theta_sorted = np.sort(theta[od_mask])
        diffs = np.diff(od_theta_sorted)
        wrap_gap = 2.0 * math.pi - (od_theta_sorted[-1] - od_theta_sorted[0])
        all_gaps = np.concatenate([diffs, [wrap_gap]])
        # Slot opening gaps are the slot_count widest gaps in OD coverage.
        slot_gaps = np.sort(all_gaps)[-slot_count:]
        opening_width_mm = float(np.median(slot_gaps) * outer_r)
    else:
        opening_width_mm = body_arc_at_od_mm

    bottom_r = float(slot_r.min())
    total_depth_mm = float(outer_r - bottom_r)

    # Opening depth: depth over which the radial profile is roughly constant
    # near the OD (parallel-sided slot opening before it widens out).
    parallel_thresh = outer_r - 0.15 * total_depth_mm
    parallel_mask = slot_r > parallel_thresh
    opening_depth_mm = float(outer_r - slot_r[parallel_mask].min()) if parallel_mask.any() else 0.5 * total_depth_mm

    # Bottom width: tangential chord at the slot bottom.
    bottom_band = slot_r < bottom_r + 0.05 * total_depth_mm
    if bottom_band.sum() >= 2:
        b_thetas = slot_theta[bottom_band]
        # Handle wrap-around at the seam.
        if b_thetas.max() - b_thetas.min() > math.pi:
            b_thetas = np.where(b_thetas < 0, b_thetas + 2 * math.pi, b_thetas)
        bottom_width_mm = float((b_thetas.max() - b_thetas.min()) * bottom_r)
    else:
        bottom_width_mm = opening_width_mm  # degenerate fallback

    # Slot area: integrate r dr dtheta over the slot region using the trapezoidal rule.
    sort_idx = np.argsort(slot_theta)
    sorted_theta = slot_theta[sort_idx]
    sorted_r = slot_r[sort_idx]
    # Handle wrap-around: if the slot straddles theta = +/-pi, shift it.
    if sorted_theta[-1] - sorted_theta[0] > math.pi:
        sorted_theta = np.where(sorted_theta < 0, sorted_theta + 2 * math.pi, sorted_theta)
        new_order = np.argsort(sorted_theta)
        sorted_theta = sorted_theta[new_order]
        sorted_r = sorted_r[new_order]
    slot_area_mm2 = float(0.5 * np.trapezoid(outer_r ** 2 - sorted_r ** 2, sorted_theta))

    # Tooth width at narrowest point: derived from periodicity.
    slot_pitch_at_od = 2 * math.pi * outer_r / slot_count
    tooth_width_min_mm = max(slot_pitch_at_od - opening_width_mm, 0.5 * opening_width_mm)

    # Yoke thickness: from slot bottom to bore.
    yoke_thickness_mm = max(bottom_r - bore_r, 0.0)

    return slot_count, {
        "opening_width_mm": opening_width_mm,
        "opening_depth_mm": opening_depth_mm,
        "total_depth_mm": total_depth_mm,
        "bottom_width_mm": bottom_width_mm,
        "area_mm2": slot_area_mm2,
        "tooth_width_min_mm": tooth_width_min_mm,
        "yoke_thickness_mm": yoke_thickness_mm,
    }


def _find_runs_circular(mask: np.ndarray) -> list[tuple[int, int]]:
    """Find runs of True in a circular boolean array. Returns [(start, end), ...]
    where end is exclusive, with wrap-around at the seam handled.
    """
    n = len(mask)
    if not mask.any():
        return []
    if mask.all():
        return [(0, n)]
    # Find transitions
    diff = np.diff(mask.astype(np.int8), append=mask[0])
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    # If we start mid-True, prepend a 0
    if mask[0]:
        starts = np.concatenate([[0], starts])
    # Pair them up, handling wrap-around
    runs: list[tuple[int, int]] = []
    if mask[0] and mask[-1] and len(starts) > len(ends):
        # Last run wraps around: starts[-1] connects to ends[0]
        first_end = ends[0]
        for s, e in zip(starts[1:-1], ends[1:], strict=False):
            runs.append((int(s), int(e)))
        runs.append((int(starts[-1]), int(first_end + n)))
    else:
        for s, e in zip(starts, ends, strict=False):
            runs.append((int(s), int(e)))
    return runs


def _run_width(start: int, end: int, n_bins: int) -> int:
    """Width of a circular run, accounting for wrap-around."""
    w = end - start
    if w <= 0:
        w += n_bins
    return w


def _circular_bin_mask(bin_idx: np.ndarray, start: int, end: int, n_bins: int) -> np.ndarray:
    """Mask of points whose bin falls within [start, end), with wrap-around."""
    start = start % n_bins
    end_mod = end % n_bins
    if end > n_bins:
        # Wraps around
        return (bin_idx >= start) | (bin_idx < end_mod)
    if start <= end_mod:
        return (bin_idx >= start) & (bin_idx < end_mod)
    return (bin_idx >= start) | (bin_idx < end_mod)


def _verify_periodicity(
    section: _PlanarSection,
    slot_count: int,
    bore_r: float,
    outer_r: float,
) -> float:
    """Rotate the OD profile by 360/Q and compare to itself; return max deviation in mm.

    Implementation: pool all OD-side boundary points, resample r(theta) onto a
    uniform grid whose size is a multiple of slot_count (so np.roll aligns
    exactly), then compare r(theta) to r(theta + 2*pi/Q).

    Mechanical analogy: this is the equivalent of measuring runout on a
    fixture-mounted part by rotating it one tooth at a time and noting the
    difference at each angular position. A truly periodic part has zero
    residual; a part with a defect or non-periodic feature shows a clear bump.
    """
    # Pool OD-side points (same logic as _measure_slots).
    boundary_points: list[np.ndarray] = []
    for p in section.polygons:
        max_r = float(np.max(np.linalg.norm(p, axis=1)))
        if max_r >= outer_r - 0.5:
            boundary_points.append(p)
    if not boundary_points:
        return 0.0
    boundary = np.vstack(boundary_points)

    theta = np.arctan2(boundary[:, 1], boundary[:, 0])
    r = np.linalg.norm(boundary, axis=1)
    order = np.argsort(theta)
    theta = theta[order]
    r = r[order]

    # Resample on a uniform theta grid whose size is a multiple of slot_count
    # so that np.roll gives exact alignment. Density scales with OD circumference
    # so small features cover roughly the same number of samples regardless of
    # part size.
    samples_per_slot = periodicity_samples_per_slot(slot_count, outer_r)
    n_uniform = slot_count * samples_per_slot
    r_uniform = radial_min_envelope(theta, r, n_uniform)

    # Rotate by exactly one slot pitch and compare.
    r_rotated = np.roll(r_uniform, samples_per_slot)
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
