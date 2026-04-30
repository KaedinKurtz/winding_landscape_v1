"""Inspect what build123d sees in a STEP file."""
import sys
from pathlib import Path
import numpy as np
import build123d as bd

path = Path(sys.argv[1])
result = bd.import_step(str(path))
solids = list(result.solids()) if hasattr(result, "solids") else [result]
solid = max(solids, key=lambda s: s.volume)

bbox = solid.bounding_box()
print(f"Bounding box:")
print(f"  X: {bbox.min.X:8.2f} .. {bbox.max.X:8.2f}  (size {bbox.max.X-bbox.min.X:.2f})")
print(f"  Y: {bbox.min.Y:8.2f} .. {bbox.max.Y:8.2f}  (size {bbox.max.Y-bbox.min.Y:.2f})")
print(f"  Z: {bbox.min.Z:8.2f} .. {bbox.max.Z:8.2f}  (size {bbox.max.Z-bbox.min.Z:.2f})")
centroid = np.array([
    0.5*(bbox.min.X + bbox.max.X),
    0.5*(bbox.min.Y + bbox.max.Y),
    0.5*(bbox.min.Z + bbox.max.Z),
])
print(f"Bounding-box centroid: {centroid}")
print()

# Histogram all geom_types — using the enum directly, not strings
all_faces = list(solid.faces())
from collections import Counter
type_counts = Counter(f.geom_type for f in all_faces)
print(f"Faces by geom_type ({len(all_faces)} total):")
for t, n in type_counts.most_common():
    print(f"  {t!r}: {n}")
print()

# Filter cylinders correctly: compare to the enum member, not a string.
cyl_faces = [f for f in all_faces if f.geom_type == bd.GeomType.CYLINDER]
print(f"Cylindrical faces (correct filter): {len(cyl_faces)}")
print()

if cyl_faces:
    sample = cyl_faces[0]
    print("Attribute snapshot of one cylindrical face:")
    for attr in sorted(dir(sample)):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(sample, attr)
            if callable(val):
                continue
            r = repr(val)
            if len(r) > 100:
                r = r[:100] + "..."
            print(f"  {attr}: {r}")
        except Exception as e:
            print(f"  {attr}: <error: {e!r}>")
    print()

def try_extract(face):
    """Try several known build123d API shapes for (radius, axis_origin, axis_dir)."""
    # Strategy 1: high-level attributes (build123d 0.7+)
    try:
        r = float(face.radius)
        aor = face.axis_of_rotation
        origin = np.array([aor.position.X, aor.position.Y, aor.position.Z])
        d = np.array([aor.direction.X, aor.direction.Y, aor.direction.Z])
        return r, origin, d / np.linalg.norm(d), "axis_of_rotation+radius"
    except Exception as e:
        err1 = repr(e)
    # Strategy 2: BRepAdaptor on the wrapped TopoDS_Face
    try:
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        adaptor = BRepAdaptor_Surface(face.wrapped)
        cyl = adaptor.Cylinder()
        ax = cyl.Axis()
        loc = ax.Location()
        direction = ax.Direction()
        origin = np.array([loc.X(), loc.Y(), loc.Z()])
        d = np.array([direction.X(), direction.Y(), direction.Z()])
        return float(cyl.Radius()), origin, d / np.linalg.norm(d), "BRepAdaptor_Surface"
    except Exception as e:
        err2 = repr(e)
    return None, None, None, f"S1:{err1} | S2:{err2}"

print("Extracting axis + radius for each cylindrical face:")
strategy_used = None
results = []
for i, face in enumerate(cyl_faces):
    r, origin, axis_dir, strat = try_extract(face)
    if r is None:
        if i < 3:
            print(f"  [{i}] FAIL: {strat[:200]}")
        continue
    if strategy_used is None:
        strategy_used = strat
    results.append((r, origin, axis_dir))
    if i < 12:
        rel = centroid - origin
        proj = rel - np.dot(rel, axis_dir) * axis_dir
        dist = float(np.linalg.norm(proj))
        print(f"  [{i}] r={r:7.2f}  origin=({origin[0]:7.2f},{origin[1]:7.2f},{origin[2]:7.2f})  axis=({axis_dir[0]:+.2f},{axis_dir[1]:+.2f},{axis_dir[2]:+.2f})  dist_to_centroid={dist:.3f}")

print(f"\n{len(results)}/{len(cyl_faces)} extracted via: {strategy_used}")

if results:
    radii = [round(r, 1) for r, _, _ in results]
    hist = Counter(radii)
    print(f"\nRadius histogram (mm -> count):")
    for r in sorted(hist):
        print(f"  {r:7.1f}: {hist[r]}")

    # Dominant axis direction (which way the part is oriented)
    axes = [tuple(np.round(a, 2)) for _, _, a in results]
    print(f"\nAxis-direction histogram:")
    for a, n in Counter(axes).most_common(5):
        print(f"  {a}: {n}")