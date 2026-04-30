"""Dump every detail of what build123d's section operation produces for our stator."""
import sys
from pathlib import Path
import numpy as np
import build123d as bd

path = Path(sys.argv[1])
result = bd.import_step(str(path))
solids = list(result.solids()) if hasattr(result, "solids") else [result]
solid = max(solids, key=lambda s: s.volume)

# Slice at mid-stack. Y is the axis (we already know).
plane = bd.Plane(
    origin=bd.Vector(0, 0, 0),
    x_dir=bd.Vector(1, 0, 0),
    z_dir=bd.Vector(0, 1, 0),
)
section = solid & plane

print(f"section type: {type(section).__name__}")
print(f"section is None: {section is None}")
print()

# What does section contain?
print("Has methods:")
for attr in ("faces", "wires", "edges", "vertices", "solids", "shells", "compounds"):
    if hasattr(section, attr):
        try:
            items = list(getattr(section, attr)())
            print(f"  .{attr}(): {len(items)} item(s)")
        except Exception as e:
            print(f"  .{attr}(): ERROR {e!r}")
print()

# Collect ALL edges (not just from wires) and report each one's geometry.
print("=== All edges in the section, by geom_type ===")
all_edges = list(section.edges())
print(f"Total edges: {len(all_edges)}")
from collections import Counter
type_counts = Counter(e.geom_type for e in all_edges)
for t, n in type_counts.most_common():
    print(f"  {t!r}: {n}")
print()

# For circular edges, dump the radius and the center.
print("=== Circular edges (radius, center, axis) ===")
circles = [e for e in all_edges if e.geom_type == bd.GeomType.CIRCLE]
print(f"{len(circles)} circular edges")
seen_radii = Counter()
for e in circles:
    try:
        r = float(e.radius)
        seen_radii[round(r, 2)] += 1
    except Exception:
        pass
print(f"Radius histogram: {dict(seen_radii)}")
print()

# For each WIRE in the section, report the radial range of its sample points.
print("=== Wires and their radial spans ===")
wires = list(section.wires()) if hasattr(section, "wires") else []
print(f"{len(wires)} wires")
for i, w in enumerate(wires):
    pts_3d = []
    for edge in w.edges():
        try:
            samples = edge.positions([k / 50 for k in range(51)])
            pts_3d.extend(samples)
        except Exception:
            pass
    if not pts_3d:
        print(f"  wire[{i}]: no points sampled")
        continue
    rs = []
    for p in pts_3d:
        # Project onto X-Z plane (since Y is the axis)
        rs.append(float(np.sqrt(p.X**2 + p.Z**2)))
    print(f"  wire[{i}]: {len(pts_3d)} pts, r_min={min(rs):.3f}, r_max={max(rs):.3f}")

# Also try collecting unconnected edges (those NOT in any wire)
print()
print("=== Edges in section directly (might include orphaned ones) ===")
direct_edges = list(section.edges())
edges_in_wires = set()
for w in wires:
    for e in w.edges():
        edges_in_wires.add(id(e))
orphans = [e for e in direct_edges if id(e) not in edges_in_wires]
print(f"  {len(orphans)} orphan edges")
for i, e in enumerate(orphans[:20]):
    if e.geom_type == bd.GeomType.CIRCLE:
        try:
            print(f"  orphan[{i}]: CIRCLE r={float(e.radius):.3f}")
        except Exception:
            print(f"  orphan[{i}]: CIRCLE (radius?)")
    else:
        print(f"  orphan[{i}]: {e.geom_type!r}")