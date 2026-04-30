"""Find a r=39 circle edge in the section and try every API to sample it."""
import sys
from pathlib import Path
import build123d as bd

path = Path(sys.argv[1])
solid = max(bd.import_step(str(path)).solids(), key=lambda s: s.volume)

plane = bd.Plane(
    origin=bd.Vector(0, 0, 0),
    x_dir=bd.Vector(1, 0, 0),
    z_dir=bd.Vector(0, 1, 0),
)
section = solid & plane

# Find one r=39 edge.
target = None
for e in section.edges():
    if e.geom_type == bd.GeomType.CIRCLE:
        try:
            r = float(e.radius)
        except Exception:
            r = -1
        if abs(r - 39.0) < 0.05:
            target = e
            break

if target is None:
    print("ERROR: no r=39 edge found")
    sys.exit(1)

print(f"Target edge geom_type: {target.geom_type}, radius: {float(target.radius):.3f}")
print(f"Edge has methods/attrs:")
for attr in ("positions", "position_at", "param_at", "param", "start_point", "end_point",
             "vertices", "length", "common_vertex", "to_axis", "u_min", "u_max",
             "param_min", "param_max"):
    if hasattr(target, attr):
        v = getattr(target, attr)
        print(f"  {attr}: {'method' if callable(v) else repr(v)}")

print()
print("=== positions([0, 0.5, 1]) ===")
try:
    out = target.positions([0.0, 0.5, 1.0])
    for i, p in enumerate(out):
        print(f"  [{i}]: ({p.X:.4f}, {p.Y:.4f}, {p.Z:.4f})  r={(p.X**2+p.Z**2)**0.5:.4f}")
except Exception as e:
    print(f"  RAISED: {type(e).__name__}: {e}")

print()
print("=== position_at(0.5) ===")
try:
    p = target.position_at(0.5)
    print(f"  ({p.X:.4f}, {p.Y:.4f}, {p.Z:.4f})  r={(p.X**2+p.Z**2)**0.5:.4f}")
except Exception as e:
    print(f"  RAISED: {type(e).__name__}: {e}")

print()
print("=== edge @ 0.5 (operator) ===")
try:
    p = target @ 0.5
    print(f"  ({p.X:.4f}, {p.Y:.4f}, {p.Z:.4f})  r={(p.X**2+p.Z**2)**0.5:.4f}")
except Exception as e:
    print(f"  RAISED: {type(e).__name__}: {e}")

print()
print("=== vertices() ===")
try:
    for i, v in enumerate(target.vertices()):
        print(f"  [{i}]: ({v.X:.4f}, {v.Y:.4f}, {v.Z:.4f})  r={(v.X**2+v.Z**2)**0.5:.4f}")
except Exception as e:
    print(f"  RAISED: {type(e).__name__}: {e}")