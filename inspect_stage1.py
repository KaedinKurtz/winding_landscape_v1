"""Run Stage 1 sub-steps individually and report what each produced."""
import sys
from pathlib import Path
import numpy as np

from winding_landscape.geometry.extraction import (
    _load_brep,
    _identify_rotation_axis,
    _stack_length_along_axis,
    _slice_at,
    _identify_bore_and_outer_radii,
    _measure_slots,
    _verify_periodicity,
)
from winding_landscape.config import load_constraints

brep_path = sys.argv[1]
constraints_path = sys.argv[2] if len(sys.argv) > 2 else "example_constraints.yaml"

solid = _load_brep(Path(brep_path))
print("Loaded solid.\n")

axis_dir, axis_origin = _identify_rotation_axis(solid)
print(f"Axis dir:    {axis_dir}")
print(f"Axis origin: {axis_origin}\n")

stack_len = _stack_length_along_axis(solid, axis_dir, axis_origin)
print(f"Stack length: {stack_len:.3f} mm\n")

section = _slice_at(solid, axis_dir, axis_origin, stack_len * 0.5)
print(f"Section produced {len(section.polygons)} polygon(s):")
for i, p in enumerate(section.polygons):
    radii = np.linalg.norm(p, axis=1)
    print(f"  poly[{i}]: {len(p)} points, r min/mean/max = {radii.min():.2f} / {radii.mean():.2f} / {radii.max():.2f} mm")
print()

bore_r, outer_r = _identify_bore_and_outer_radii(section)
print(f"Bore radius:  {bore_r:.3f} mm")
print(f"Outer radius: {outer_r:.3f} mm\n")

slot_count, slot_features = _measure_slots(section, bore_r, outer_r)
print(f"Detected slot count: {slot_count}")
print("Slot features:")
for k, v in slot_features.items():
    print(f"  {k}: {v:.3f}")
print()

dev = _verify_periodicity(section, slot_count, bore_r, outer_r)
print(f"Periodicity deviation: {dev:.3f} mm  (warn @ 0.05, fatal @ 1.0)")