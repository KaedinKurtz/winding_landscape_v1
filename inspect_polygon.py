"""Run my actual _slice_at code and dump what it produced for each wire."""
import sys
from pathlib import Path
import numpy as np
import build123d as bd
from winding_landscape.geometry.extraction import _slice_at, _identify_rotation_axis, _stack_length_along_axis

path = Path(sys.argv[1])
solid = max(bd.import_step(str(path)).solids(), key=lambda s: s.volume)
axis_dir, axis_origin = _identify_rotation_axis(solid)
stack_len = _stack_length_along_axis(solid, axis_dir, axis_origin)

section = _slice_at(solid, axis_dir, axis_origin, stack_len * 0.5)
print(f"_slice_at produced {len(section.polygons)} polygon(s):")
for i, p in enumerate(section.polygons):
    radii = np.linalg.norm(p, axis=1)
    print(f"  poly[{i}]: {len(p)} points, r min/mean/max = {radii.min():.4f} / {radii.mean():.4f} / {radii.max():.4f}")
    # Show the histogram of radii rounded to 0.1mm
    from collections import Counter
    hist = Counter(round(float(r), 1) for r in radii)
    print(f"    radius histogram: {dict(sorted(hist.items()))}")