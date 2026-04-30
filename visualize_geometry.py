"""Stage 1 debug visualization for stator geometry extraction.

Re-runs the extraction on a BREP and produces a four-panel matplotlib figure:

  1. Mid-stack cross-section in the part's xy frame, with bore/OD circles and
     slot-pitch radials overlaid. Confirms the slice came out clean and the
     polygons have the expected radial extents.
  2. A single slot, rotated so its center is at the top, with dimension lines
     for opening width, bottom width, total depth, and yoke thickness. Hover
     check: do the arrows visually match what the algorithm reported?
  3. Angular profile r(theta) over the full 360 degrees, with sampled points,
     OD-coverage points highlighted, and the slot detection threshold drawn.
     Useful for spotting bins that were misclassified or missing.
  4. Periodicity overlay: r(theta) and r(theta + 2*pi/Q) plotted together over
     a few slot pitches, with the deviation envelope shaded. A clean part
     shows the two curves overlapping; any defect or non-periodic feature
     stands out as a bump.

Usage
-----
    python visualize_geometry.py [BREP_PATH] [--save PATH] [--no-show]

Defaults to the 10010 Stator test part if no BREP is given.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from winding_landscape.geometry.extraction import (
    _identify_rotation_axis,
    _load_brep,
    _measure_slots,
    _slice_at,
    _stack_length_along_axis,
    periodicity_samples_per_slot,
    radial_min_envelope,
)


DEFAULT_BREP = "/Users/kaedinkurtz/Downloads/10010 Stator.STEP"


def visualize(brep_path: Path, save: Path | None = None, show: bool = True) -> None:
    if not brep_path.exists():
        raise FileNotFoundError(f"BREP not found: {brep_path}")

    print(f"Loading {brep_path.name}...", flush=True)
    solid = _load_brep(brep_path)
    axis, axis_origin, radii = _identify_rotation_axis(solid)
    length, ax_min, ax_max = _stack_length_along_axis(solid, axis, axis_origin)
    midstack = 0.5 * (ax_min + ax_max)
    print(f"Slicing at axial offset {midstack:.3f} mm...", flush=True)
    section = _slice_at(solid, axis, axis_origin, midstack)

    bore_r = float(min(radii))
    outer_r = float(max(radii))
    print(f"Measuring slots (bore={bore_r:.2f}, OD={outer_r:.2f})...", flush=True)
    slot_count, slot_features = _measure_slots(section, bore_r, outer_r)

    fig, axs = plt.subplots(2, 2, figsize=(15, 13))
    title_line1 = f"Stage 1 diagnostics — {brep_path.name}"
    title_line2 = (
        f"Q={slot_count}  •  OD={2*outer_r:.2f}mm  •  ID={2*bore_r:.2f}mm  "
        f"•  L={length:.2f}mm  •  slot depth={slot_features['total_depth_mm']:.2f}mm  "
        f"•  opening={slot_features['opening_width_mm']:.2f}mm  "
        f"•  yoke={slot_features['yoke_thickness_mm']:.2f}mm"
    )
    fig.suptitle(f"{title_line1}\n{title_line2}", fontsize=11)

    _draw_section(axs[0, 0], section, bore_r, outer_r, slot_count)
    _draw_slot_zoom(axs[0, 1], section, bore_r, outer_r, slot_count, slot_features)
    _draw_angular_profile(axs[1, 0], section, outer_r, slot_count)
    _draw_periodicity(axs[1, 1], section, slot_count, outer_r)

    fig.tight_layout(rect=(0, 0, 1, 0.94))

    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=120, bbox_inches="tight")
        print(f"Saved figure to {save}")

    if show:
        plt.show()


def _od_side_boundary(section, outer_r: float) -> np.ndarray:
    """Pool boundary points from polygons whose max radius reaches the OD."""
    polys = [
        p for p in section.polygons
        if float(np.max(np.linalg.norm(p, axis=1))) >= outer_r - 0.5
    ]
    if not polys:
        return np.zeros((0, 2))
    return np.vstack(polys)


def _draw_section(ax, section, bore_r: float, outer_r: float, slot_count: int) -> None:
    ax.set_aspect("equal")

    for p in section.polygons:
        rs = np.linalg.norm(p, axis=1)
        is_outer = rs.max() >= outer_r - 0.5
        color = "tab:blue" if is_outer else "tab:orange"
        label = "OD-side polygon" if is_outer else "Bore-side polygon"
        ax.plot(p[:, 0], p[:, 1], ".", color=color, ms=2.5, alpha=0.65, label=label)

    theta = np.linspace(0, 2 * np.pi, 360)
    ax.plot(bore_r * np.cos(theta), bore_r * np.sin(theta), "g--",
            lw=1.0, alpha=0.7, label=f"Bore (r={bore_r:.2f})")
    ax.plot(outer_r * np.cos(theta), outer_r * np.sin(theta), "r--",
            lw=1.0, alpha=0.7, label=f"OD (r={outer_r:.2f})")

    for k in range(slot_count):
        t = 2 * np.pi * k / slot_count
        ax.plot([0, outer_r * 1.05 * np.cos(t)],
                [0, outer_r * 1.05 * np.sin(t)],
                "k:", lw=0.4, alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    ax.legend([h for h, _ in uniq], [l for _, l in uniq], loc="upper right", fontsize=8)

    ax.set_title("Mid-stack cross-section (xy in part frame)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.grid(True, alpha=0.3)


def _draw_slot_zoom(
    ax,
    section,
    bore_r: float,
    outer_r: float,
    slot_count: int,
    slot_features: dict,
) -> None:
    ax.set_aspect("equal")
    boundary = _od_side_boundary(section, outer_r)
    if len(boundary) == 0:
        ax.text(0.5, 0.5, "No OD-side polygon", ha="center", va="center",
                transform=ax.transAxes)
        return

    theta_all = np.arctan2(boundary[:, 1], boundary[:, 0])
    r_all = np.linalg.norm(boundary, axis=1)

    sub_od_mask = r_all < outer_r - 0.5
    sub_theta = np.sort(theta_all[sub_od_mask])
    if len(sub_theta) >= 3:
        diffs = np.diff(sub_theta)
        big = diffs > math.radians(2.0)
        starts = np.concatenate([[0], np.where(big)[0] + 1])
        ends = np.concatenate([np.where(big)[0] + 1, [len(sub_theta)]])
        first_slot = sub_theta[starts[0]:ends[0]]
        slot_center = float(np.mean(first_slot))
    else:
        slot_center = 0.0

    rotation = math.pi / 2 - slot_center
    cos_t, sin_t = math.cos(rotation), math.sin(rotation)
    rot_x = boundary[:, 0] * cos_t - boundary[:, 1] * sin_t
    rot_y = boundary[:, 0] * sin_t + boundary[:, 1] * cos_t

    rot_theta = np.arctan2(rot_x, rot_y)
    pitch = 2 * math.pi / slot_count
    in_window = np.abs(rot_theta) < 1.2 * pitch

    ax.plot(rot_x[in_window], rot_y[in_window], ".", color="tab:blue",
            ms=3, alpha=0.7, label="Section points")

    arc_t = np.linspace(math.pi / 2 - 1.3 * pitch, math.pi / 2 + 1.3 * pitch, 100)
    ax.plot(outer_r * np.cos(arc_t), outer_r * np.sin(arc_t),
            "r--", lw=1.0, alpha=0.7, label=f"OD r={outer_r:.2f}")
    ax.plot(bore_r * np.cos(arc_t), bore_r * np.sin(arc_t),
            "g--", lw=1.0, alpha=0.7, label=f"Bore r={bore_r:.2f}")

    bottom_r = outer_r - slot_features["total_depth_mm"]
    ax.plot(bottom_r * np.cos(arc_t), bottom_r * np.sin(arc_t),
            color="purple", lw=1.0, ls=":", alpha=0.7,
            label=f"Slot bottom r={bottom_r:.2f}")

    opening_half = 0.5 * slot_features["opening_width_mm"] / outer_r
    bottom_half = 0.5 * slot_features["bottom_width_mm"] / bottom_r
    op_l = (outer_r * math.cos(math.pi / 2 - opening_half),
            outer_r * math.sin(math.pi / 2 - opening_half))
    op_r = (outer_r * math.cos(math.pi / 2 + opening_half),
            outer_r * math.sin(math.pi / 2 + opening_half))
    bo_l = (bottom_r * math.cos(math.pi / 2 - bottom_half),
            bottom_r * math.sin(math.pi / 2 - bottom_half))
    bo_r = (bottom_r * math.cos(math.pi / 2 + bottom_half),
            bottom_r * math.sin(math.pi / 2 + bottom_half))

    _dim_arrow(ax, op_l, op_r, f"opening: {slot_features['opening_width_mm']:.2f} mm",
               offset=(0, 1.5), color="darkred")
    _dim_arrow(ax, bo_l, bo_r, f"bottom: {slot_features['bottom_width_mm']:.2f} mm",
               offset=(0, -1.5), color="purple")

    depth_x = (outer_r + 1.0) * math.cos(math.pi / 2 + 1.1 * pitch * 0.5)
    depth_top = (depth_x, outer_r + 0)
    depth_bot = (depth_x, bottom_r + 0)
    _dim_arrow(ax, depth_top, depth_bot,
               f"depth: {slot_features['total_depth_mm']:.2f} mm",
               offset=(1.0, 0), color="navy", rotation=90)

    yoke_x = (bore_r - 1.5) * math.cos(math.pi / 2 - 1.1 * pitch * 0.5)
    yoke_top = (yoke_x, bottom_r)
    yoke_bot = (yoke_x, bore_r)
    _dim_arrow(ax, yoke_top, yoke_bot,
               f"yoke: {slot_features['yoke_thickness_mm']:.2f} mm",
               offset=(-1.0, 0), color="darkgreen", rotation=90)

    win_x = rot_x[in_window]
    win_y = rot_y[in_window]
    if len(win_x) > 0:
        x_min = min(win_x.min(), bore_r * math.cos(math.pi / 2 + 1.2 * pitch)) - 2
        x_max = max(win_x.max(), bore_r * math.cos(math.pi / 2 - 1.2 * pitch)) + 2
        y_min = min(win_y.min(), bore_r) - 2
        y_max = win_y.max() + 3
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    text = (
        f"Q (slots): {slot_count}\n"
        f"slot opening: {slot_features['opening_width_mm']:.3f} mm\n"
        f"slot bottom:  {slot_features['bottom_width_mm']:.3f} mm\n"
        f"total depth:  {slot_features['total_depth_mm']:.3f} mm\n"
        f"yoke:         {slot_features['yoke_thickness_mm']:.3f} mm\n"
        f"tooth (min):  {slot_features['tooth_width_min_mm']:.3f} mm\n"
        f"slot area:    {slot_features['area_mm2']:.3f} mm²"
    )
    ax.text(0.02, 0.02, text, transform=ax.transAxes, va="bottom", ha="left",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85))

    ax.set_title(f"Slot detail (rotated; native slot at θ={math.degrees(slot_center):.1f}°)")
    ax.set_xlabel("x' (mm)")
    ax.set_ylabel("y' (mm)")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)


def _dim_arrow(
    ax, p1: tuple[float, float], p2: tuple[float, float], label: str,
    offset: tuple[float, float] = (0, 0), color: str = "black",
    rotation: float = 0,
) -> None:
    """Draw a two-headed dimension arrow with a label."""
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.2))
    mid = ((p1[0] + p2[0]) / 2 + offset[0], (p1[1] + p2[1]) / 2 + offset[1])
    ax.text(mid[0], mid[1], label, color=color, fontsize=8,
            ha="center", va="center", rotation=rotation,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=color, alpha=0.9, lw=0.5))


def _draw_angular_profile(ax, section, outer_r: float, slot_count: int) -> None:
    boundary = _od_side_boundary(section, outer_r)
    if len(boundary) == 0:
        ax.text(0.5, 0.5, "No OD-side polygon", ha="center", va="center",
                transform=ax.transAxes)
        return

    theta = np.arctan2(boundary[:, 1], boundary[:, 0])
    r = np.linalg.norm(boundary, axis=1)

    ax.plot(np.degrees(theta), r, ".", color="tab:blue", ms=1.5,
            alpha=0.4, label="all sampled points")

    od_tol = 0.1
    od_mask = r >= outer_r - od_tol
    ax.plot(np.degrees(theta[od_mask]), r[od_mask], ".", color="red", ms=2.5,
            alpha=0.85, label=f"OD coverage (r ≥ {outer_r - od_tol:.2f})")

    n_bins = 720
    bin_edges = np.linspace(-math.pi, math.pi, n_bins + 1)
    bin_idx = np.clip(np.digitize(theta, bin_edges) - 1, 0, n_bins - 1)
    r_per_bin = np.full(n_bins, outer_r)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            r_per_bin[b] = float(np.median(r[mask]))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ax.plot(np.degrees(bin_centers), r_per_bin, "-", color="orange",
            lw=0.8, alpha=0.9, label="median r per 0.5° bin")

    r_min_global = float(r_per_bin.min())
    threshold = outer_r - 0.2 * (outer_r - r_min_global)
    ax.axhline(threshold, color="darkorange", lw=1.0, ls="--", alpha=0.8,
               label=f"slot threshold ({threshold:.2f})")
    ax.axhline(outer_r, color="red", lw=0.6, ls=":", alpha=0.5)
    ax.axhline(r_min_global, color="purple", lw=0.6, ls=":", alpha=0.5)

    ax.set_title(f"Angular profile r(θ) — {slot_count} slot regions, "
                 f"slot pitch = {360.0/slot_count:.2f}°")
    ax.set_xlabel("θ (degrees)")
    ax.set_ylabel("r (mm)")
    ax.set_xlim(-180, 180)
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(True, alpha=0.3)


def _draw_periodicity(ax, section, slot_count: int, outer_r: float) -> None:
    boundary = _od_side_boundary(section, outer_r)
    if len(boundary) == 0:
        ax.text(0.5, 0.5, "No OD-side polygon", ha="center", va="center",
                transform=ax.transAxes)
        return

    theta = np.arctan2(boundary[:, 1], boundary[:, 0])
    r = np.linalg.norm(boundary, axis=1)
    order = np.argsort(theta)
    theta = theta[order]
    r = r[order]

    samples_per_slot = periodicity_samples_per_slot(slot_count, outer_r)
    n_uniform = slot_count * samples_per_slot
    arc_step_mm = (2.0 * math.pi * outer_r) / n_uniform
    theta_u = np.linspace(-math.pi, math.pi, n_uniform, endpoint=False)
    r_u = radial_min_envelope(theta, r, n_uniform)
    r_rolled = np.roll(r_u, samples_per_slot)
    deviation = np.abs(r_u - r_rolled)

    ax.plot(np.degrees(theta_u), r_u, "-", color="tab:blue", lw=0.9,
            alpha=0.85, label="r(θ)")
    ax.plot(np.degrees(theta_u), r_rolled, "--", color="tab:green", lw=0.9,
            alpha=0.85, label=f"r(θ − 2π/{slot_count})")
    ax.fill_between(np.degrees(theta_u),
                    np.minimum(r_u, r_rolled),
                    np.maximum(r_u, r_rolled),
                    color="red", alpha=0.20,
                    label=f"deviation (max {deviation.max():.3f} mm)")

    pitch_deg = 360.0 / slot_count
    ax.set_xlim(-2 * pitch_deg, 2 * pitch_deg)
    ax.set_title(
        f"Periodicity check: max deviation = {deviation.max():.3f} mm   "
        f"(grid: {samples_per_slot} samples/slot, ≈{arc_step_mm:.3f} mm arc step at OD)"
    )
    ax.set_xlabel("θ (degrees)")
    ax.set_ylabel("r (mm)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("brep", nargs="?", default=DEFAULT_BREP,
                        help=f"Path to BREP (default: {DEFAULT_BREP})")
    parser.add_argument("--save", default=None,
                        help="Save figure to this PNG path")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't open the interactive matplotlib window")
    args = parser.parse_args()

    brep_path = Path(args.brep).expanduser()
    save_path = Path(args.save).expanduser() if args.save else None

    try:
        visualize(brep_path, save=save_path, show=not args.no_show)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
