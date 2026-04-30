#!/usr/bin/env python3
"""Tiny stdlib-only test harness for environments without pytest installed.

Runs the same test_*.py files pytest would run, but using Python introspection.
Supports a useful subset:
  - test_* functions in test_*.py modules
  - The 'materials_db', 'synthetic_geometry', 'reasonable_constraints' fixtures
    from tests/conftest.py
  - tmp_path (creates a temp dir per test)
  - parametrize decorator (limited support: list-of-tuples)
  - pytest.approx, pytest.raises (stdlib equivalents)

NOT a substitute for pytest -- this exists only because the offline test
environment can't pip install. In the user's installed env, just run pytest.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path

# ---------- pytest shim ----------

class _Approx:
    def __init__(self, expected, rel=1e-6, abs_=1e-12):
        self.expected = expected
        self.rel = rel
        self.abs = abs_
    def __eq__(self, other):
        return abs(other - self.expected) <= max(self.rel * abs(self.expected), self.abs)
    def __repr__(self):
        return f"approx({self.expected})"

def approx(expected, rel=1e-6, abs=1e-12):
    return _Approx(expected, rel=rel, abs_=abs)

@contextmanager
def raises(exc_type):
    try:
        yield
    except exc_type:
        return
    raise AssertionError(f"Expected {exc_type.__name__} to be raised")

class _ParamMarker:
    def __init__(self, argnames, argvalues):
        self.argnames = [a.strip() for a in argnames.split(",")]
        self.argvalues = argvalues

class _Mark:
    @staticmethod
    def parametrize(argnames, argvalues):
        def decorator(func):
            func._pytest_parametrize = _ParamMarker(argnames, argvalues)
            return func
        return decorator

class _PytestStub:
    approx = staticmethod(approx)
    raises = staticmethod(raises)
    mark = _Mark()

sys.modules["pytest"] = _PytestStub  # type: ignore[assignment]


# ---------- fixture providers ----------

def _build_fixture(name: str, tmp_root: Path):
    if name == "materials_db":
        from winding_landscape.materials.database import load_materials
        return load_materials()
    if name == "synthetic_geometry":
        from winding_landscape.geometry.stator_geometry import StatorGeometry
        return StatorGeometry(
            bore_radius_mm=10.0, outer_radius_mm=25.0, stack_length_mm=20.0,
            slot_count=12, slot_opening_width_mm=2.0, slot_opening_depth_mm=1.0,
            slot_total_depth_mm=8.0, slot_bottom_width_mm=4.5,
            slot_area_mm2=30.0, slot_useful_area_mm2=27.0,
            tooth_width_min_mm=4.0, yoke_thickness_mm=7.0,
            airgap_mm=0.5, periodicity_tolerance_mm=0.01,
        )
    if name == "reasonable_constraints":
        from winding_landscape.config import Constraints
        return Constraints(
            operating_targets={
                "target_continuous_torque_Nm": 0.15,
                "target_peak_torque_Nm": 0.5,
                "target_max_speed_rpm": 2000,
            },
            electrical_envelope={
                "supply_voltage_V": 36.0,
                "max_phase_current_A_rms": 10.0,
            },
        )
    if name == "tmp_path":
        d = Path(tempfile.mkdtemp(dir=tmp_root))
        return d
    raise KeyError(f"Unknown fixture {name}")


# ---------- test discovery + execution ----------

def discover_test_files(root: Path) -> list[Path]:
    return sorted(root.rglob("test_*.py"))


def import_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def run_test_function(func, name, tmp_root):
    """Run a single test function, applying fixtures by parameter name."""
    sig = inspect.signature(func)

    # Handle parametrize.
    if hasattr(func, "_pytest_parametrize"):
        marker = func._pytest_parametrize
        successes = 0
        failures = []
        for case_idx, case in enumerate(marker.argvalues):
            kwargs = dict(zip(marker.argnames, case, strict=False))
            # Fill in any other params from fixtures.
            for p in sig.parameters:
                if p not in kwargs:
                    kwargs[p] = _build_fixture(p, tmp_root)
            try:
                func(**kwargs)
                successes += 1
            except Exception:
                failures.append((case_idx, case, traceback.format_exc()))
        return successes, failures
    else:
        kwargs = {p: _build_fixture(p, tmp_root) for p in sig.parameters}
        try:
            func(**kwargs)
            return 1, []
        except Exception:
            return 0, [(None, None, traceback.format_exc())]


def main():
    tests_dir = Path(__file__).parent
    test_files = discover_test_files(tests_dir)
    print(f"Discovered {len(test_files)} test files.")

    tmp_root = Path(tempfile.mkdtemp(prefix="wl_tests_"))
    total_pass = 0
    total_fail = 0
    failures_by_test = []

    for tf in test_files:
        if "__init__" in tf.name:
            continue
        rel = tf.relative_to(tests_dir)
        print(f"\n=== {rel} ===")
        try:
            mod = import_module_from_path(tf)
        except Exception as e:
            print(f"  IMPORT FAILED: {e}")
            traceback.print_exc()
            total_fail += 1
            continue
        for name, obj in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            ok, fails = run_test_function(obj, name, tmp_root)
            total_pass += ok
            for fail in fails:
                total_fail += 1
                failures_by_test.append((f"{rel}::{name}", fail))
            if fails:
                # Mark failures on the same line.
                for f in fails:
                    case_idx, case, _ = f
                    if case_idx is not None:
                        print(f"  FAIL: {name}[{case_idx}]")
                    else:
                        print(f"  FAIL: {name}")
            else:
                print(f"  pass: {name} ({ok} run{'s' if ok > 1 else ''})")

    print()
    print(f"=== Summary: {total_pass} passed, {total_fail} failed ===")
    if failures_by_test:
        print("\n=== Failures ===")
        for name, fail in failures_by_test:
            case_idx, case, tb = fail
            print(f"\n--- {name} {f'(case {case_idx}: {case})' if case_idx is not None else ''} ---")
            print(tb)
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
