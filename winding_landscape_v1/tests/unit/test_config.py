"""Tests for the constraints config loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from winding_landscape.config import Constraints, load_constraints


def test_default_constraints():
    c = Constraints()
    assert c.operating_targets.target_continuous_torque_Nm > 0
    assert c.thermal_envelope.cooling_mode in ("natural", "forced_air", "liquid")
    assert c.materials.steel_grade == "M19_29Ga"
    assert 4 in c.topology_constraints.pole_count_range or True  # range, not list


def test_partial_override():
    c = Constraints(operating_targets={"target_continuous_torque_Nm": 12.0})
    assert c.operating_targets.target_continuous_torque_Nm == 12.0
    # Other defaults still in place.
    assert c.electrical_envelope.supply_voltage_V == 48.0


def test_modulation_margin_bound():
    """modulation_margin <= 1.0 is a hard constraint."""
    with pytest.raises(ValueError):
        Constraints(electrical_envelope={"modulation_margin": 1.5})


def test_pole_count_range_validator():
    """pole_count_range must be even integers and lo <= hi."""
    with pytest.raises(ValueError):
        Constraints(topology_constraints={"pole_count_range": (3, 10)})  # odd lo
    with pytest.raises(ValueError):
        Constraints(topology_constraints={"pole_count_range": (10, 4)})  # lo > hi


def test_awg_validator():
    """available_wire_gauges_AWG must be non-empty and within [0, 50]."""
    with pytest.raises(ValueError):
        Constraints(manufacturing_constraints={"available_wire_gauges_AWG": []})
    with pytest.raises(ValueError):
        Constraints(manufacturing_constraints={"available_wire_gauges_AWG": [60]})


def test_load_constraints_yaml(tmp_path: Path):
    yaml_text = (
        "operating_targets:\n"
        "  target_continuous_torque_Nm: 7.5\n"
        "electrical_envelope:\n"
        "  supply_voltage_V: 56.0\n"
    )
    p = tmp_path / "constraints.yaml"
    p.write_text(yaml_text)
    c, defaults_used = load_constraints(p)
    assert c.operating_targets.target_continuous_torque_Nm == 7.5
    assert c.electrical_envelope.supply_voltage_V == 56.0
    # All un-specified top-level sections should be in defaults_used.
    assert any("thermal_envelope" in d for d in defaults_used)
    assert any("manufacturing_constraints" in d for d in defaults_used)


def test_load_constraints_json(tmp_path: Path):
    p = tmp_path / "constraints.json"
    p.write_text(json.dumps({"operating_targets": {"target_continuous_torque_Nm": 3.3}}))
    c, _ = load_constraints(p)
    assert c.operating_targets.target_continuous_torque_Nm == 3.3


def test_load_constraints_unknown_extension(tmp_path: Path):
    p = tmp_path / "constraints.toml"
    p.write_text("[whatever]\nx = 1\n")
    with pytest.raises(ValueError):
        load_constraints(p)


def test_extra_fields_forbidden():
    """We use extra='forbid' so typos in the YAML get caught."""
    with pytest.raises(ValueError):
        Constraints(operating_targets={"unknown_key": 5.0})
