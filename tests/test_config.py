import dataclasses
import json
import math

import pytest

from prob_reco_sbi.flash import PRESETS, Geometry, Optics, Prior, Readout, SimConfig, matching_preset, preset
from prob_reco_sbi.flash.config import config_fields


def test_presets_are_the_physics_ladder():
    toggles = {name: (c.optics.absorption, c.optics.scattering, c.optics.reflection) for name, c in PRESETS.items()}
    assert toggles == {
        "straight": (False, False, False),
        "absorb": (True, False, False),
        "absorb-scatter": (True, True, False),
        "full": (True, True, True),
    }
    assert preset() == SimConfig() == PRESETS["full"]


def test_defaults_are_the_seminar_model():
    c = SimConfig()
    assert (c.geometry.radius, c.geometry.n_sensors, c.geometry.sensor_width) == (1.0, 32, 0.10)
    assert c.geometry.photocoverage == pytest.approx(32 * 0.10 / (2 * math.pi))
    assert round(c.geometry.photocoverage, 3) == 0.509
    o = c.optics
    assert (o.light_yield, o.speed_of_light, o.absorption_length, o.scattering_length) == (1000, 0.2, 1.5, 2.0)
    assert (o.reflectivity, o.reflection_model, o.photon_statistics, o.max_steps) == (0.5, "diffuse", "poisson", 80)
    r = c.readout
    assert (r.detection_efficiency, r.time_jitter, r.dark_rate, r.readout_window) == (1.0, 1.0, 0.0, 100.0)
    p = c.prior
    assert (p.r_max, p.energy_min, p.energy_max, p.t0_min, p.t0_max) == (1.0, 0.5, 5.0, 0.0, 10.0)
    assert p.fixed_vertex is None
    assert p.fixed_energy is None


def test_overrides_on_top_of_a_preset():
    c = preset("straight", reflection=True, n_sensors=64, sensor_width=0.05, time_jitter=0, fixed_vertex=(0.1, -0.2))
    assert c.optics.reflection
    assert not c.optics.absorption
    assert c.geometry.n_sensors == 64
    assert c.readout.time_jitter == 0.0
    assert c.prior.fixed_vertex == (0.1, -0.2)
    assert PRESETS["straight"].optics.reflection is False  # the preset itself is unchanged
    assert c.replace(readout=Readout(dark_rate=5.0)).readout.dark_rate == 5.0


def test_unknown_names_give_helpful_errors():
    with pytest.raises(TypeError, match="absorption"):
        preset("full", absorbtion=False)
    with pytest.raises(ValueError, match="unknown preset"):
        preset("everything")


@pytest.mark.parametrize("name", list(PRESETS))
def test_dict_and_json_round_trip(name):
    c = preset(name, fixed_vertex=(0.25, 0.5), fixed_energy=2.0, dark_rate=1e6)
    assert SimConfig.from_dict(c.to_dict()) == c
    assert SimConfig.from_json(c.to_json()) == c
    assert json.loads(c.to_json())["prior"]["fixed_vertex"] == [0.25, 0.5]
    assert SimConfig.from_dict({}) == SimConfig()
    with pytest.raises(ValueError, match="unknown parameters"):
        SimConfig.from_dict({"optics": {"colour": "blue"}})
    with pytest.raises(ValueError, match="unknown configuration sections"):
        SimConfig.from_dict({"electronics": {}})


def test_configurations_are_frozen_and_hashable():
    c = SimConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.optics.reflectivity = 1.0  # type: ignore[misc]
    assert len({c, preset("full"), preset("absorb")}) == 2


def test_values_are_coerced():
    import numpy as np

    g = Geometry(radius=np.float32(2.0), n_sensors=np.int64(16))
    assert type(g.radius) is float
    assert type(g.n_sensors) is int
    assert Prior(fixed_vertex=[0, 0]).fixed_vertex == (0.0, 0.0)


@pytest.mark.parametrize(
    ("make", "error"),
    [
        (lambda: Optics(reflectivity=1.5), ValueError),
        (lambda: Optics(absorption_length=-1), ValueError),
        (lambda: Optics(scattering_length=math.inf), ValueError),
        (lambda: Optics(max_steps=0), ValueError),
        (lambda: Optics(reflection_model="glossy"), ValueError),
        (lambda: Optics(photon_statistics="binomial"), ValueError),
        (lambda: Optics(absorption="yes"), TypeError),
        (lambda: Geometry(n_sensors=2.5), TypeError),
        (lambda: Geometry(n_sensors=0), ValueError),
        (lambda: Geometry(n_sensors=64, sensor_width=0.2), ValueError),
        (lambda: Readout(detection_efficiency=1.1), ValueError),
        (lambda: Readout(time_jitter=-1), ValueError),
        (lambda: Prior(energy_min=3, energy_max=2), ValueError),
        (lambda: Prior(t0_min=5, t0_max=1), ValueError),
        (lambda: preset("full", r_max=1.2), ValueError),
        (lambda: preset("full", fixed_vertex=(0.9, 0.9)), ValueError),
        (lambda: SimConfig(optics=Readout()), TypeError),
    ],
)
def test_invalid_values_are_rejected(make, error):
    with pytest.raises(error):
        make()


def test_matching_preset():
    assert matching_preset(preset("absorb")) == "absorb"
    assert matching_preset(preset("absorb", reflectivity=0.9)) is None


def test_every_parameter_has_help_and_unique_name():
    names = [f.name for _, f in config_fields()]
    assert len(names) == len(set(names))
    assert all(f.metadata["help"] for _, f in config_fields())


def test_summary_mentions_the_processes():
    text = preset("absorb").summary()
    assert "lambda_a = 1.5 m" in text
    assert "scattering: off" in text
    assert "reflection: off" in text
