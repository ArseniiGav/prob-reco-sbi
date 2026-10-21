"""Configuration of the toy detector: geometry, optics, readout and prior, plus the physics presets.

A configuration is a frozen :class:`SimConfig` made of four parts:

* :class:`Geometry`: disc radius and the photosensor arcs on the wall;
* :class:`Optics`: light yield, speed of light and the three switchable processes (absorption, Rayleigh scattering,
  wall reflection);
* :class:`Readout`: detection efficiency, time jitter and dark noise of the sensors;
* :class:`Prior`: the distribution of the event parameters (vertex, energy) and of the event time t0.

Every default equals the seminar model, i.e. the ``"full"`` preset. Parameter names are unique across the four parts,
so a modified copy is made with flat keywords::

    from prob_reco_sbi.flash import preset

    config = preset("straight", reflection=True, n_sensors=16)
    config = config.replace(time_jitter=0.0)
"""

from __future__ import annotations

import dataclasses
import difflib
import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

__all__ = [
    "PRESETS",
    "Geometry",
    "Optics",
    "Prior",
    "Readout",
    "SimConfig",
    "config_fields",
    "matching_preset",
    "preset",
]

PhotonStatistics = Literal["poisson", "fixed"]
ReflectionModel = Literal["diffuse", "specular"]


def _param(default: Any, help: str, unit: str = "", **extra: Any) -> Any:
    """Dataclass field with a help text and a unit (used by the documentation and the command line)."""
    return field(default=default, metadata={"help": help, "unit": unit, **extra})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _coerce(f: dataclasses.Field[Any], value: Any, owner: str) -> Any:
    """Convert ``value`` to the type of the field's default (also turns NumPy scalars into Python numbers)."""
    name = f"{owner}.{f.name}"
    default = f.default
    if value is None:
        if default is None:
            return None
        raise TypeError(f"{name} must not be None")
    if isinstance(default, bool):
        if not isinstance(value, bool) and type(value).__name__ != "bool_":
            raise TypeError(f"{name} must be True or False, got {value!r}")
        return bool(value)
    if isinstance(default, str):
        choices = f.metadata["choices"]
        if value not in choices:
            raise ValueError(f"{name} must be one of {', '.join(map(repr, choices))}, got {value!r}")
        return str(value)
    if isinstance(value, (bool, str)):
        raise TypeError(f"{name} must be a number, got {value!r}")
    if isinstance(default, int):
        if float(value) != int(value):
            raise TypeError(f"{name} must be an integer, got {value!r}")
        return int(value)
    numbers = tuple(float(v) for v in value) if f.metadata.get("nargs") == 2 else (float(value),)
    if len(numbers) != (2 if f.metadata.get("nargs") == 2 else 1):
        raise ValueError(f"{name} must be a pair of numbers, got {value!r}")
    if not all(math.isfinite(v) for v in numbers):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return numbers if len(numbers) == 2 else numbers[0]


class _Part:
    """Behaviour shared by the four configuration parts: type coercion and validation on construction."""

    def __post_init__(self) -> None:
        for f in dataclasses.fields(self):  # type: ignore[arg-type]
            object.__setattr__(self, f.name, _coerce(f, getattr(self, f.name), type(self).__name__))
        self._validate()

    def _validate(self) -> None:
        """Check the parameter values (each part overrides this)."""


@dataclass(frozen=True)
class Geometry(_Part):
    """Disc detector with photosensor arcs on its wall.

    Sensor ``j`` (``j = 0 ... n_sensors - 1``) is centred at the angle ``phi_j = 2 pi j / n_sensors`` and covers the
    arc ``|phi - phi_j| <= sensor_width / (2 radius)``. The photocoverage is ``n_sensors * sensor_width / (2 pi R)``
    (0.509 for the defaults).
    """

    radius: float = _param(1.0, "detector radius R", "m")
    n_sensors: int = _param(32, "number of photosensors M on the wall")
    sensor_width: float = _param(0.10, "arc length a of each sensor", "m")

    @property
    def photocoverage(self) -> float:
        """Fraction of the wall covered by sensors, ``M a / (2 pi R)``."""
        return self.n_sensors * self.sensor_width / (2 * math.pi * self.radius)

    def _validate(self) -> None:
        _require(self.radius > 0, f"radius must be positive, got {self.radius}")
        _require(self.n_sensors >= 1, f"n_sensors must be at least 1, got {self.n_sensors}")
        _require(self.sensor_width > 0, f"sensor_width must be positive, got {self.sensor_width}")
        _require(
            self.photocoverage <= 1 + 1e-12,
            f"sensors overlap: {self.n_sensors} x {self.sensor_width} m exceeds the wall length "
            f"2 pi R = {2 * math.pi * self.radius:.4g} m",
        )


@dataclass(frozen=True)
class Optics(_Part):
    """Light production and propagation.

    Each event emits ``N`` photons isotropically at its vertex, ``N ~ Poisson(Y E)`` (or ``N = round(Y E)`` with
    ``photon_statistics="fixed"``). A photon flies straight with speed ``c_n`` until it is absorbed or scattered in the
    bulk, or reaches the wall. A process that is switched off behaves as an infinite length (absorption, scattering)
    or as a black wall (reflection); with absorption and scattering both off, photons fly straight to the wall.
    Photons still in flight after ``max_steps`` interactions are counted as lost.

    Specular reflection in a perfect circle keeps the angle of incidence: a photon emitted exactly at the centre
    bounces along one diameter; if that diameter misses the sensors, the photon is absorbed by the wall at one of its
    bounces (with ``reflectivity = 1`` it bounces until ``max_steps`` and is counted as lost).
    """

    light_yield: float = _param(1000.0, "photons emitted per MeV, Y", "photons/MeV")
    photon_statistics: PhotonStatistics = _param(
        "poisson",
        "number of photons per event: 'poisson' (N ~ Poisson(Y E)) or 'fixed' (N = round(Y E))",
        choices=("poisson", "fixed"),
    )
    speed_of_light: float = _param(0.2, "speed of light in the medium, c_n", "m/ns")
    absorption: bool = _param(True, "absorption in the bulk")
    absorption_length: float = _param(1.5, "absorption length lambda_a", "m")
    scattering: bool = _param(True, "Rayleigh scattering in the bulk, phase function p(theta) ~ 1 + cos^2(theta)")
    scattering_length: float = _param(2.0, "scattering length lambda_s", "m")
    reflection: bool = _param(True, "reflection on the wall between the sensors")
    reflectivity: float = _param(0.5, "reflection probability rho on the wall between the sensors")
    reflection_model: ReflectionModel = _param(
        "diffuse",
        "wall reflection: 'diffuse' (Lambertian, p(alpha) ~ cos(alpha) to the normal) or 'specular' (mirror)",
        choices=("diffuse", "specular"),
    )
    max_steps: int = _param(80, "maximum number of interactions per photon; photons still in flight are lost")

    def _validate(self) -> None:
        _require(self.light_yield >= 0, f"light_yield must be >= 0, got {self.light_yield}")
        _require(self.speed_of_light > 0, f"speed_of_light must be positive, got {self.speed_of_light}")
        _require(self.absorption_length > 0, f"absorption_length must be positive, got {self.absorption_length}")
        _require(self.scattering_length > 0, f"scattering_length must be positive, got {self.scattering_length}")
        _require(0 <= self.reflectivity <= 1, f"reflectivity must be in [0, 1], got {self.reflectivity}")
        _require(1 <= self.max_steps <= 10_000, f"max_steps must be in [1, 10000], got {self.max_steps}")


@dataclass(frozen=True)
class Readout(_Part):
    """Photosensor response.

    A photon reaching a sensor is detected with probability ``detection_efficiency`` (otherwise it is absorbed by the
    sensor). Its time is ``t0 + path / c_n + N(0, time_jitter^2)``. Dark noise adds ``Poisson(dark_rate *
    readout_window)`` hits per event and sensor, with times uniform in ``[0, readout_window]``. The readout window only
    sets the dark-noise exposure: signal photons are never cut.
    """

    detection_efficiency: float = _param(1.0, "probability that a photon reaching a sensor is detected, epsilon")
    time_jitter: float = _param(1.0, "Gaussian time resolution sigma_t of a detected photon (0 disables)", "ns")
    dark_rate: float = _param(0.0, "dark-noise rate per sensor (0 disables)", "Hz")
    readout_window: float = _param(100.0, "readout window [0, T] in which dark-noise hits are generated", "ns")

    @property
    def mean_noise_hits(self) -> float:
        """Mean number of dark-noise hits per event and sensor, ``dark_rate * readout_window``."""
        return self.dark_rate * self.readout_window * 1e-9

    def _validate(self) -> None:
        _require(
            0 <= self.detection_efficiency <= 1,
            f"detection_efficiency must be in [0, 1], got {self.detection_efficiency}",
        )
        _require(self.time_jitter >= 0, f"time_jitter must be >= 0, got {self.time_jitter}")
        _require(self.dark_rate >= 0, f"dark_rate must be >= 0, got {self.dark_rate}")
        _require(self.readout_window > 0, f"readout_window must be positive, got {self.readout_window}")


@dataclass(frozen=True)
class Prior(_Part):
    """Distribution of the event parameters theta = (x0, y0, E) and of the event time t0.

    The vertex is uniform in the disc ``r <= r_max`` (the same density per unit area), the energy is uniform in
    ``[energy_min, energy_max]`` and the event time uniform in ``[t0_min, t0_max]``. ``fixed_vertex`` and
    ``fixed_energy`` replace the corresponding distribution by a single value (to simulate the same theta many times).
    """

    r_max: float = _param(1.0, "maximum vertex radius (the vertex is uniform in the disc r <= r_max)", "m")
    energy_min: float = _param(0.5, "lower end of the uniform energy prior", "MeV")
    energy_max: float = _param(5.0, "upper end of the uniform energy prior", "MeV")
    t0_min: float = _param(0.0, "lower end of the uniform prior of the event time t0", "ns")
    t0_max: float = _param(10.0, "upper end of the uniform prior of the event time t0", "ns")
    fixed_vertex: tuple[float, float] | None = _param(
        None, "use this vertex (x0, y0) for every event", "m", nargs=2, metavar=("X", "Y")
    )
    fixed_energy: float | None = _param(None, "use this energy for every event", "MeV", metavar="E")

    def _validate(self) -> None:
        _require(self.r_max >= 0, f"r_max must be >= 0, got {self.r_max}")
        _require(
            0 <= self.energy_min <= self.energy_max,
            f"need 0 <= energy_min <= energy_max, got [{self.energy_min}, {self.energy_max}]",
        )
        _require(self.t0_min <= self.t0_max, f"need t0_min <= t0_max, got [{self.t0_min}, {self.t0_max}]")
        if self.fixed_energy is not None:
            _require(self.fixed_energy >= 0, f"fixed_energy must be >= 0, got {self.fixed_energy}")


@dataclass(frozen=True)
class SimConfig:
    """Complete configuration of a simulation (everything except the seed and the device).

    Parameters
    ----------
    geometry, optics, readout, prior
        The four parts; each defaults to the seminar model.

    Examples
    --------
    >>> config = SimConfig().replace(scattering=False, n_sensors=16)
    >>> config.optics.scattering, config.geometry.n_sensors
    (False, 16)
    """

    geometry: Geometry = field(default_factory=Geometry)
    optics: Optics = field(default_factory=Optics)
    readout: Readout = field(default_factory=Readout)
    prior: Prior = field(default_factory=Prior)

    SECTIONS: ClassVar[tuple[str, ...]] = ("geometry", "optics", "readout", "prior")

    def __post_init__(self) -> None:
        for name, cls in zip(self.SECTIONS, _PART_CLASSES, strict=True):
            if not isinstance(getattr(self, name), cls):
                raise TypeError(f"SimConfig.{name} must be a {cls.__name__}, got {getattr(self, name)!r}")
        radius = self.geometry.radius
        _require(
            self.prior.r_max <= radius + 1e-12,
            f"r_max = {self.prior.r_max} m exceeds the detector radius {radius} m",
        )
        if self.prior.fixed_vertex is not None:
            x, y = self.prior.fixed_vertex
            _require(
                math.hypot(x, y) <= radius + 1e-12,
                f"fixed_vertex ({x}, {y}) lies outside the detector (radius {radius} m)",
            )

    def replace(self, **overrides: Any) -> SimConfig:
        """Return a copy with some parameters changed.

        Parameters
        ----------
        **overrides
            Flat parameter names of any part (e.g. ``absorption=False``, ``n_sensors=16``, ``dark_rate=1e6``) or a
            whole part (e.g. ``readout=Readout(time_jitter=0.0)``).

        Returns
        -------
        SimConfig
            The modified copy (the original is unchanged).
        """
        parts = {name: getattr(self, name) for name in self.SECTIONS}
        changes: dict[str, dict[str, Any]] = {name: {} for name in self.SECTIONS}
        for key, value in overrides.items():
            if key in self.SECTIONS:
                parts[key] = value
            elif key in _PARAMETER_PART:
                changes[_PARAMETER_PART[key]][key] = value
            else:
                close = difflib.get_close_matches(key, _PARAMETER_PART, n=3)
                hint = f" Did you mean {' or '.join(map(repr, close))}?" if close else ""
                raise TypeError(f"unknown configuration parameter {key!r}.{hint}")
        for name, kwargs in changes.items():
            if kwargs:
                parts[name] = dataclasses.replace(parts[name], **kwargs)
        return SimConfig(**parts)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Nested dictionary of plain Python values (JSON-serialisable)."""
        out: dict[str, dict[str, Any]] = {}
        for name in self.SECTIONS:
            part = dataclasses.asdict(getattr(self, name))
            out[name] = {key: list(v) if isinstance(v, tuple) else v for key, v in part.items()}
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Mapping[str, Any]]) -> SimConfig:
        """Inverse of :meth:`to_dict`; missing parameters take their default values."""
        unknown = set(data) - set(cls.SECTIONS)
        if unknown:
            raise ValueError(f"unknown configuration sections: {sorted(unknown)}")
        parts = {}
        for name, part_cls in zip(cls.SECTIONS, _PART_CLASSES, strict=True):
            values = dict(data.get(name, {}))
            extra = set(values) - {f.name for f in dataclasses.fields(part_cls)}
            if extra:
                raise ValueError(f"unknown parameters in section {name!r}: {sorted(extra)}")
            parts[name] = part_cls(**values)
        return cls(**parts)

    def to_json(self, indent: int | None = None) -> str:
        """JSON string of :meth:`to_dict` (stored in every data file)."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> SimConfig:
        """Inverse of :meth:`to_json`."""
        return cls.from_dict(json.loads(text))

    def summary(self) -> str:
        """Short human-readable description (used by ``flashsim info``)."""
        g, o, r, p = self.geometry, self.optics, self.readout, self.prior
        absorption = f"lambda_a = {o.absorption_length:g} m" if o.absorption else "off"
        scattering = f"lambda_s = {o.scattering_length:g} m" if o.scattering else "off"
        reflection = f"rho = {o.reflectivity:g} ({o.reflection_model})" if o.reflection else "off (black wall)"
        if p.fixed_vertex is not None:
            vertex = f"fixed at ({p.fixed_vertex[0]:g}, {p.fixed_vertex[1]:g}) m"
        else:
            vertex = f"uniform in r <= {p.r_max:g} m"
        if p.fixed_energy is not None:
            energy = f"fixed at {p.fixed_energy:g} MeV"
        else:
            energy = f"uniform in [{p.energy_min:g}, {p.energy_max:g}] MeV"
        return "\n".join(
            [
                f"geometry:   R = {g.radius:g} m, {g.n_sensors} sensors of {g.sensor_width:g} m "
                f"(photocoverage {g.photocoverage:.3f})",
                f"light:      Y = {o.light_yield:g} photons/MeV ({o.photon_statistics}), "
                f"c_n = {o.speed_of_light:g} m/ns",
                f"absorption: {absorption}; scattering: {scattering}; reflection: {reflection}",
                f"readout:    efficiency {r.detection_efficiency:g}, jitter {r.time_jitter:g} ns, "
                f"dark rate {r.dark_rate:g} Hz in [0, {r.readout_window:g}] ns",
                f"prior:      vertex {vertex}; energy {energy}; t0 uniform in [{p.t0_min:g}, {p.t0_max:g}] ns",
                f"transport:  at most {o.max_steps} interactions per photon",
            ]
        )


_PART_CLASSES: tuple[type[Any], ...] = (Geometry, Optics, Readout, Prior)
_PARAMETER_PART: dict[str, str] = {}  # parameter name -> name of the part that owns it
for _part, _cls in zip(SimConfig.SECTIONS, _PART_CLASSES, strict=True):
    for _f in dataclasses.fields(_cls):
        if _f.name in _PARAMETER_PART:
            raise RuntimeError(f"duplicate parameter name {_f.name!r}")
        _PARAMETER_PART[_f.name] = _part


def config_fields() -> Iterator[tuple[str, dataclasses.Field[Any]]]:
    """Iterate over ``(part name, dataclass field)`` for every parameter, in definition order."""
    for name, cls in zip(SimConfig.SECTIONS, _PART_CLASSES, strict=True):
        for f in dataclasses.fields(cls):
            yield name, f


def _ladder(absorption: bool, scattering: bool, reflection: bool) -> SimConfig:
    return SimConfig(optics=Optics(absorption=absorption, scattering=scattering, reflection=reflection))


#: The cumulative "physics ladder": each preset switches on one more optical process.
PRESETS: dict[str, SimConfig] = {
    "straight": _ladder(absorption=False, scattering=False, reflection=False),
    "absorb": _ladder(absorption=True, scattering=False, reflection=False),
    "absorb-scatter": _ladder(absorption=True, scattering=True, reflection=False),
    "full": _ladder(absorption=True, scattering=True, reflection=True),
}


def preset(name: str = "full", **overrides: Any) -> SimConfig:
    """Return the configuration of a named preset, optionally with parameters changed on top.

    Parameters
    ----------
    name
        One of ``"straight"``, ``"absorb"``, ``"absorb-scatter"``, ``"full"`` (see :data:`PRESETS`).
    **overrides
        Parameters to change, as for :meth:`SimConfig.replace`.

    Returns
    -------
    SimConfig

    Examples
    --------
    >>> preset("straight", reflection=True).optics.reflection
    True
    """
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {', '.join(PRESETS)}")
    return PRESETS[name].replace(**overrides) if overrides else PRESETS[name]


def matching_preset(config: SimConfig) -> str | None:
    """Name of the preset equal to ``config``, or ``None`` if the configuration was modified."""
    for name, candidate in PRESETS.items():
        if candidate == config:
            return name
    return None
