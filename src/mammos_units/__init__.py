r"""Quantities (values with units).

Historically, :mod:`mammos_units` was implemented as a thin extension of
``astropy.units``. That original design was very convenient because Astropy
already provided:

- a rich unit registry
- quantities with NumPy integration
- magnetic-field equivalencies such as ``magnetic_flux_field``
- a familiar ``Quantity.to(..., equivalencies=...)`` API

The package now uses Pint as its runtime backend to keep dependencies lighter.
However, the public API is still intentionally shaped to look much closer to
the earlier Astropy-based interface than to raw Pint. In practice, that means
this module does not expose Pint directly and unmodified; instead it provides a
small compatibility layer that recreates the parts of the former Astropy
surface that the package, tests, and notebooks rely on.

The main compatibility decisions are:

- expose units at module level, so users can continue to write ``u.m``,
  ``u.T``, ``u.Oe``, ``u.mu_B`` and similar expressions
- provide a custom :class:`MammosQuantity` with Astropy-like convenience
  attributes such as ``.value``, ``.unit``, ``.si``, and ``.cgs``
- preserve an Astropy-like equivalency workflow via
  ``Quantity.to(..., equivalencies=...)`` and
  :func:`set_enabled_equivalencies`
- keep magnetic conversions explicit where the old Astropy API required an
  equivalency, even if Pint's native electromagnetic definitions would allow a
  direct conversion

This file therefore serves two purposes:

1. implement the Pint-backed behavior
2. document which parts exist primarily for backward compatibility with the
   Astropy era of the package
"""

from __future__ import annotations

import importlib.metadata
import math
from collections.abc import Iterable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pint

__version__ = importlib.metadata.version(__package__)


class UnitConversionError(pint.errors.DimensionalityError):
    """Compatibility error type matching Astropy's name.

    Pint raises :class:`pint.errors.DimensionalityError` for incompatible unit
    conversions. The previous Astropy-based implementation exposed
    ``UnitConversionError``, and the notebooks and tests refer to that name
    explicitly. Subclassing here lets us preserve the old public exception name
    while still fitting naturally into Pint's error model.
    """


_SPECIAL_UNIT_REPLACEMENTS = {
    "gauss": (1e-4, "tesla"),
    "oersted": (1000 / (4 * math.pi), "ampere / meter"),
    "maxwell": (1e-8, "weber"),
}


class MammosQuantity(pint.Quantity):
    """Pint quantity with a small Astropy-compatible surface.

    Pint quantities already handle most numerical work well, but their public
    API is not a drop-in match for Astropy quantities. This subclass restores a
    few high-value conveniences that existing mammos-users expect:

    - ``.value`` and ``.unit`` aliases
    - ``.si`` and ``.cgs`` properties
    - ``.to(..., equivalencies=...)`` support
    - a more Astropy-like ``repr`` for notebook and doctest output

    The goal is not perfect emulation of Astropy; it is stable compatibility
    for the subset of behavior that this package intentionally exposes.
    """

    @property
    def value(self):
        return self.magnitude

    @property
    def unit(self):
        return self.units

    @property
    def si(self):
        return _to_system(self, "mks")

    @property
    def cgs(self):
        return _to_system(self, "cgs")

    def to(self, other=None, *contexts, equivalencies=None, **ctx_kwargs):
        """Convert to a different unit, optionally using mammos equivalencies.

        The conversion order is deliberate:

        1. try Pint's direct conversion machinery
        2. try the local "special unit" bridge for cgs-style magnetic units
        3. try explicit mammos equivalencies

        This keeps ordinary Pint behavior intact where possible, but still
        preserves the Astropy-era semantics where mammos historically relied on
        explicit equivalencies.
        """
        target = _as_unit(other)
        active = equivalencies if equivalencies is not None else _enabled_equivalencies
        # In the Astropy version, induction/field-strength conversions such as
        # T <-> A/m or G <-> Oe were not freely available: users had to opt in
        # via magnetic_flux_field(). Pint's own electromagnetic definitions can
        # make some of these paths appear directly convertible, so we
        # explicitly guard this boundary to preserve the older API contract.
        if active is None and _crosses_field_strength_and_induction(self, target):
            raise UnitConversionError(self.units, target)
        try:
            return super().to(target, *contexts, **ctx_kwargs)
        except pint.errors.DimensionalityError:
            try:
                return _convert_via_special_units(self, target)
            except UnitConversionError:
                for equivalency in _iter_equivalencies(active):
                    try:
                        return equivalency.convert(self, target)
                    except UnitConversionError:
                        continue
                raise UnitConversionError(self.units, target) from None

    def __repr__(self):
        return f"<Quantity {self.value} {self.units:~}>"

    def _repr_latex_(self):
        return f"${self:~L}$"


class MammosRegistry(pint.UnitRegistry):
    """Registry whose default quantity type is :class:`MammosQuantity`."""

    Quantity = MammosQuantity


ureg = MammosRegistry()
Quantity = ureg.Quantity
EquivalencyName = list[str]


def _setup_registry() -> None:
    """Define mammos-specific units and aliases on the Pint registry.

    Most SI units come directly from Pint. We only define the additional names
    that were historically provided by mammos or commonly used in the example
    notebooks.
    """
    ureg.define("@alias oersted = Oe")
    ureg.define("@alias gauss = G = Gauss")
    ureg.define("@alias maxwell = Mx")
    ureg.define("@alias angstrom = Angstrom")
    ureg.define("f_u = [] = formula_unit")
    ureg.define("atom = []")
    ureg.define("mu_B = 9.2740100783e-24 * joule / tesla")


def _tag_unit(unit: pint.Unit, latex: str | None = None) -> pint.Unit:
    """Attach small bits of Astropy-like metadata to a Pint unit.

    Astropy units expose formatting metadata such as ``_format["latex"]`` and
    the existing tests inspect that attribute directly. Pint units are flexible
    enough to carry custom attributes, so we use that escape hatch here instead
    of re-wrapping every unit object.
    """
    if latex is not None:
        unit._format = {"latex": latex}
    return unit


_setup_registry()

f_u = _tag_unit(ureg.f_u, r"\mathrm{f.u.}")
formula_unit = f_u
mu_B = _tag_unit(ureg.mu_B, r"\mu_B")
atom = _tag_unit(ureg.atom, r"\mathrm{atom}")
G = ureg.G
Gauss = ureg.Gauss
Oe = ureg.Oe
Mx = ureg.Mx
Angstrom = ureg.Angstrom
dimensionless_unscaled = ureg.dimensionless
Equivalency = None  # assigned after class definition


class _UnitContext:
    """State token mirroring Astropy's ``set_enabled_equivalencies`` API.

    The key compatibility detail is that Astropy's helper both mutates global
    state immediately and returns a context-manager token that can later restore
    the previous state. We follow that same pattern so that both of these forms
    continue to work:

    - ``u.set_enabled_equivalencies(eq)``
    - ``with u.set_enabled_equivalencies(eq): ...``
    """

    def __init__(self, new_equivalencies):
        global _enabled_equivalencies
        self._new_equivalencies = new_equivalencies
        self._old_equivalencies = _enabled_equivalencies
        _enabled_equivalencies = self._new_equivalencies

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        global _enabled_equivalencies
        _enabled_equivalencies = self._old_equivalencies
        return False

    def __repr__(self):
        return f"<astropy.units.core._UnitContext at {hex(id(self))}>"


@dataclass(frozen=True)
class MammosEquivalency:
    """Minimal, explicit replacement for Astropy's richer equivalency objects.

    Astropy models equivalencies as structured conversion rules understood by
    its quantity system. Pint has a different concept, so mammos keeps a small
    dispatch object of its own and applies it from :meth:`MammosQuantity.to`.
    """

    kind: str
    params: dict[str, Any]

    @property
    def name(self) -> EquivalencyName:
        return [self.kind]

    def convert(self, quantity: MammosQuantity, target: pint.Unit) -> MammosQuantity:
        if self.kind == "magnetic_flux_field":
            return _convert_magnetic_flux_field(quantity, target, self.params["mu_r"])
        if self.kind == "moment_induction":
            return _convert_moment_induction(quantity, target, self.params["volume"])
        if self.kind == "temperature_energy":
            return _convert_temperature_energy(quantity, target)
        raise UnitConversionError(quantity.units, target)


Equivalency = MammosEquivalency

constants = SimpleNamespace(
    muB=9.2740100783e-24 * ureg.joule / ureg.tesla,
    mu0=4 * math.pi * 1e-7 * ureg.newton / ureg.ampere**2,
    k_B=1.380649e-23 * ureg.joule / ureg.kelvin,
)

_enabled_equivalencies = None


def _iter_equivalencies(equivalencies) -> Iterable[MammosEquivalency]:
    """Normalize ``None``, a single equivalency, or an iterable of them."""
    if equivalencies is None:
        return ()
    if isinstance(equivalencies, MammosEquivalency):
        return (equivalencies,)
    return tuple(equivalencies)


def _as_unit(value) -> pint.Unit:
    """Accept either a Pint unit object or a unit string.

    The notebooks use both ``quantity.to("Oe")`` and ``quantity.to(u.Oe)``.
    Normalizing at the boundary keeps the rest of the implementation simpler.
    """
    if isinstance(value, str):
        return ureg.parse_units(value)
    return value


def _to_system(quantity: MammosQuantity, system: str) -> MammosQuantity:
    """Convert to base units of a named Pint system.

    This backs the Astropy-like ``.si`` and ``.cgs`` properties.
    """
    factor, unit = ureg.get_base_units(quantity.units, system=system)
    return Quantity(quantity.magnitude * factor, unit)


def _convert_via_special_units(
    quantity: MammosQuantity, target: pint.Unit
) -> MammosQuantity:
    """Bridge magnetic cgs names through explicit SI surrogate units.

    Pint ships electromagnetic cgs units such as gauss, oersted, and maxwell,
    but their dimensional treatment does not line up with the Astropy behavior
    mammos historically relied on. For the mammos API we want:

    - ``G`` to behave like ``1e-4 T``
    - ``Oe`` to behave like ``1000 / (4*pi) A/m``
    - ``Mx`` to behave like ``1e-8 Wb``

    To achieve that, we temporarily replace those units with SI surrogates,
    ask Pint to do the ordinary conversion, and then rebuild the target unit.
    """
    if not (_contains_special_unit(quantity.units) or _contains_special_unit(target)):
        raise UnitConversionError(quantity.units, target)

    source_factor, source_unit = _surrogate_unit(quantity.units)
    target_factor, target_unit = _surrogate_unit(target)
    surrogate = Quantity(quantity.magnitude * source_factor, source_unit)
    try:
        converted = super(MammosQuantity, surrogate).to(target_unit)
    except pint.errors.DimensionalityError as exc:
        raise UnitConversionError(quantity.units, target) from exc
    return Quantity(converted.magnitude / target_factor, target)


def _contains_special_unit(unit: pint.Unit) -> bool:
    return any(name in _SPECIAL_UNIT_REPLACEMENTS for name in unit._units)


def _surrogate_unit(unit: pint.Unit) -> tuple[float, pint.Unit]:
    """Return a multiplicative SI surrogate for units with special handling."""
    factor = 1.0
    surrogate = ureg.dimensionless
    for name, power in unit._units.items():
        if name in _SPECIAL_UNIT_REPLACEMENTS:
            base_factor, base_name = _SPECIAL_UNIT_REPLACEMENTS[name]
            factor *= base_factor**power
            surrogate *= ureg.parse_units(base_name) ** power
        else:
            surrogate *= ureg.parse_units(name) ** power
    return factor, surrogate


def _convert_magnetic_flux_field(
    quantity: MammosQuantity, target: pint.Unit, mu_r: float
) -> MammosQuantity:
    """Implement the old Astropy magnetic-flux equivalency.

    This covers conversion between magnetic field strength ``H`` and magnetic
    induction ``B`` using ``B = mu0 * mu_r * H``.
    """
    source_h = _try_convert(quantity, ureg.ampere / ureg.meter)
    if source_h is not None:
        target_b = _try_convert(Quantity(1, target), ureg.tesla)
        if target_b is not None:
            induction = source_h * constants.mu0 * mu_r
            return induction.to(target)

    source_b = _try_convert(quantity, ureg.tesla)
    if source_b is not None:
        target_h = _try_convert(Quantity(1, target), ureg.ampere / ureg.meter)
        if target_h is not None:
            field = source_b / (constants.mu0 * mu_r)
            return field.to(target)

    raise UnitConversionError(quantity.units, target)


def _convert_moment_induction(
    quantity: MammosQuantity, target: pint.Unit, volume: MammosQuantity
) -> MammosQuantity:
    """Convert between ``mu_B`` per counting unit and induction.

    Astropy equivalencies operate on magnitudes plus explicitly attached source
    and target units. Here we reproduce the same effect directly with Pint
    quantities while keeping the public API identical.
    """
    source_b = _try_convert(quantity, ureg.tesla)
    if source_b is not None:
        for counting_unit in (f_u, atom):
            candidate = mu_B / counting_unit
            if _try_convert(Quantity(1, target), candidate) is not None:
                magnitude = (
                    (source_b * volume / (constants.mu0 * constants.muB))
                    .to(ureg.dimensionless)
                    .magnitude
                )
                return Quantity(magnitude, target)

    for counting_unit in (f_u, atom):
        candidate = mu_B / counting_unit
        source_m = _try_convert(quantity, candidate)
        target_is_induction = _try_convert(Quantity(1, target), ureg.tesla) is not None
        if source_m is not None and target_is_induction:
            induction = source_m.magnitude * constants.muB * constants.mu0 / volume
            return induction.to(target)

    raise UnitConversionError(quantity.units, target)


def _convert_temperature_energy(
    quantity: MammosQuantity, target: pint.Unit
) -> MammosQuantity:
    """Convert between thermal energy and temperature via Boltzmann's constant."""
    source_temperature = _try_convert(quantity, ureg.kelvin)
    target_is_energy = _try_convert(Quantity(1, target), ureg.joule) is not None
    if source_temperature is not None and target_is_energy:
        return (source_temperature * constants.k_B).to(target)

    source_energy = _try_convert(quantity, ureg.joule)
    target_is_temperature = _try_convert(Quantity(1, target), ureg.kelvin) is not None
    if source_energy is not None and target_is_temperature:
        return (source_energy / constants.k_B).to(target)

    raise UnitConversionError(quantity.units, target)


def _try_convert(quantity: MammosQuantity, target: pint.Unit) -> MammosQuantity | None:
    """Attempt conversion and return ``None`` instead of raising on failure."""
    try:
        return _plain_to(quantity, target)
    except (UnitConversionError, pint.errors.DimensionalityError):
        return None


def _plain_to(quantity: MammosQuantity, target: pint.Unit) -> MammosQuantity:
    """Perform a conversion without consulting mammos equivalencies.

    This helper is important for internal probing. It lets the implementation
    ask "is this quantity compatible with that unit?" without accidentally
    recursing back through global or explicit equivalency handling.
    """
    try:
        return super(MammosQuantity, quantity).to(target)
    except pint.errors.DimensionalityError:
        return _convert_via_special_units(quantity, target)


def _crosses_field_strength_and_induction(
    quantity: MammosQuantity, target: pint.Unit
) -> bool:
    """Check whether a conversion crosses the H/B boundary.

    That boundary is where we intentionally preserve Astropy's requirement for
    an explicit magnetic equivalency.
    """
    return (_is_field_strength(quantity) and _is_induction_unit(target)) or (
        _is_induction(quantity) and _is_field_strength_unit(target)
    )


def _is_field_strength(quantity: MammosQuantity) -> bool:
    return _try_convert(quantity, ureg.ampere / ureg.meter) is not None


def _is_induction(quantity: MammosQuantity) -> bool:
    return _try_convert(quantity, ureg.tesla) is not None


def _is_field_strength_unit(unit: pint.Unit) -> bool:
    return _try_convert(Quantity(1, unit), ureg.ampere / ureg.meter) is not None


def _is_induction_unit(unit: pint.Unit) -> bool:
    return _try_convert(Quantity(1, unit), ureg.tesla) is not None


def set_enabled_equivalencies(equivalencies):
    """Enable conversions globally until reset or context-manager exit.

    This mirrors Astropy's global-equivalency helper because the notebooks and
    earlier mammos versions use it directly.
    """
    return _UnitContext(equivalencies)


def magnetic_flux_field(mu_r: float = 1.0) -> MammosEquivalency:
    """Equivalency between magnetic field strength and induction.

    Args:
        mu_r: Relative permeability used in ``B = mu0 * mu_r * H``.
    """
    return MammosEquivalency("magnetic_flux_field", {"mu_r": mu_r})


def temperature_energy() -> MammosEquivalency:
    """Equivalency between temperature and thermal energy.

    This is the Pint-backed replacement for the Astropy helper used in the
    example notebook to convert Kelvin to Joule via Boltzmann's constant.
    """
    return MammosEquivalency("temperature_energy", {})


def moment_induction(volume: MammosQuantity) -> MammosEquivalency:
    r"""Equivalency for magnetic moment per formula unit and magnetic induction.

    This helper is one of the main reasons the compatibility layer exists at
    all. Astropy's built-in equivalency mechanism made it natural to say:

    ``moment.to(u.T, equivalencies=u.moment_induction(volume))``

    Pint does not provide that same API directly, so mammos reintroduces it
    here in a deliberately small, explicit form.

    Args:
        volume: The volume over which the magnetic moment is distributed.

    Returns:
        A mammos equivalency object for use with :meth:`Quantity.to`.

    Examples:
        >>> import mammos_units as u
        >>> vol = 4 * u.angstrom**3
        >>> eq = u.moment_induction(vol)
        >>> moment = 2.5 * u.mu_B / u.f_u
        >>> b_field = moment.to(u.T, equivalencies=eq)
        >>> round(b_field.value, 8)
        7.28379048
    """
    if not isinstance(volume, Quantity):
        raise TypeError("Volume must be a Quantity")

    volume = volume.to(ureg.meter**3)
    if volume.value <= 0:
        raise ValueError("Volume must be positive")

    return MammosEquivalency("moment_induction", {"volume": volume})


def isclose(a: MammosQuantity, b: MammosQuantity, rtol=1e-9, atol=None):
    """Quantity-aware ``numpy.isclose`` helper.

    The tests previously imported ``isclose`` from Astropy. Providing a local
    helper keeps callers from needing to know whether mammos is backed by
    Astropy or Pint.
    """
    converted = b.to(a.units)
    atol_value = 0.0
    if atol is not None:
        atol_value = atol.to(a.units).value if hasattr(atol, "to") else atol
    return np.isclose(a.value, converted.value, rtol=rtol, atol=atol_value)


def __getattr__(name: str):
    """Fallback to the underlying Pint registry for standard unit names."""
    try:
        return getattr(ureg, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc


__all__ = [
    "Angstrom",
    "Equivalency",
    "G",
    "Gauss",
    "Mx",
    "Oe",
    "Quantity",
    "UnitConversionError",
    "atom",
    "constants",
    "dimensionless_unscaled",
    "f_u",
    "formula_unit",
    "isclose",
    "magnetic_flux_field",
    "moment_induction",
    "mu_B",
    "set_enabled_equivalencies",
    "temperature_energy",
    "ureg",
]
