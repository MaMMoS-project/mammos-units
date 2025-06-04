"""Quantities (values with units).

mammos_units is an extension of astropy.units. Please refer to
https://docs.astropy.org/en/stable/units/ref_api.html.

The following additional units are defined in this package:

- formula_unit`
- `mu_B`
- `atom`
"""

from __future__ import annotations

import importlib.metadata
from typing import TYPE_CHECKING

import astropy.constants as constants
from astropy.units import *

if TYPE_CHECKING:
    import astropy.units

__version__ = importlib.metadata.version(__package__)

def_unit(
    ["f_u", "formula_unit"], format={"latex": r"\mathrm{f.u.}"}, namespace=globals()
)
def_unit("mu_B", constants.muB, format={"latex": r"\mu_B"}, namespace=globals())
def_unit("atom", format={"latex": r"\mathrm{atom}"}, namespace=globals())


def moment_induction(volume: astropy.units.Quantity) -> astropy.units.Equivalency:
    r"""Equivalency for magnetic moment per formula unit and magnetic induction.

    Equivalency for converting between magnetic moment per counting unit
    (either formula unit or per atom) and magnetic induction (Tesla).

    This equivalency handles the conversion between magnetic moment units
    (μ_B/f.u. or μ_B/atom) and magnetic induction (Tesla) based on a given volume.

    The conversion is based on the relation:

    .. math::

        B = \frac{\mu_0 \cdot m}{V}

    Where:
    - B is the magnetic induction in Tesla
    - μ_0 is the vacuum permeability
    - m is the magnetic moment in Bohr magnetons per counting unit
    - V is the volume per counting unit

    Args:
        volume: The volume over which the magnetic moment is distributed.
            This can be in any unit of volume that can be converted to m³.

    Returns:
        The equivalency object that can be passed to the `equivalencies`
        argument of `astropy.units.Quantity.to()`.

    Raises:
        ValueError: If the volume is negative.
        TypeError: If the input is not an astropy Quantity object.

    Examples:
        >>> import mammos_units as u
        >>> vol = 4 * u.angstrom**3
        >>> eq = u.moment_induction(vol)
        >>> moment = 2.5 * u.mu_B / u.f_u
        >>> b_field = moment.to(u.T, equivalencies=eq)
        >>> b_field
        <Quantity 7.28379049 T>

        >>> b_field.to(u.mu_B / u.f_u, equivalencies=eq)
        <Quantity 2.5 mu_B / f_u>

    """
    if not isinstance(volume, Quantity):
        raise TypeError("Volume must be a Quantity")

    volume = volume.to(m**3)

    # Check if volume is negative
    if volume.value <= 0:
        raise ValueError("Volume must be positive")

    return Equivalency(
        [
            (
                mu_B / f_u,
                T,
                lambda x: x * constants.muB * constants.mu0 / volume,
                lambda x: x * volume / (constants.mu0 * constants.muB),
            ),
            (
                mu_B / atom,
                T,
                lambda x: x * constants.muB * constants.mu0 / volume,
                lambda x: x * volume / (constants.mu0 * constants.muB),
            ),
        ],
        "moment_induction",
    )
