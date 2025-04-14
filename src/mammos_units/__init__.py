from astropy.units import *
import astropy.constants as constants

def_unit(['f_u', 'formula_unit'], format={'latex': r'\mathrm{f.u.}'}, namespace=globals())
def_unit('mu_B', constants.muB, format={'latex': r'\mu_B'}, namespace=globals())
def_unit('atom', format={'latex': r'\mathrm{atom}'}, namespace=globals())

def moment_induction(volume):
    """
    Equivalency for converting between magnetic moment per counting unit
    (either formula unit or per atom) and magnetic induction (Tesla).
    
    This equivalency handles the conversion between magnetic moment units
    (μ_B/f.u. or μ_B/atom) and magnetic induction (Tesla) based on a given volume.
    
    The conversion is based on the relation:
    
    .. math::
        B = \\frac{\\mu_0 \\cdot m}{V}
    
    Where:
    - B is the magnetic induction in Tesla
    - μ_0 is the vacuum permeability
    - m is the magnetic moment in Bohr magnetons per counting unit
    - V is the volume per counting unit
    
    Parameters
    ----------
    volume : `~astropy.units.Quantity`
        The volume over which the magnetic moment is distributed.
        This can be in any unit of volume that can be converted to m³.
Must be a positive value.
    
    Returns
    -------
    equivalency : `~astropy.units.Equivalency`
        The equivalency object that can be passed to the `equivalencies` 
        argument of `astropy.units.Quantity.to()`.
    
    Examples
    --------
    >>> import mammos_units as u
    >>> vol = 4 * u.angstrom**3
    >>> eq = u.moment_induction(vol)
    >>> moment = 2.5 * u.mu_B / u.f_u
    >>> b_field = moment.to(u.T, equivalencies=eq)
    >>> b_field
    <Quantity 7.28379049 T>
    
    >>> b_field.to(u.mu_B / u.f_u, equivalencies=eq)
    <Quantity 2.5 mu_B / f_u>

    Raises
    ------
    ValueError
        If the volume is negative.
    TypeError
        If the input is not an astropy Quantity object.
    """
    if not isinstance(volume, Quantity):
        raise TypeError("Volume must be a Quantity")
    
    volume = volume.to(m**3)
            
    # Check if volume is negative
    if volume.value <= 0:
        raise ValueError("Volume must be positive")
        
    return Equivalency(
        [(mu_B/f_u, T, lambda x: x * constants.muB * constants.mu0 / volume, lambda x: x * volume / (constants.mu0 * constants.muB)),
         (mu_B/atom, T, lambda x: x * constants.muB * constants.mu0 / volume, lambda x: x * volume / (constants.mu0 * constants.muB))],
        "moment_induction",
    )
