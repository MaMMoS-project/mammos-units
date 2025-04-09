from astropy.units import *
import astropy.constants as constants

def_unit(['f_u', 'formula_unit'], format={'latex': r'\mathrm{f.u.}'}, namespace=globals())
def_unit('mu_B', constants.muB, format={'latex': r'\mu_B'}, namespace=globals())
def_unit('atom', format={'latex': r'\mathrm{atom}'}, namespace=globals())

def moment_induction(volume):
    volume = volume.to(m**3)
    return Equivalency(
        [(mu_B/f_u, T, lambda x: x * constants.muB * constants.mu0 / volume, lambda x: x * volume / (constants.mu0 * constants.muB)),
         (mu_B/atom, T, lambda x: x * constants.muB * constants.mu0 / volume, lambda x: x * volume / (constants.mu0 * constants.muB))],
        "moment_induction",
    )
