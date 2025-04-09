import mammos_units as u
import pytest


def test_new_units():
    # Formula Unit
    assert hasattr(u, "f_u")
    assert u.f_u == u.formula_unit

    # Bohr Magneton
    assert hasattr(u, "mu_B")
    assert u.mu_B == u.constants.muB

    # Atom
    assert hasattr(u, "atom")


def test_equivalency_creation():
    """Test creating equivalency objects with different volumes"""
    vol1 = 100 * u.angstrom**3

    eq1 = u.moment_induction(vol1)

    assert eq1 is not None
    assert eq1.name == ["moment_induction"]


@pytest.mark.parametrize("counting_unit", [u.f_u, u.atom])
def test_moment_induction_equivalency(counting_unit):
    """Test conversion between mu_B/f_u and Tesla"""
    vol = 100 * u.angstrom**3
    vol_m3 = vol.to(u.m**3)
    eq = u.moment_induction(vol)

    # Test forward conversion: mu_B/counting_unit → Tesla
    moment = 2.5 * u.mu_B / counting_unit
    polarisation = moment.to(u.T, equivalencies=eq)
    expected = 2.5 * u.constants.muB * u.constants.mu0 / vol_m3
    assert abs(polarisation.value - expected.value) < 1e-12

    # Test reverse conversion: Tesla → mu_B/counting_unit
    polarisation = 1e-3 * u.T
    moment = polarisation.to(u.mu_B / counting_unit, equivalencies=eq)
    expected = 1e-3 * vol_m3 / (u.constants.mu0 * u.constants.muB)
    assert abs(moment.value - expected.value) < 1e-12

    # Test forward and reverse conversion: mu_B/counting_unit → Tesla → mu_B/counting_unit
    moment = 2.5 * u.mu_B / counting_unit
    polarisation = moment.to(u.T, equivalencies=eq)
    reversed_moment = polarisation.to(u.mu_B / counting_unit, equivalencies=eq)
    assert abs(reversed_moment.value - moment.value) < 1e-12
