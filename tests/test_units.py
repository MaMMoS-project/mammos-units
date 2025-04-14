import mammos_units as u
import pytest
import numpy as np
from astropy.units import isclose


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

    # Test with different volume units
    vol2 = 1e-28 * u.m**3
    eq2 = u.moment_induction(vol2)
    assert eq2 is not None


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
    assert np.isclose(polarisation.value, expected.value, atol=1e-12)

    # Test reverse conversion: Tesla → mu_B/counting_unit
    polarisation = 1e-3 * u.T
    moment = polarisation.to(u.mu_B / counting_unit, equivalencies=eq)
    expected = 1e-3 * vol_m3 / (u.constants.mu0 * u.constants.muB)
    assert np.isclose(moment.value, expected.value, atol=1e-12)

    # Test forward and reverse conversion: mu_B/counting_unit → Tesla → mu_B/counting_unit
    moment = 2.5 * u.mu_B / counting_unit
    polarisation = moment.to(u.T, equivalencies=eq)
    reversed_moment = polarisation.to(u.mu_B / counting_unit, equivalencies=eq)
    assert np.isclose(reversed_moment.value, moment.value, atol=1e-12)


def test_unit_latex_format():
    """Test that units have the correct LaTeX formatting"""
    assert u.f_u._format["latex"] == r"\mathrm{f.u.}"
    assert u.mu_B._format["latex"] == r"\mu_B"
    assert u.atom._format["latex"] == r"\mathrm{atom}"


def test_moment_induction_different_volume_units():
    """Test moment_induction with different volume units"""
    # Using cubic nanometers
    vol_nm = 1.0 * u.nm**3
    eq_nm = u.moment_induction(vol_nm)

    # Using cubic angstroms
    vol_ang = 1000.0 * u.angstrom**3  # 1 nm³ = 1000 Å³
    eq_ang = u.moment_induction(vol_ang)

    # Test that they give the same conversion
    moment = 1.0 * u.mu_B / u.f_u
    b_field_nm = moment.to(u.T, equivalencies=eq_nm)
    b_field_ang = moment.to(u.T, equivalencies=eq_ang)

    assert isclose(b_field_nm, b_field_ang)


def test_array_conversion():
    """Test conversion of arrays of values"""
    vol = 100 * u.angstrom**3
    eq = u.moment_induction(vol)

    # Array of moments
    moments = np.array([1.0, 2.0, 3.0]) * u.mu_B / u.f_u
    b_fields = moments.to(u.T, equivalencies=eq)

    assert len(b_fields) == 3
    assert np.isclose(b_fields[1], 2*b_fields[0])
    assert np.isclose(b_fields[2], 3*b_fields[0])


@pytest.mark.parametrize(
    "invalid_quantity",
    [
        100 * u.kg,  # Mass unit
        5 * u.m,  # Length unit
        42 * u.dimensionless_unscaled,  # Dimensionless quantity
    ],
)
def test_moment_induction_error_non_volume(invalid_quantity):
    with pytest.raises(u.UnitConversionError):
        u.moment_induction(invalid_quantity)


@pytest.mark.parametrize(
    "invalid_argument",
    [
        100,
        "k",
    ],
)
def test_moment_induction_error_invalid_argument(invalid_argument):
    with pytest.raises(TypeError, match="Volume must be a Quantity"):
        u.moment_induction(invalid_argument)


def test_moment_induction_error_nonpositive_volume():

    neg_vol = -100 * u.angstrom**3

    with pytest.raises(ValueError, match="Volume must be positive"):
        u.moment_induction(neg_vol)

    zero_vol = 0 * u.angstrom**3

    with pytest.raises(ValueError, match="Volume must be positive"):
        eq = u.moment_induction(zero_vol)


def test_moment_induction_error_wrong_unit_conversion():
    """Test error when trying to convert incompatible units with the equivalency"""
    vol = 100 * u.angstrom**3
    eq = u.moment_induction(vol)

    length = 10 * u.m
    with pytest.raises(u.UnitConversionError):
        length.to(u.T, equivalencies=eq)

    field = 0.5 * u.T
    with pytest.raises(u.UnitConversionError):
        field.to(u.m, equivalencies=eq)

    magnetisation = 1e5 * u.A / u.m
    with pytest.raises(u.UnitConversionError):
        magnetisation.to(u.mu_B / u.atom, equivalencies=eq)

    # Chain conversion
    magnetisation.to(u.T, equivalencies=u.magnetic_flux_field()).to(
        u.mu_B / u.atom, equivalencies=eq
    )
