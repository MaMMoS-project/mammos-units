import mammos_units as mu

def test_new_units():
    # Formula Unit
    fu = mu.f_u_
    assert fu.si == mu.dimensionless_unscaled
    assert fu == mu.Formula_Unit
    
    # Bohr magneton
    mu_B = mu.mu_B
    assert mu_B.si == mu.constants.muB
    