import mammos_units


def test_version():
    """Check that __version__ exists and is a string."""
    assert isinstance(mammos_units.__version__, str)
