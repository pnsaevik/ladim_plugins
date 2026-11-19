import pytest

from ladim_plugins.utils.vertical_mixing import check_no_ibm_vertical_mixing


def test_old_vertical_mixing_option_is_rejected():
    check_no_ibm_vertical_mixing(dict(ibm=dict(speed=1)))
    check_no_ibm_vertical_mixing(dict())
    with pytest.raises(ValueError, match="numerics.vert_mix"):
        check_no_ibm_vertical_mixing(dict(ibm=dict(vertical_mixing=0.001)))
