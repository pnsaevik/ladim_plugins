import numpy as np
from ..utils import light, density, viscosity


def sinkvel_egg(mu_w, dens_w, dens_egg, diam_egg):
    # Calculate maximum diameter in Stokes formula
    dens_diff = dens_w - dens_egg
    dmax = (
        (9.0 * mu_w * mu_w) / (1025.0 * 9.81 * np.abs(dens_diff))
    ) ** (1 / 3)

    small_W = -(1 / 18) * (1 / mu_w) * 9.81 * (diam_egg ** 2) * dens_diff
    large_W = (
        -0.08825 * (diam_egg - 0.4 * dmax) * np.abs(dens_diff) ** (2 / 3)
        * mu_w ** (-1 / 3) * np.sign(dens_diff)
    )

    return np.where(diam_egg <= dmax, small_W, large_W)


def cod_egg_bouy(n):
    """
    Cod egg buoyancy according to Stenevik, E. K., S. Sundby, and A. L. Agnalt (2008).
    "Buoyancy and vertical distribution of Norwegian coastal cod (Gadus morhua) eggs from
    different areas along the coast". ICES (International Council for the Exploration of
    the Sea) Journal of Marine Science 65:1198–1202.
    """
    return 32.41 + 0.69 * np.random.randn(n)
