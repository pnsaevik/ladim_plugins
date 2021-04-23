import numpy as np
from ..utils import light, density, viscosity


def growth_cod_larvae(temp, weight, dt):
    """
    Incremental growth of NA cod larvae as reported by Folkvord (2005):
    "Comparison of size-at-age of larval Atlantic cod (Gadus morhua) from different
    populations based on size- and temperature-dependent growth models". Canadian Journal
    of Fisheries and Aquatic Sciences 62(5). https://doi.org/10.1139/f05-008

    :param temp: Temperature [degrees Celcius]
    :param weight: Larvae dry weight [mg]
    :param dt: Growth time [s]
    :return: Dry weight increase [mg]
    """

    w = np.log(weight)
    sec2day = 1 / 86400
    GR_percent = (1.08 + temp * (1.79 + w * (- 0.074 + w * (-0.0965 + w * 0.0112))))
    GR = np.log(1 + 0.01 * GR_percent)
    return (np.exp(GR * sec2day * dt) - 1) * weight


def weight_to_length(weight):
    """
    Compute larvae length from dry weight using equation (6) of Folkvord (2005):
    "Comparison of size-at-age of larval Atlantic cod (Gadus morhua) from different
    populations based on size- and temperature-dependent growth models". Canadian Journal
    of Fisheries and Aquatic Sciences 62(5). https://doi.org/10.1139/f05-008

    :param weight: Dry weight [mg]
    :return: Body length of larvae [mm]
    """

    w = np.log(weight)
    return np.exp(2.296 + w * (0.277 - w * 0.005128))


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
