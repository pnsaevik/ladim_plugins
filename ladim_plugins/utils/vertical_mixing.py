REMOVED_OPTIONS = ("vertical_mixing", "vertdiff_dt", "vertdiff_dz", "vertdiff_max")


def check_no_ibm_vertical_mixing(config):
    """Raise an error if a removed option for vertical diffusion is used

    The vertical diffusion is done by ladim (ladim >= 2.10.0), configured by
    ``numerics.vert_mix``. Options left in an old config file would otherwise
    be ignored silently.
    """
    used = [k for k in REMOVED_OPTIONS if k in config.get("ibm", {})]
    if used:
        raise ValueError(f"The options {', '.join('ibm.' + k for k in used)} are "
                         f"removed. Use numerics.vert_mix instead (requires "
                         f"ladim >= 2.10.0)")
