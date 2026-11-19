def check_no_ibm_vertical_mixing(config):
    """Raise an error if the removed option ``ibm.vertical_mixing`` is used

    The vertical diffusion is done by ladim (ladim >= 2.10.0), configured by
    ``numerics.vert_mix``. An option left in an old config file would otherwise
    be ignored silently.
    """
    if "vertical_mixing" in config.get("ibm", {}):
        raise ValueError("The option ibm.vertical_mixing is removed. Use "
                         "numerics.vert_mix instead (requires ladim >= 2.10.0)")
