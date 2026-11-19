# Saithe

The module represents saithe larvae that have a preferential compass direction
upon birth. The module is based on the `larvae` module.


## Usage

Modify `ladim.yaml` and `particles.rls` to suit your needs.

The config file `ladim.yaml` contains configuration parameters. See in-file
comments for details. Compared to the `larvae` module, this module has an
additional parameter `extra_spreading` (True/False), which determines if
individual directional behaviour should be turned on.

The file `particles.rls` is a tab-delimited text file containing particle
release time and location, as well as particle attributes at the release time.
The order of the columns is given by the entry `particle_release.variables`
within `ladim.yaml`. To produce a release file, one may also use the `makrel`
tool provided with `ladim_plugins`. See comments in `release.yaml` for details.

Finally, copy `ladim.yaml` and `particles.rls` to a separate directory and
run `ladim` here.


## History

November 2026: Vertical diffusion is done by ladim, configured by `numerics.vert_mix` (requires
ladim >= 2.10.0). The option `ibm.vertical_mixing` is removed and gives an error. The diffusivity
1e-4 m2/s was fixed in the IBM, and must now be set by `numerics.vert_mix`.

Created by Pål Næverlid Sævik (2024)
