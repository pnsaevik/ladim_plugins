# Salmon lice

The module represents salmon lice having a vertical swimming behaviour, and
matures at individual rates due to ambient temperature conditions. 

The model is based on a number of scientific works on salmon lice behaviour.
Vertical migration is based on [Myksvoll et al. (2011)](http://dx.doi.org/10.1080/19425120.2011.595258).
Maturation is based on [Myksvoll et al. (2011)](http://dx.doi.org/10.1080/19425120.2011.595258).

## Usage

Modify `ladim.yaml` and `particles.rls` to suit your needs.

Common changes applied to `ladim.yaml`:
- Start date of simulation (`time_control.start_time`)
- Stop date of simulation (`time_control.stop_time`)
- Forcing input file (`gridforce.input_file`)
- Vertical diffusion parameter (`ibm.vertical_mixing`)
- Horizontal diffusion parameter (`numerics.diffusion`)
- Egg diameter (`ibm.egg_diam`)
- Time step length (`numerics.dt`)
- Output frequency (`output_variables.outper`)

The file `particles.rls` is a tab-delimited text file containing particle
release time and location, as well as particle attributes at the release time.
The order of the columns is given by the entry `particle_release.variables`
within `ladim.yaml`.

Finally, copy `ladim.yaml` and `particles.rls` to a separate directory and
run `ladim` here.


## Output

The simulation result is stored in a file specified by the `files.output_file`
entry in `ladim.yaml`. The output variables are specified by the
`output_variables` entries. 

## SVIM example

An example of configuration file and release file used in a real scenario is
provided by `svim_ladim.yaml` and `svim_particles.rls`. The code used to make
the release file is provided by `svim_make_release.m`.  
