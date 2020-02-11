# Salmon lice

The module represents salmon lice having a vertical swimming behaviour due to
ambient light and salt conditions. The module is used by IMR in operational
simulations of salmon lice distribution within Norwegian fjords.

The model is based on a number of scientific works on salmon lice behaviour, including
- [Author et al. (2011)](http://dx.doi.org/doicode_here)
- New salinity model: [Author et al. (2011)](http://dx.doi.org/doicode_here)

## Usage

Modify `ladim.yaml` and `particles.rls` to suit your needs.

Common changes applied to `ladim.yaml`:
- Start date of simulation (`time_control.start_time`)
- Stop date of simulation (`time_control.stop_time`)
- Forcing input file (`gridforce.input_file`)
- Vertical diffusion parameter (`ibm.vertical_mixing`)
- Horizontal diffusion parameter (`numerics.diffusion`)
- Time step length (`numerics.dt`)
- Output frequency (`output_variables.outper`)
- Salinity model (`ibm.salinity_model`)

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

## Real-world example 

An example of configuration file and release file used in a real scenario is
provided by `anne2019_ladim.yaml` and `anne2019_particles.rls`.


## History

Created by Bjørn Ådlandsvik (2018).

Refined by Anne Dagrun Sandvik and Pål Næverlid Sævik (2019) to account for
improved understanding of salmon lice freshwater avoidance behaviour.
