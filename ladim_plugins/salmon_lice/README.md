# Salmon lice

The module represents salmon lice having a vertical swimming behaviour due to
ambient light and salinity conditions. The module is used by IMR in operational
simulations of salmon lice distribution within Norwegian fjords.

The model is based on a number of scientific works on salmon lice behaviour, including
- [Myksvoll et al. (2018)](https://doi.org/10.1371/journal.pone.0201338)
- [Myksvoll et al. (2020)](https://doi.org/10.3354/aei00359)
- [Sandvik et al. (2020)](https://doi.org/10.1093/icesjms/fsz256)

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
provided by `sandvik2019_ladim.yaml` and `sandvik2019_particles.rls`.


## History

April 2024: Major overhaul of the IBM module
November 2022: Added function to compute salmon lice infectivity 

2019: Refined by Anne Dagrun Sandvik and Pål Næverlid Sævik to account
for improved understanding of salmon lice freshwater avoidance behaviour. 

Created by Ingrid Askeland Johnsen (2014).


## Model assumptions

### Ageing

The IBM variable `age` represents the age of each lice particle in degree-days.
Stage development is deterministic.   