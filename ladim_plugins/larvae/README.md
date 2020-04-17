# Fish larvae

The module represents fish larvae.

The theoretical foundation for the module is documented by 
[Vikebø et al. (2007)](https://doi.org/10.3354/meps06979), where the module is
used to predict the dispersal of cod eggs and larvae. The module is also used
by [Vikebø et al. (2010)](https://doi.org/10.1093/icesjms/fsq084) to predict
dispersal of herring eggs and larvae.

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

## History

Created by Frode Vikebø (2007)
Added new set of parameters for herring (Frode Vikebø, 2010)
Ported from Fortran to Python (Mari Myksvoll, 2019)
Modified to match the new Ladim plugin format (Pål Næverlid Sævik, 2020) 
