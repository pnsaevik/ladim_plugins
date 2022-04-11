# Sand eel module

The module represents larvae of sand eel. This is a simple module, only random
vertical movement is implemented.


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

Created by Pål Næverlid Sævik (2022)
