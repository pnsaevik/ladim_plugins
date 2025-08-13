# Thredds module

The module enables using a Thredds server to provide s-coordinate 
ROMS netCDF files as forcing data.


## Usage

Modify `ladim.yaml` and `particles.rls` to suit your needs. In particluar, 
the forcing input keyword `gridforce.input_file` in `ladim.yaml` must be changed to contain a
string pattern for the thredds server. An example string is given as a
commented line.
The format string follows standard python conventions. 

Other common changes applied to `ladim.yaml`:
- Start date of simulation (`time_control.start_time`)
- Stop date of simulation (`time_control.stop_time`)
- Horizontal diffusion parameter (`numerics.diffusion`)
- IBM module and parameters (`ibm.module`)
- Time step length (`numerics.dt`)
- Output frequency (`output_variables.outper`)

The file `particles.rls` is a tab-delimited text file containing particle
release time and location, as well as particle attributes at the release time.
The order of the columns is given by the entry `particle_release.variables`
within `ladim.yaml`.

Finally, copy `ladim.yaml` and `particles.rls` to a separate directory and
run `ladim` here.


## Output

The simulation result is stored in a file specified by the `output_file`
entry in `ladim.yaml`. The output variables are specified by the
`output_variables` entries. 


## History

Created by Pål Næverlid Sævik as part of the ADepoPlan project (2025).
