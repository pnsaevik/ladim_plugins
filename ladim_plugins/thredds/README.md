# NK800 model with S-coordinates from met.no

The module simplifies the use of NorKyst800 S-coordinate forcing data from the
THREDDS server https://thredds.met.no/.


## Usage

Modify `ladim.yaml` and `particles.rls` to suit your needs. In particluar, 
the forcing input keyword `gridforce.input_file` in `ladim.yaml` must be changed to contain a
string pattern for the thredds server. Currently, the default string pattern is
```python
default_pattern = (
    "https://thredds.met.no/thredds/dodsC/fou-hi/norkyst800m-1h/"
    "NorKyst-800m_ZDEPTHS_his.an.{year:04}{month:02}{day:02}00.nc"
)
``` 

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

Created by Pål Næverlid Sævik (2025).
