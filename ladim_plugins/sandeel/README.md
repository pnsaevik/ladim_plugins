# Sand eel module

The module represents larvae of sand eel. The IBM behaviour includes egg
maturation, larval growth and metamorphosis, and constant vertical mixing
during migration.


## Usage

The eggs should be released around 1st of January, since this is the time when
sandeels are spawning. In the release file, eggs should be placed at the depth
where they will start their migration after being hatched. The internal egg
maturation routine will sample the bottom temperature regardless of the
vertical particle position.

The particle variable `hatch_rate` is uniformly random between 0 and 1, and
represents the development rate at the egg stage. A value of 0.0 represents an
early hatcher, and a value of 1.0 represents a late hatcher.

The particle variable `stage` starts at zero and gradually increases towards 1
when eggs are ready to hatch. After hatching, the larvae follow the currents
passively with a constant vertical mixing (down to a specified depth).

After hatching, the `stage` variable gradually increases towards 2 which marks
the metamorphosis stage. After metamorphosis, the larvae becomes immobile. The
variable `stage` is linearly related to the larval length, where 1.0 is the
initial length (7.73 mm) and 2.0 is the final length (40 mm).  

The sample files `ladim.yaml` and `particles.rls` are a starting point for
making simulations.

Common changes applied to `ladim.yaml`:
- Start date of simulation (`time_control.start_time`)
- Stop date of simulation (`time_control.stop_time`)
- Forcing input file (`gridforce.input_file`)
- Vertical diffusion parameter (`ibm.vertical_mixing`)
- Max vertical diffusion depth (`ibm.max_depth`) 
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

Update September 2022: Add hatching and metamorphose behaviour

Created by Pål Næverlid Sævik (2022)
