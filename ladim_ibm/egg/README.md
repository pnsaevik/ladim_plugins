# Boyant eggs

_Brief description goes here._

The theoretical foundation for the module is documented by 
[Myksvoll et al. (2011)](http://dx.doi.org/10.1080/19425120.2011.595258).

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

## SVIM example

An example of configuration file and release file used in a real scenario is
provided by `svim_ladim.yaml` and `svim_particles.rls`. The code used to make
the release file is provided by `svim_make_release.m`.  
 

Start with make_release_egg.m
- choose grid
- choose release locations in grid coordinates
- choose release depth
- choose egg buoyancy
- choose release time

Output is release_svim_egg.rls

Check ibm_egg.py
- choose egg diameter (cod=0.0014, saithe=0.0011)
- specify maximum depth of particles

Check ladim.yaml
- choose start and stop time
- add path to grid and input file
- add name of release file
- ibm_module: 'ibm_egg'
- choose output frequency and multiple files
- choose proper time step and diffusion depending on horizontal resoultion
