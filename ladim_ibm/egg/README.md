# Boyant eggs

_Brief description goes here._

The theoretical foundation for the module is documented by 
[Myksvoll et al. (2011)](http://dx.doi.org/10.1080/19425120.2011.595258).

## Usage

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
