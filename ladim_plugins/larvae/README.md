# Fish larvae

The module represents fish eggs that mature into larvae. Vertical diffusion
and light-induced swimming behaviour is included. Larvae growth and mortality
is included.  

The theoretical foundation for the module is documented by 
[Vikebø et al. (2007)](https://doi.org/10.3354/meps06979),
[Vikebø et al. (2010)](https://doi.org/10.1093/icesjms/fsq084) and
[Myksvoll et al. (2011)](http://dx.doi.org/10.1080/19425120.2011.595258).


## Algorithm

1.  Particles whose age (in degree-days) are greater than `ibm.hatch_age` are
    considered larvae. The rest is regarded as eggs.
    
2.  Vertical egg movement. Eggs move according to their buoyancy (plus
    vertical mixing). Terminal velocity is computed from Stokes formula.
    
3.  Vertical larvae movement. Larvae swim speed is computed from their length.
    They swim upwards when it is dark, and downwards when it is lit.
    
4.  Larvae growth and mortality. Computed from a predefined formula.


Larvae growth rate from [Folkvord (2005)](https://dx.doi.org/10.1139/f05-008):

```
T = temperature in celcius
t = growth time in days
W = ln[larva weight in grams]
f = 1.08 + 1.79 T - 0.074 TW - 0.0965 TW² + 0.0112 TW³
r = ln(1 + 0.001 f)

```

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
