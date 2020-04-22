![travis](https://travis-ci.com/pnsaevik/ladim_plugins.svg?branch=master)

# Plugins for LADiM
This repository contains plugins for the _Lagrangian Advection and Diffusion
Model_ (LADiM), which is the particle tracking software used at the Institute
of Marine Research (IMR).
([https://github.com/bjornaa/ladim](https://github.com/bjornaa/ladim)) 

Documentation and examples are provided for each model.

## List of available plugins

Name                                                   | Description       
------------------------------------------------------ | ------------------
[chemicals](ladim_plugins/chemicals)         | Passive tracer    
[egg](ladim_plugins/egg)                     | Buoyant fish eggs 
[salmon_lice](ladim_plugins/salmon_lice)     | Salmon lice larvae
[sedimentation](ladim_plugins/sedimentation) | Sinking particles 
[lunar_eel](ladim_plugins/lunar_eel)         | Glass eels with lunar compass
 

## Installation

Install using the following commands (the first command can be skipped if
LADiM is already present on your system):

```
pip install git+https://github.com/bjornaa/ladim.git
pip install git+https://github.com/pnsaevik/ladim_plugins.git
```

The installation can be tested with the command
```
pytest -Wignore --pyargs ladim_plugins
``` 
This command will run ladim on each of the plugins, using the sample `ladim.yaml`
and `particle.rls` files found in the subpackage folders. The tests succeed if
`ladim` is present on the system, `ladim_plugins` is installed correctly, and the
output from the ladim runs matches exactly with the `out.nc` files found in the
subpackage folders. 


## Usage

1. Copy `ladim.yaml` and `particle.rls` from the IBM subpackage of interest
   into the working directory. 
2. Make desired changes to the `yaml` and `rls` files. More detailed
   instructions are found in the `README.md` file within the subpackage.
3. Run `ladim` and the output is written to `out.nc`. 


## Contribute

To add new plugins, contact the maintainer of the `ladim_plugins` repository. A
properly structured IBM subpackage has the following ingredients:

1. A file `__init__.py` containing the statement `from .ibm import IBM`
2. The IBM module itself, named `ibm.py`
3. A `README.md` file containing instructions for use
4. A simple test example. This includes a `ladim.yaml` configuration file,
   a `particles.rls` release file, a `forcing.nc` forcing file (may be
   copied from another subpackage), and a ladim output file named `out.nc`.
5. Optionally a generic routine for generating release files, named
  `make_release.py` or `make_release.m`.
6. Optionally additional automated test modules, named `test_*.py`

An ideal test example should be quick to run and easy to analyze, 
but still be complicated enough to demonstrate most
capabilities of the IBM model. To achieve this, it may be a good idea to use
only a few time steps and a few particles at selected positions. In some cases
it may also be necessary to specify somewhat unrealistic particle parameters
to demonstrate certain features.

The test is ran using the command `pytest -Wignore --pyargs ladim_plugins`. The
test succeeds if ladim is able to run the example, and the output matches the
contents of `out.nc`.


## History

**Version 1.1** (Apr 2020): Resuspension added to `sedimentation` module.

**Version 1.0** (Dec 2019): Initial version, including the modules `chemicals`, `egg`,
`salmon_lice` and `sedimentation`.
