# IBMs for LADiM
This repository contains individual-based models created at the Institute of
Marine Research (IMR) for use in the in-house particle tracking software
_Lagrangian Advection and Diffusion Model_ (LADiM).

Documentation and examples are provided for each model.

## List of available IBMs

Name | Description | Original author
---- | ----------- | ---------------
sedimentation | Sinking particles | Fernandez, Marcos Carvajalino
egg           | Buoyant fish eggs | Myksvoll, Mari Skuggedal

## Installation

Use `git` to download the repository into a local folder
```
git clone https://git.imr.no/a5606/ladim_ibm.git <ladim_ibm_local_dir>
```

Then use `pip` to install the package. This step is optional, but it
registers the IBMs into the `ladim_ibm` namespace which makes them easier to
access. It also allows automated tests.
```
pip install -e <ladim_ibm_local_dir>
```

The installation can be tested with the command
```
pytest -Wignore --pyargs ladim_ibm
``` 
This command will run ladim on each of the IBMs, using the sample `ladim.yaml`
and `particle.rls` files found in the subpackage folders. The tests succeed if
`ladim` is present on the system, `ladim_ibm` is installed correctly, and the
output from the ladim runs matches exactly with the `out.nc` files found in the
subpackage folders. 


## Usage

1. Copy `ladim.yaml` and `particle.rls` from the IBM subpackage of interest
   into the working directory. 
2. Make desired changes to the `yaml` and `rls` files. More detailed
   instructions are found in the `README.md` file within the subpackage.
3. If the `pip` installation step was skipped, the `ibm.py` module must be
   copied as well, and the `ibm.module` entry in `ladim.yaml` must be changed
   to `"ibm"`.
4. Run `ladim` and the output is written to `out.nc`. 


## Add new IBMs

To add new IBMs, contact the maintainer of the `ladim_ibm` repository. A
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

The test is ran using the command `pytest -Wignore --pyargs ladim_ibm`. The
test succeeds if ladim is able to run the example, and the output matches the
contents of `out.nc`.
