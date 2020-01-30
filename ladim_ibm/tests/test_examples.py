import ladim
import tempfile
import contextlib
import os
import pathlib
import yaml
import xarray as xr
import pytest
import numpy as np
import pkg_resources


module_names = ["egg", "sedimentation"]


@pytest.mark.parametrize("module_name", module_names)
def test_output_matches_snapshot(module_name):
    out = run_ladim(module_name)
    # out.to_netcdf(get_module_dir(module_name).joinpath('out.nc'))
    ref = xr.load_dataset(get_module_dir(module_name).joinpath('out.nc'))
    check_equal(out, ref)


def check_equal(new, ref):
    dt = {'date': ''}
    assert {**new.attrs, **dt} == {**ref.attrs, **dt}
    assert new.variables.keys() == ref.variables.keys()
    assert new.coords.keys() == ref.coords.keys()
    assert new.data_vars.keys() == ref.data_vars.keys()
    assert new.dims.items() == ref.dims.items()

    for k in new.variables.keys():
        assert new.variables[k].values.tolist() == ref.variables[k].values.tolist()


def run_ladim(module_name):
    # Change into a temporary folder `test_dir`
    with chdir_temp() as test_dir:
        np.random.seed(0)  # To ensure consistent results
        ladim.main(get_config(module_name))

        # Read and return output data
        return xr.load_dataset(test_dir.joinpath('out.nc'))


def get_config(module_name):
    # Load yaml config string
    package = 'ladim_ibm.' + module_name
    with pkg_resources.resource_stream(package, 'ladim.yaml') as config_file:
        config_string = config_file.read()

    # Append module dir to file names so that ladim can find the config files
    # from a different folder
    module_dir = get_module_dir(module_name)
    config_dict = yaml.safe_load(config_string)
    config_dict['files']['particle_release_file'] = os.path.join(
        module_dir, config_dict['files']['particle_release_file'])
    config_dict['gridforce']['input_file'] = os.path.join(
        module_dir, config_dict['gridforce']['input_file'])

    # Re-serialize as yaml
    import io
    buf = io.StringIO()
    yaml.safe_dump(config_dict, buf)
    buf.seek(0)
    return buf.read()


def get_module_dir(module_name):
    this_dir = pathlib.Path(__file__).parent
    package_dir = this_dir.parent
    return package_dir.joinpath(module_name)


@contextlib.contextmanager
def chdir_temp():
    tempdir = None
    try:
        tempdir = pathlib.Path(tempfile.mkdtemp(prefix='ladim_ibm_test_dir_'))
        curdir = os.getcwd()
        os.chdir(tempdir)
        yield tempdir
        os.chdir(curdir)
    finally:
        if tempdir is not None:
            for fname in tempdir.glob('*'):
                fname.unlink()
            tempdir.rmdir()
