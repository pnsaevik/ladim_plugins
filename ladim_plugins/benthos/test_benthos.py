import xarray as xr
from ladim_plugins.tests import test_examples as tex
from ladim_plugins.benthos import make_release


def test_output_matches_snapshot():
    module_name = 'benthos'
    out = tex.run_ladim(module_name)
    # out.to_netcdf(get_module_dir(module_name).joinpath('out.nc'))
    ref = xr.load_dataset(tex.get_module_dir(module_name).joinpath('out.nc'))
    tex.check_equal(out, ref)


class Test_make_release:
    def test_returns_dataframe(self):
        config = dict(

        )

        from pandas import DataFrame
        result = make_release.main(config)
        assert isinstance(result, DataFrame)