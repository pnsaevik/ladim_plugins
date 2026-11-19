import numpy as np
import pytest
import xarray as xr
import yaml
from scipy.ndimage import map_coordinates

import ladim
from ladim_plugins.tests import test_examples as te


@pytest.mark.parametrize("module_name", ["sedimentation", "mine"])
def test_resuspension(module_name, tmp_path):
    # Particles are released on the sea floor (inactive, settled). They are
    # resuspended if the bottom shear stress is larger than `taucrit`, and are then
    # lifted off the sea floor by the vertical diffusion of ladim.
    seabed = xr.open_dataset(te.get_module_dir(module_name) / 'forcing.nc')['h'].values
    x0, y0, num = 4, 4, 20
    z0 = float(seabed[y0, x0]) - 0.001  # Just above the sea floor

    def run(taucrit):
        config = yaml.safe_load(te.get_config(module_name))
        start = config['time_control']['start_time']
        rls = tmp_path / 'particles.rls'
        rls.write_text(''.join(
            f"0\t{start:%Y-%m-%dT%H:%M:%S}\t{x0}\t{y0}\t{z0}\t0.0003\n" for _ in range(num)
        ))
        config['files']['particle_release_file'] = str(rls)
        config['ibm'].update(taucrit=taucrit, lifespan=3600)  # [N/m2], [s]
        config['numerics'].update(seed=1)

        with te.chdir_temp() as test_dir:
            ladim.main(yaml.safe_dump(config))
            out = xr.load_dataset(test_dir / 'out.nc', decode_cf=False)

        n = int(out['particle_count'].values[-1])  # Final time step
        return {k: out[k].values[-n:] for k in ('X', 'Y', 'Z', 'active')}

    # No resuspension: Nothing moves
    result = run(taucrit=1000)
    assert np.all(result['Z'] == z0)
    assert np.all(result['active'] == 0)

    # Resuspension: A large part of the particles are above the sea floor
    result = run(taucrit=0)
    depth = map_coordinates(seabed, [result['Y'], result['X']], order=1)
    height = depth - result['Z']
    tol = 1e-4  # The output positions are float32
    assert np.mean(height > 0.02) > 0.5
    assert np.all(height > -tol)
    assert np.all((result['active'] != 0) == (height > tol))  # Mobile or settled again
