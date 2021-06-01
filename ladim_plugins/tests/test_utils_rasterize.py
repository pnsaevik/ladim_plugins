import pytest
from ladim_plugins.utils import rasterize
import xarray as xr
import numpy as np


class Test_ladim_chunks:
    @pytest.fixture(scope='class')
    def data(self):
        ladim_0 = xr.Dataset(
            data_vars=dict(
                instance_offset=5,
                release_time=xr.Variable(
                    dims='particle',
                    data=np.datetime64('2000') + np.arange(5) + np.timedelta64(1, 'D'),
                ),
                farmid=xr.Variable(dims='particle', data=[101, 102, 101, 102, 101]),
                pid=xr.Variable(
                    dims='particle_instance',
                    data=[0, 0, 1, 0, 1, 2, 1, 2, 3, 4],
                ),
                X=xr.Variable(
                    dims='particle_instance',
                    data=[1, 1.1, 2, 1.2, 2.1, 3, 2.2, 3.1, 4, 5],
                ),
                particle_count=xr.Variable(
                    dims='time',
                    data=[1, 2, 3, 4],
                ),
            ),
            coords=dict(
                time=np.datetime64('2000') + np.arange(4) + np.timedelta64(1, 'D'),
            ),
        )
        ladim_1 = ladim_0.isel(time=range(2), particle_instance=range(3))
        ladim_2 = ladim_0.isel(time=range(2, 4), particle_instance=range(3, 10))
        ladim_2['instance_offset'] += 3

        return [ladim_0, ladim_1, ladim_2]

    def test_number_of_chunks_equals_number_of_time_slots_if_unlimited_rows(self, data):
        _, ladim_1, ladim_2 = data
        chunks = rasterize.ladim_chunks([ladim_1, ladim_2], ['X'])
        num_chunks = sum(1 for _ in chunks)
        assert num_chunks == len(ladim_1.time) + len(ladim_2.time)

    def test_number_of_chunks_increases_if_small_maxrow(self, data):
        _, ladim_1, ladim_2 = data
        chunks = rasterize.ladim_chunks([ladim_1, ladim_2], ['X'], max_rows=2)
        num_chunks = sum(1 for _ in chunks)
        assert num_chunks > len(ladim_1.time) + len(ladim_2.time)

    def test_chunks_of_merged_and_separated_datasets_are_equivalent(self, data):
        ladim_0, ladim_1, ladim_2 = data
        chunks_12 = rasterize.ladim_chunks([ladim_1, ladim_2], ['X'])
        chunks_0 = rasterize.ladim_chunks([ladim_0], ['X'])
        for chunk_0, chunk_12 in zip(chunks_0, chunks_12):
            assert chunk_0['X'].tolist() == chunk_12['X'].tolist()

    def test_returns_dict_of_ndarrays(self, data):
        _, ladim_1, ladim_2 = data
        varnames = ['X', 'farmid', 'time']
        chunk = next(rasterize.ladim_chunks([ladim_1, ladim_2], varnames))
        assert isinstance(chunk, dict)
        assert set(chunk.keys()) == set(varnames)
        assert all(isinstance(v, np.ndarray) for v in chunk.values())

    def test_chunks_have_correct_size(self, data):
        ladim_0, ladim_1, ladim_2 = data
        chunks = rasterize.ladim_chunks([ladim_1, ladim_2], ['X'])
        pcounts_expected = ladim_0.particle_count.values.tolist()
        pcounts_actual = [len(chunk['X']) for chunk in chunks]
        assert pcounts_actual == pcounts_expected

    def test_broadcasts_particle_vars(self, data):
        ladim_0, ladim_1, ladim_2 = data
        chunks = rasterize.ladim_chunks([ladim_1, ladim_2], ['farmid'])
        pcounts_expected = ladim_0.particle_count.values.tolist()
        pcounts_actual = [len(chunk['farmid']) for chunk in chunks]
        assert pcounts_actual == pcounts_expected

    def test_broadcasts_time_vars(self, data):
        ladim_0, ladim_1, ladim_2 = data
        chunks = rasterize.ladim_chunks([ladim_1, ladim_2], ['time'])
        pcounts_expected = ladim_0.particle_count.values.tolist()
        pcounts_actual = [len(chunk['time']) for chunk in chunks]
        assert pcounts_actual == pcounts_expected
