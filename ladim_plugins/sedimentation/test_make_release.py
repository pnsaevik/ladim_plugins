import numpy as np
from ladim_plugins.sedimentation import make_release
import pandas as pd


class Test_main:
    def test_returns_dataframe_when_empty_config(self):
        config = dict()
        result = make_release.main(config)
        assert isinstance(result, pd.DataFrame)

    def test_returns_correct_number_of_particles(self):
        config = dict(num_particles=10)
        result = make_release.main(config)
        assert len(result) == 10

    def test_correct_latlon_when_given(self):
        config = dict(num_particles=5, location=dict(lat=1, lon=2))
        result = make_release.main(config)
        assert result['lat'].values.tolist() == [1, 1, 1, 1, 1]
        assert result['lon'].values.tolist() == [2, 2, 2, 2, 2]

    def test_correct_time_when_given(self):
        config = dict(
            num_particles=5, start_time='2000-01-01', stop_time='2000-01-02')
        result = make_release.main(config)
        assert result['release_time'].values.tolist() == [
            '2000-01-01T00:00:00',
            '2000-01-01T06:00:00',
            '2000-01-01T12:00:00',
            '2000-01-01T18:00:00',
            '2000-01-02T00:00:00',
        ]

    def test_location_when_polygon(self):
        np.random.seed(0)
        config = dict(
            num_particles=5,
            location=dict(lat=[0, 0, 1], lon=[1, 0, 1]),
        )
        result = make_release.main(config)
        assert result['lat'].values.tolist() == [
            0.3541058869333439,
            0.4375872112626925,
            0.10822699921792023,
            0.03633723949897072,
            0.3834415188257777,
        ]
        assert result['lon'].values.tolist() == [
            0.5623808488506793,
            0.966482131015597,
            0.5401824381239879,
            0.11074060120630969,
            0.45447757702366465,
        ]
        assert np.all(result['lon'].values >= result['lat'].values)

    def test_multiple_groups(self):
        config = [
            dict(num_particles=2, group_id=1),
            dict(num_particles=3, group_id=2),
        ]
        result = make_release.main(config)
        assert result['group_id'].values.tolist() == [1, 1, 2, 2, 2]
