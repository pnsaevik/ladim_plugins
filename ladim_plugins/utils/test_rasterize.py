from ladim_plugins.utils import rasterize
import pytest
import xarray as xr
import numpy as np
from uuid import uuid4


@pytest.fixture(scope='module')
def ladim_dset():
    return xr.Dataset(
        data_vars=dict(
            X=xr.Variable('particle_instance', [5, 5, 6, 6, 5, 6]),
            Y=xr.Variable('particle_instance', [60, 60, 60, 61, 60, 62]),
            Z=xr.Variable('particle_instance', [0, 1, 2, 3, 4, 5]),
            lon=xr.Variable('particle_instance', [5, 5, 6, 6, 5, 6]),
            lat=xr.Variable('particle_instance', [60, 60, 60, 61, 60, 62]),
            instance_offset=xr.Variable((), 0),
            farm_id=xr.Variable('particle', [12345, 12346, 12347, 12348]),
            pid=xr.Variable('particle_instance', [0, 1, 2, 3, 1, 2]),
            particle_count=xr.Variable('time', [4, 2]),
        ),
        coords=dict(
            time=np.array(['2000-01-02', '2000-01-03']).astype('datetime64[D]'),
        ),
    )


@pytest.fixture(scope='class')
def ladim_dset2(ladim_dset):
    d = ladim_dset.copy(deep=True)
    d['instance_offset'] += d.dims['particle_instance']
    d = d.assign_coords(time=d.time + np.timedelta64(2, 'D'))
    return d


class Test_get_edges:
    def test_returns_midpoint_edges_when_asymmetric_int_array(self):
        centers = np.array([2, 4, 8, 10, 20])
        edges = rasterize.get_edges(centers)
        assert edges.tolist() == [1, 3, 6, 9, 15, 25]

    def test_returns_float_array_when_int_input(self):
        centers = np.array([2, 4, 8, 10, 20]).astype(np.int32)
        edges = rasterize.get_edges(centers)
        assert edges.dtype == float

    def test_returns_midpoint_edges_when_asymmetric_datetime64_array(self):
        centers = np.datetime64('2000-01-01') + np.array([2, 4, 8, 10, 20]) * np.timedelta64(1, 'h')
        edges = rasterize.get_edges(centers)
        assert edges.astype(str).tolist() == [
            '2000-01-01T01',
            '2000-01-01T03',
            '2000-01-01T06',
            '2000-01-01T09',
            '2000-01-01T15',
            '2000-01-02T01',
        ]


class Test_ladim_iterator:
    def test_returns_one_dataset_per_timestep_when_multiple_datasets(self, ladim_dset, ladim_dset2):
        it = rasterize.ladim_iterator([ladim_dset, ladim_dset2])
        dsets = list(it)
        assert len(dsets) == ladim_dset.dims['time'] + ladim_dset2.dims['time']

    def test_returns_correct_time_selection(self, ladim_dset):
        iterator = rasterize.ladim_iterator([ladim_dset])
        particle_count = [d.particle_count.values.tolist() for d in iterator]
        assert particle_count == [[4, 4, 4, 4], [2, 2]]

        iterator = rasterize.ladim_iterator([ladim_dset])
        time = [d.time.values for d in iterator]
        assert len(time) == 2
        assert time[0].astype(str).tolist() == ['2000-01-02T00:00:00.000000000'] * 4
        assert time[1].astype(str).tolist() == ['2000-01-03T00:00:00.000000000'] * 2

    def test_returns_correct_instance_selection(self, ladim_dset):
        iterator = rasterize.ladim_iterator([ladim_dset])
        z = [d.Z.values.tolist() for d in iterator]
        assert z == [[0, 1, 2, 3], [4, 5]]

        iterator = rasterize.ladim_iterator([ladim_dset])
        pid = [d.pid.values.tolist() for d in iterator]
        assert pid == [[0, 1, 2, 3], [1, 2]]

    def test_broadcasts_particle_variables(self, ladim_dset):
        iterator = rasterize.ladim_iterator([ladim_dset])
        farm_id = [d.farm_id.values.tolist() for d in iterator]
        assert farm_id == [[12345, 12346, 12347, 12348], [12346, 12347]]

    def test_updates_instance_offset(self, ladim_dset):
        iterator = rasterize.ladim_iterator([ladim_dset])
        offset = [d.instance_offset.values.tolist() for d in iterator]
        assert offset == [0, 4]


class Test_get_conc:
    """Opprinnelig epost:

    Hei,

    Jeg tenker på sikt å lage et noenlunde brukervennlig skript for å lage konsentrasjonsfiler av ladim-filer.

    Men jeg innser at det er skrekkelig mange måter å gjøre dette på, og ulike behov. For eksempel:

    -	Aggregering av partikler på samme grid som input-griddet
    -	Aggregering på finere skala (f.eks. en halv eller tredjedels gridcelle)
    -	Aggregering på grovere skala (f.eks. 10 gridceller). Dette krever at vi tar hensyn til om cellene er på land.
    -	Aggregering på ulike dybdenivåer
    -	Aggregering på ulike tidsperioder
    -	Vekting på f.eks. infektivitet
    -	Filtrering på f.eks. anleggsnummer eller dybde
    -	Valg av aggregeringsmetode (sum, gjennomsnitt per tid, sum per areal)
    -	Inkludering av projeksjonsinformasjon til bruk i kartprogrammer

    Jeg trenger litt innspill på hvordan et slikt skript skal utformes. Jeg ser for meg å lage et skript som kan kjøres fra kommandolinjen, med en config-fil som input. For eksempel:

    ladim_conc config.yaml

    der config.yaml ser omtrent slik ut:

    ladim_file: out.nc
    grid_file: norkyst_800m_blabla.nc  # Optional, is required only for the smoother and for map projection information
    output_file: conc.nc  # Or possibly a set of files?
    weights: time * infectivity / volume
    filter: (40 <= age) & (age <= 170)
    resolution:
                    X: 1
                    Y: 1
                    Z: 5
                    farm_id: discrete
                    time: 1 hour
    limits:  # If absent, compute from ladim particle file
                    X: [0, 1000]
                    Y: [0, 1000]
                    Z: [0, 40]
                    farm_id: [12345, 12346, 12347, 12348]
                    time: [2000-01-01, 2000-01-31]
    smoother: square9  # The value of one grid cell is smeared across a square area of 9x9 grid cells (Default: none)

    Vil et slikt skript dekke behovene dere har? Er det unødvendig komplisert? Config-filen kan sikkert gjøres kortere med et fornuftig sett av default-verdier.

    Pål
    """

    def test_counts_particles_if_standard_input(self, ladim_dset):
        conf = dict(
            resolution=dict(X=1),
            limits=dict(X=[0, 10]),
            input_file=ladim_dset,
            output_file='test_smoke.nc',
            diskless=True,
        )
        with rasterize.ladim_conc(**conf) as out:
            assert out.getData('histogram').tolist() == [
                0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0,
            ]

    def test_can_use_time_bins(self, ladim_dset):
        conf = dict(
            resolution=dict(time=[12, 'h']),
            limits=dict(time=['2000-01-02', '2000-01-04']),
            input_file=ladim_dset,
            output_file='test_smoke.nc',
            diskless=True,
        )
        with rasterize.ladim_conc(**conf) as out:
            assert out.getData('histogram').tolist() == [
                4, 0, 2, 0, 0,
            ]
            attrs = out.getAttrs('time')
            assert attrs['units'] == "microseconds since 1970-01-01"
            assert attrs['calendar'] == "proleptic_gregorian"
            dates = ['2000-01-02T00', '2000-01-02T12', '2000-01-03T00', '2000-01-03T12', '2000-01-04T00']
            assert out.getData('time').tolist() == (
                np.array(dates).astype('datetime64[h]') - np.datetime64('1970-01-01')
            ).astype('timedelta64[us]').astype('i8').tolist()

    def test_finds_limits_if_not_given(self, ladim_dset):
        conf = dict(
            resolution=dict(X=1),
            input_file=ladim_dset,
            output_file='test_limits.nc',
            diskless=True,
        )
        with rasterize.ladim_conc(**conf) as out:
            assert out.getData('X').tolist() == [5, 6, 7]

    def test_finds_limits_if_some_are_missing(self, ladim_dset):
        conf = dict(
            resolution=dict(X=1, Y=1),
            limits=dict(X=[4, 7]),
            input_file=ladim_dset,
            output_file='test_limits.nc',
            diskless=True,
        )
        with rasterize.ladim_conc(**conf) as out:
            assert out.getData('X').tolist() == [4, 5, 6, 7]
            assert out.getData('Y').tolist() == [60, 61, 62, 63]

    def test_can_apply_filter_function(self, ladim_dset):
        def preprocess(farm_id):
            return (farm_id > 12345) & (farm_id < 12348)

        conf = dict(
            resolution=dict(X=1),
            limits=dict(X=[0, 10]),
            input_file=ladim_dset,
            output_file='test_filter.nc',
            diskless=True,
        )

        with rasterize.ladim_conc(**conf, afilter=None) as out:
            assert out.getData('histogram').sum() == 6

        with rasterize.ladim_conc(**conf, afilter=preprocess) as out:
            assert out.getData('histogram').sum() == 4

    def test_can_apply_filter_string(self, ladim_dset):
        filter_str = '(farm_id > 12345) & (farm_id < 12348)'

        conf = dict(
            resolution=dict(X=1),
            limits=dict(X=[0, 10]),
            input_file=ladim_dset,
            output_file='test_filter.nc',
            diskless=True,
        )

        with rasterize.ladim_conc(**conf, afilter=None) as out:
            assert out.getData('histogram').sum() == 6

        with rasterize.ladim_conc(**conf, afilter=filter_str) as out:
            assert out.getData('histogram').sum() == 4

    def test_can_apply_filter_funcname(self, ladim_dset):
        filter_str = 'ladim_plugins.utils.test_rasterize.func_farmid_filter'

        conf = dict(
            resolution=dict(X=1),
            limits=dict(X=[0, 10]),
            input_file=ladim_dset,
            output_file='test_filter.nc',
            diskless=True,
        )

        with rasterize.ladim_conc(**conf, afilter=None) as out:
            assert out.getData('histogram').sum() == 6

        with rasterize.ladim_conc(**conf, afilter=filter_str) as out:
            assert out.getData('histogram').sum() == 4

    def test_can_apply_weight_function(self, ladim_dset):
        def weight_fn(X, Y):
            return X + Y

        conf = dict(
            resolution=dict(X=1),
            limits=dict(X=[0, 10]),
            input_file=ladim_dset,
            output_file='test_weight.nc',
            weights=weight_fn,
            diskless=True,
        )

        with rasterize.ladim_conc(**conf) as out:
            assert out.getData('histogram').tolist() == [
                0, 0, 0, 0, 0, 195, 201, 0, 0, 0, 0,
            ]

    def test_can_apply_weight_string(self, ladim_dset):
        conf = dict(
            resolution=dict(X=1),
            limits=dict(X=[0, 10]),
            input_file=ladim_dset,
            output_file='test_weight.nc',
            weights="X + Y",
            diskless=True,
        )
        with rasterize.ladim_conc(**conf) as out:
            assert out.getData('histogram').tolist() == [
                0, 0, 0, 0, 0, 195, 201, 0, 0, 0, 0,
            ]

    def test_can_apply_weight_funcname(self, ladim_dset):
        conf = dict(
            resolution=dict(X=1),
            limits=dict(X=[0, 10]),
            input_file=ladim_dset,
            output_file='test_weight.nc',
            weights='ladim_plugins.utils.test_rasterize.func_x_plus_y',
            diskless=True,
        )

        with rasterize.ladim_conc(**conf) as out:
            assert out.getData('histogram').tolist() == [
                0, 0, 0, 0, 0, 195, 201, 0, 0, 0, 0,
            ]

    def test_can_split_across_multiple_datasets(self, ladim_dset):
        conf = dict(
            resolution=dict(Y=1, X=1),
            limits=dict(X=[0, 10], Y=[60, 63]),
            input_file=ladim_dset,
            output_file='test_smoke.nc',
            filesplit_dims=['Y'],
            diskless=True,
        )
        with rasterize.ladim_conc(**conf) as out:
            data = np.array([
                [0, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ])
            assert out.getData('histogram').tolist() == data.tolist()
            assert out.getDataset(Y=0)['histogram'][:].tolist() == data[0:1, :].tolist()
            assert out.getDataset(Y=1)['histogram'][:].tolist() == data[1:2, :].tolist()
            assert out.getDataset(Y=2)['histogram'][:].tolist() == data[2:3, :].tolist()
            assert out.getDataset(Y=3)['histogram'][:].tolist() == data[3:4, :].tolist()


@pytest.fixture(scope='module')
def fnames():
    import pkg_resources
    try:
        yield dict(
            outdata=pkg_resources.resource_filename('ladim_plugins.chemicals', 'out.nc'),
        )
    finally:
        pkg_resources.cleanup_resources()


class Test_LadimInputStream:
    def test_can_initialise_from_xr_dataset(self, ladim_dset):
        with rasterize.LadimInputStream(ladim_dset) as dset:
            dset.read()

    def test_can_initialise_from_multiple_xr_datasets(self, ladim_dset, ladim_dset2):
        with rasterize.LadimInputStream([ladim_dset, ladim_dset2]) as dset:
            dset.read()

    def test_can_initialise_from_filename(self, fnames):
        ladim_fname = fnames['outdata']
        with rasterize.LadimInputStream(ladim_fname) as dset:
            dset.read()

    def test_can_initialise_from_multiple_filenames(self, fnames):
        ladim_fname = fnames['outdata']
        with rasterize.LadimInputStream([ladim_fname, ladim_fname]) as dset:
            dset.read()

    def test_can_seek_to_dataset_beginning(self, ladim_dset):
        with rasterize.LadimInputStream(ladim_dset) as dset:
            first_chunk = dset.read()
            second_chunk = dset.read()
            dset.seek(0)
            third_chunk = dset.read()

        assert first_chunk.pid.values.tolist() != second_chunk.pid.values.tolist()
        assert first_chunk.pid.values.tolist() == third_chunk.pid.values.tolist()

    def test_reads_one_timestep_at_the_time(self, ladim_dset, ladim_dset2):
        with rasterize.LadimInputStream([ladim_dset, ladim_dset2]) as dset:
            pids = list(c.pid.values.tolist() for c in dset.chunks())
            assert len(pids) == ladim_dset.dims['time'] + ladim_dset2.dims['time']
            assert pids == [[0, 1, 2, 3], [1, 2], [0, 1, 2, 3], [1, 2]]

    def test_broadcasts_time_vars_when_reading(self, ladim_dset, ladim_dset2):
        with rasterize.LadimInputStream([ladim_dset, ladim_dset2]) as dset:
            counts = list(c.particle_count.values.tolist() for c in dset.chunks())
            assert counts == [[4, 4, 4, 4], [2, 2], [4, 4, 4, 4], [2, 2]]

    def test_broadcasts_particle_vars_when_reading(self, ladim_dset, ladim_dset2):
        with rasterize.LadimInputStream([ladim_dset, ladim_dset2]) as dset:
            farmid = list(c.farm_id.values.tolist() for c in dset.chunks())
            assert farmid == [
                [12345, 12346, 12347, 12348],
                [12346, 12347],
                [12345, 12346, 12347, 12348],
                [12346, 12347],
            ]

    def test_autolimit_aligns_to_wholenumber_resolution_points(self, ladim_dset, ladim_dset2):
        with rasterize.LadimInputStream([ladim_dset, ladim_dset2]) as dset:
            assert dset.find_limits(dict(X=1)) == dict(X=[5, 7])
            assert dset.find_limits(dict(X=2)) == dict(X=[4, 8])
            assert dset.find_limits(dict(X=10)) == dict(X=[0, 10])

    def test_autolimit_aligns_to_wholenumber_resolution_points_when_time(self, ladim_dset, ladim_dset2):
        with rasterize.LadimInputStream([ladim_dset, ladim_dset2]) as dset:
            limits = dset.find_limits(dict(time=np.timedelta64(1, 'h')))
            assert list(limits.keys()) == ['time']
            timestr = np.array(limits['time']).astype('datetime64[h]').astype(str)
            assert timestr.tolist() == ['2000-01-02T00', '2000-01-05T01']

            limits = dset.find_limits(dict(time=np.timedelta64(6, 'h')))
            assert list(limits.keys()) == ['time']
            timestr = np.array(limits['time']).astype('datetime64[h]').astype(str)
            assert timestr.tolist() == ['2000-01-02T00', '2000-01-05T06']


class Test_MultiDataset:
    @pytest.fixture(scope="function")
    def mdset(self):
        with rasterize.MultiDataset(uuid4(), diskless=True) as d:
            yield d

    def test_can_set_main_coordinate(self, mdset):
        mdset.createCoord('mycoord', [1, 2, 3])
        assert mdset.getCoord('mycoord')[:].tolist() == [1, 2, 3]

    def test_can_set_main_variable(self, mdset):
        mdset.createCoord('A', [1, 2, 3])
        mdset.createCoord('B', [10, 20])
        mdset.createVariable('myvar', np.arange(6).reshape((2, 3)), ('B', 'A'))
        assert mdset.getData('myvar').tolist() == [[0, 1, 2], [3, 4, 5]]

    def test_can_get_partial_data_of_main_variable(self, mdset):
        mdset.createCoord('A', [1, 2, 3])
        mdset.createCoord('B', [10, 20])
        mdset.createVariable('myvar', np.arange(6).reshape((2, 3)), ('B', 'A'))

        data = mdset.getData('myvar')
        idx = np.s_[:1, 1:]
        assert mdset.getData('myvar', idx).tolist() == data[idx].tolist()

    def test_can_set_attrs_when_creating_main_coord_or_variable(self, mdset):
        mdset.createCoord('mycoord', [1, 2, 3], attrs=dict(myatt=123))
        mdset.createVariable('myvar', [4, 5, 6], 'mycoord', attrs=dict(myatt=9))
        assert mdset.getAttrs('mycoord') == dict(myatt=123)
        assert mdset.getAttrs('myvar') == dict(myatt=9)

    def test_can_set_data_of_main_variable_after_creation(self, mdset):
        mdset.createCoord('mycoord', [1, 2, 3])
        mdset.createVariable('myvar', [4, 5, 6], 'mycoord')
        mdset.setData('myvar', [7, 8, 9])
        assert mdset.getData('myvar').tolist() == [7, 8, 9]

    def test_can_set_partial_data_of_main_variable_after_creation(self, mdset):
        mdset.createCoord('mycoord', [1, 2, 3])
        mdset.createVariable('myvar', [4, 5, 6], 'mycoord')
        mdset.setData('myvar', 7, idx=0)
        assert mdset.getData('myvar').tolist() == [7, 5, 6]
        mdset.setData('myvar', [0, 9], idx=np.s_[1:])
        assert mdset.getData('myvar').tolist() == [7, 0, 9]

    def test_can_set_initial_constant_data_of_multifile_variable_at_creation(self, mdset):
        mdset.createCoord('A', [1, 2, 3])
        mdset.createCoord('B', [4, 5], cross_dataset=True)

        mdset.createVariable('x', 1, ('B', 'A'))
        assert mdset.getData('x').tolist() == [[1, 1, 1], [1, 1, 1]]

    def test_can_set_attrs_of_multifile_variable(self, mdset):
        mdset.createCoord('A', [1, 2, 3])
        mdset.createCoord('B', [4, 5], cross_dataset=True)
        mdset.createVariable('x', 1, ('B', 'A'), attrs=dict(myatt=123))
        assert mdset.getAttrs('x') == dict(myatt=123)

    def test_can_set_initial_slicewise_data_of_multifile_variable_at_creation(self, mdset):
        mdset.createCoord('A', [1, 2, 3])
        mdset.createCoord('B', [4, 5], cross_dataset=True)

        mdset.createVariable('x', np.arange(3).reshape((1, 3)), ('B', 'A'))
        assert mdset.getData('x').tolist() == [[0, 1, 2], [0, 1, 2]]

    def test_can_get_partial_data_of_multifile_variable(self, mdset):
        mdset.createCoord('A', [1, 2, 3, 4])
        mdset.createCoord('B', [5, 6, 7, 8, 9], cross_dataset=True)

        mdset.createVariable('x', np.arange(4), ('B', 'A'))
        assert mdset.getData('x', np.s_[1:3, 1:4]).tolist() == [[1, 2, 3], [1, 2, 3]]

    def test_can_set_partial_data_of_multifile_variable_after_creation(self, mdset):
        mdset.createCoord('A', [1, 2, 3])
        mdset.createCoord('B', [5, 6, 7, 8], cross_dataset=True)

        mdset.createVariable('x', 1, ('B', 'A'))
        mdset.setData('x', [[8], [9]], np.s_[1:3, 1:2])
        data = mdset.getData('x').tolist()
        assert data == [[1, 1, 1], [1, 8, 1], [1, 9, 1], [1, 1, 1]]

    def test_can_set_partial_data_when_multiple_crosscoords(self, mdset):
        mdset.createCoord('A', [1, 2])
        mdset.createCoord('B', [3, 4, 5], cross_dataset=True)
        mdset.createCoord('C', [6, 7, 8, 9])
        mdset.createCoord('D', [8, 7, 6, 5, 4], cross_dataset=True)

        mdset.createVariable('x', 1, ('A', ))
        mdset.createVariable('y', 2, ('B', 'A', ))
        mdset.createVariable('z', 3, ('C', 'B', 'A', ))
        mdset.createVariable('w', 4, ('D', 'C', 'B', 'A', ))

        in_data = np.arange(6).reshape((1, 2, 3, 1))
        idx = np.s_[1::5, :2, -3:, 0:1]
        mdset.setData('w', in_data, idx)
        out_data = mdset.getData('w')
        assert out_data[idx].ravel().tolist() == in_data.ravel().tolist()

    def test_cannot_create_variables_after_subdataset_creation(self, mdset):
        # Create main-dataset coordinate (does not lock dataset)
        mdset.createCoord('A', [1, 2])
        # Create main-dataset variable (does not lock datset)
        mdset.createVariable('x', 1, 'A')
        # Set main-dataset variable data (does not lock dataset)
        mdset.setData('x', 2)
        # Create cross-dataset coordinate (does not lock dataset)
        mdset.createCoord('B', [1, 2, 3], cross_dataset=True)
        # Create cross-dataset variable (does not lock dataset)
        mdset.createVariable('y', 3, 'B')
        # Access cross-dataset variable (DOES lock dataset)
        mdset.setData('y', 4)
        # Cannot create new main-dataset coordinates when locked
        with pytest.raises(TypeError):
            mdset.createCoord('C', [1, 2])
        # Cannot create new cross-dataset coordinates when locked
        with pytest.raises(TypeError):
            mdset.createCoord('D', [1, 2], cross_dataset=True)
        # Cannot create new main-dataset variables when locked
        with pytest.raises(TypeError):
            mdset.createVariable('z', 5, 'A')
        # Cannot create new cross-dataset variables when locked
        with pytest.raises(TypeError):
            mdset.createVariable('w', 6, 'B')
        # CAN access variable data after locking
        mdset.setData('x', 7)
        assert mdset.getData('x').tolist() == [7, 7]
        mdset.setData('y', 8)
        assert mdset.getData('y').tolist() == [8, 8, 8]


class Test_Histogrammer:
    def test_can_compute_centers_from_resolution_and_limits(self):
        h = rasterize.Histogrammer(
            resolution=dict(x=1, y=2, z=4),
            limits=dict(x=[0, 4], y=[1, 10], z=[-10, 10]),
        )
        assert h.coords['x']['centers'].tolist() == [0, 1, 2, 3, 4]
        assert h.coords['y']['centers'].tolist() == [1, 3, 5, 7, 9]
        assert h.coords['z']['centers'].tolist() == [-10, -6, -2, 2, 6, 10]

    def test_can_compute_edges_from_resolution_and_limits(self):
        h = rasterize.Histogrammer(
            resolution=dict(x=1, y=2, z=4),
            limits=dict(x=[0, 4], y=[1, 10], z=[-10, 10]),
        )
        assert h.coords['x']['edges'].tolist() == [-.5, .5, 1.5, 2.5, 3.5, 4.5]
        assert h.coords['y']['edges'].tolist() == [0, 2, 4, 6, 8, 10]
        assert h.coords['z']['edges'].tolist() == [-12, -8, -4, 0, 4, 8, 12]

    def test_can_generate_histogram_piece_from_chunk(self):
        h = rasterize.Histogrammer(
            resolution=dict(z=3, y=2, x=1),
            limits=dict(z=[0, 3], y=[0, 4], x=[0, 3]),
        )
        chunk = xr.Dataset(dict(x=[0, 1, 3], y=[0, 2, 4], z=[0, 1, 3]))
        hist_piece = next(h.make(chunk))
        assert hist_piece['values'].tolist() == [
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]],
        ]
        start = [idx.start for idx in hist_piece['indices']]
        assert start == [0, 0, 0]
        stop = [idx.stop for idx in hist_piece['indices']]
        assert stop == [2, 3, 4]


class Test_adaptive_histogram:
    def test_(self):
        x = [1, 2, 3, 4]
        y = [10, 20, 30, 40]
        z = [100, 200, 300, 400]

        xbins = [0, .5, 1.5, 3, 4]
        ybins = [15, 25, 35, 45, 50]
        zbins = [0, 1000]

        sample = [z, y, x]
        bins = [zbins, ybins, xbins]

        hist2 = np.zeros([len(zbins)-1, len(ybins)-1, len(xbins)-1], dtype=int)
        hist, idx = rasterize.Histogrammer.adaptive_histogram(sample, bins)
        hist2[idx] = hist

        hist3, _ = np.histogramdd(sample, bins)
        assert hist3.tolist() == hist2.tolist()


def func_farmid_filter(farm_id):
    return (farm_id > 12345) & (farm_id < 12348)


def func_x_plus_y(X, Y):
    return X + Y
