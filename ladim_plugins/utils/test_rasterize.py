from ladim_plugins.utils import rasterize
import pytest
import xarray as xr
import numpy as np
import netCDF4 as nc


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
        with nc.Dataset('test_smoke.nc', 'w', diskless=True) as out_dset:
            rasterize.ladim_conc(
                resolution=dict(X=1),
                limits=dict(X=[0, 10]),
                input_file=ladim_dset,
                output_file=out_dset,
            )

            assert out_dset.variables['histogram'][:].tolist() == [
                0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0,
            ]

    def test_can_use_time_bins(self, ladim_dset):
        with nc.Dataset('test_smoke.nc', 'w', diskless=True) as out_dset:
            rasterize.ladim_conc(
                resolution=dict(time=[12, 'h']),
                limits=dict(time=['2000-01-02', '2000-01-04']),
                input_file=ladim_dset,
                output_file=out_dset,
            )

            assert out_dset.variables['histogram'][:].tolist() == [
                4, 0, 2, 0, 0,
            ]
            assert out_dset.variables['time'].units == "microseconds since 1970-01-01"
            assert out_dset.variables['time'].calendar == "proleptic_gregorian"
            dates = ['2000-01-02T00', '2000-01-02T12', '2000-01-03T00', '2000-01-03T12', '2000-01-04T00']
            assert out_dset.variables['time'][:].tolist() == (
                np.array(dates).astype('datetime64[h]') - np.datetime64('1970-01-01')
            ).astype('timedelta64[us]').astype('i8').tolist()

    def test_finds_limits_if_not_given(self, ladim_dset):
        with nc.Dataset('test_limits.nc', 'w', diskless=True) as out_dset:
            rasterize.ladim_conc(
                resolution=dict(X=1),
                input_file=ladim_dset,
                output_file=out_dset,
            )

            assert out_dset.variables['X'][:].tolist() == [5, 6]

    def test_can_apply_filter_function(self, ladim_dset):
        def preprocess(chunk):
            idx = (chunk.farm_id.values > 12345) & (chunk.farm_id.values < 12348)
            return chunk.isel({chunk.farm_id.dims[0]: idx})

        conf = dict(
            resolution=dict(X=1),
            limits=dict(X=[0, 10]),
            input_file=ladim_dset,
        )

        with nc.Dataset('test_filter.nc', 'w', diskless=True) as out_dset:
            rasterize.ladim_conc(**conf, output_file=out_dset, afilter=None)
            assert out_dset.variables['histogram'][:].sum() == 6

        with nc.Dataset('test_filter.nc', 'w', diskless=True) as out_dset:
            rasterize.ladim_conc(**conf, output_file=out_dset, afilter=preprocess)
            assert out_dset.variables['histogram'][:].sum() == 4

    def test_can_apply_filter_string(self, ladim_dset):
        filter_str = '(farm_id > 12345) & (farm_id < 12348)'

        conf = dict(
            resolution=dict(X=1),
            limits=dict(X=[0, 10]),
            input_file=ladim_dset,
        )

        with nc.Dataset('test_filter.nc', 'w', diskless=True) as out_dset:
            rasterize.ladim_conc(**conf, output_file=out_dset, afilter=None)
            assert out_dset.variables['histogram'][:].sum() == 6

        with nc.Dataset('test_filter.nc', 'w', diskless=True) as out_dset:
            rasterize.ladim_conc(**conf, output_file=out_dset, afilter=filter_str)
            assert out_dset.variables['histogram'][:].sum() == 4

    def test_can_apply_weight_function(self, ladim_dset):
        def preprocess(chunk):
            return chunk.X + chunk.Y

        with nc.Dataset('test_weight.nc', 'w', diskless=True) as out_dset:
            rasterize.ladim_conc(
                resolution=dict(X=1),
                limits=dict(X=[0, 10]),
                input_file=ladim_dset,
                output_file=out_dset,
                weights=preprocess,
            )
            assert out_dset.variables['histogram'][:].tolist() == [
                0, 0, 0, 0, 0, 195, 201, 0, 0, 0, 0,
            ]

    def test_can_apply_weight_string(self, ladim_dset):
        with nc.Dataset('test_weight.nc', 'w', diskless=True) as out_dset:
            rasterize.ladim_conc(
                resolution=dict(X=1),
                limits=dict(X=[0, 10]),
                input_file=ladim_dset,
                output_file=out_dset,
                weights="X + Y",
            )
            assert out_dset.variables['histogram'][:].tolist() == [
                0, 0, 0, 0, 0, 195, 201, 0, 0, 0, 0,
            ]


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
