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


class Test_ladim_iterator:
    def test_returns_one_dataset_per_timestep_when_multiple_datasets(self, ladim_dset, ladim_dset2):
        it = rasterize.ladim_iterator([ladim_dset, ladim_dset2])
        dsets = list(it)
        assert len(dsets) == ladim_dset.dims['time'] + ladim_dset2.dims['time']

    def test_returns_correct_time_selection(self, ladim_dset):
        iterator = rasterize.ladim_iterator([ladim_dset])
        particle_count = [d.particle_count.values.item() for d in iterator]
        assert particle_count == ladim_dset.particle_count.values.tolist()

        iterator = rasterize.ladim_iterator([ladim_dset])
        time = [d.time.values.item() for d in iterator]
        assert time == ladim_dset.time.values.tolist()

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

            assert out_dset.variables['bincount'][:].tolist() == [
                0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0,
            ]

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
            assert out_dset.variables['bincount'][:].sum() == 6

        with nc.Dataset('test_filter.nc', 'w', diskless=True) as out_dset:
            rasterize.ladim_conc(**conf, output_file=out_dset, afilter=preprocess)
            assert out_dset.variables['bincount'][:].sum() == 4
