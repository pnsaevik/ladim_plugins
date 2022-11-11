import xarray as xr
import numpy as np
import pytest

from ladim_plugins.shrimp import ibm

try:
    import holoviews as hv
    import matplotlib.pyplot as plt
    hv.extension('matplotlib')

except ImportError:
    hv = None
    plt = None


@pytest.mark.skip
class Test_growth:
    @pytest.fixture()
    def grid(self):
        pass

    @pytest.fixture()
    def state(self):
        return dict(
            Z=np.zeros(5),
            stage=np.ones(5),
            age=np.zeros(5),
            temp=np.array([1, 3, 5, 8, 10]),
            time=np.datetime64('2000-01-01'),
        )

    @pytest.fixture()
    def forcing(self):
        pass

    @pytest.fixture()
    def shrimp_ibm(self):
        config = dict(
            ibm=dict(
                vertical_mixing=0,
            ),
            dt=86400,
        )

        return ibm.IBM(config)

    @pytest.fixture()
    def result(self, grid, state, forcing, shrimp_ibm):
        stage = []
        t = []

        start = np.datetime64('2000-01-01T00')
        stop = np.datetime64('2000-02-10T00')

        state['time'] = start
        while stop > state['time']:
            stage.append(state['stage'].copy())
            t.append(state['time'].copy())
            shrimp_ibm.update_ibm(grid, state, forcing)
            state['time'] += np.timedelta64(1, 's') * shrimp_ibm.dt

        return xr.Dataset(
            data_vars=dict(
                stage=xr.Variable(
                    dims=('time', 'pid'),
                    data=stage,
                ),
            ),
            coords=dict(
                time=xr.Variable(
                    dims='time',
                    data=t,
                ),
            ),
        )

    def test_looks_sensible(self, result):
        plot = plot_particle(result.stage)
        fig = hv.render(plot, backend='matplotlib')
        plt.figure(fig)
        plt.show()


@pytest.mark.skip
class Test_vertical_distribution:
    @pytest.fixture()
    def grid(self):
        class Grid:
            @staticmethod
            def lonlat(x, y):
                return 5 + np.zeros_like(x), 70 + np.zeros_like(y)

        return Grid()

    @pytest.fixture()
    def state(self):
        num = 1000
        return dict(
            depth_quantile=np.zeros(num),
            X=np.zeros(num),
            Y=np.zeros(num),
            Z=np.linspace(0, 40, num),
            temp=np.ones(num),
            age=np.zeros(num),
            stage=np.ones(num)*3,
            time=np.datetime64('2000-01-01'),
        )

    @pytest.fixture()
    def forcing(self):
        pass

    @pytest.fixture()
    def shrimp_ibm(self):
        config = dict(
            ibm=dict(
                mindepth_day=[20, 20, 20, 20, 20],  # Minimum preferred depth at daytime [m]
                maxdepth_day=[20, 20, 200, 200, 200],  # Maximum preferred depth at daytime [m]
                mindepth_night=[0, 0, 0, 0, 0],  # Minimum preferred depth at nighttime [m]
                maxdepth_night=[0, 0, 100, 100, 100],  # Maximum preferred depth at nighttime [m]
                vertical_mixing=[0.01, 0.01, 0.25, 0.25, 0.25],  # Random vertical mixing coefficient [m2/s]
                vertical_speed=[0.001, 0.001, 0.005, 0.005, 0.005],  # Vertical speed if outside preferred depth range [m/s]
            ),
            dt=600,
        )

        return ibm.IBM(config)

    @pytest.fixture()
    def result(self, grid, state, forcing, shrimp_ibm):
        Z = []
        t = []

        start = np.datetime64('2000-03-01T00')
        stop = np.datetime64('2000-03-05T00')

        state['time'] = start
        while stop > state['time']:
            Z.append(state['Z'].copy())
            t.append(state['time'].copy())
            shrimp_ibm.update_ibm(grid, state, forcing)
            state['time'] += np.timedelta64(1, 's') * shrimp_ibm.dt

        return xr.Dataset(
            data_vars=dict(
                Z=xr.Variable(
                    dims=('time', 'pid'),
                    data=Z,
                ),
            ),
            coords=dict(
                time=xr.Variable(
                    dims='time',
                    data=t,
                ),
            ),
        )

    def test_looks_sensible(self, result):
        plot = plot_quantile(-result.Z)
        fig = hv.render(plot, backend='matplotlib')
        plt.figure(fig)
        plt.show()


def plot_quantile(darr):
    q = darr.quantile([.125, .25, .375, 0.5, .625, .75, .875], dim=darr.dims[-1])

    x = q.time.values
    y1 = q.isel(quantile=0).values
    y2 = q.isel(quantile=1).values
    y3 = q.isel(quantile=2).values
    y4 = q.isel(quantile=3).values
    y5 = q.isel(quantile=4).values
    y6 = q.isel(quantile=5).values
    y7 = q.isel(quantile=6).values
    y_ex = darr[:, 0].values

    xname = darr.dims[0]
    yname = darr.name

    area_lgt = hv.Area((x, y1, y7), kdims=xname, vdims=[yname, 'y2'])
    area_med = hv.Area((x, y2, y6), kdims=xname, vdims=[yname, 'y2'])
    area_drk = hv.Area((x, y3, y5), kdims=xname, vdims=[yname, 'y2'])
    line_mid = hv.Curve((x, y4), kdims=xname, vdims=yname)
    line_ex = hv.Curve((x, y_ex), kdims=xname, vdims=yname)

    area_lgt = area_lgt.opts(facecolor='#e0e0e0', edgecolor='black')
    area_med = area_med.opts(facecolor='#c0c0c0', edgecolor='black')
    area_drk = area_drk.opts(facecolor='#a0a0a0', edgecolor='black')
    line_mid = line_mid.opts(color='black')
    line_ex = line_ex.opts(color='red', linewidth=.5)

    plot = area_lgt * area_med * area_drk * line_mid * line_ex

    return plot.opts(fig_inches=12, aspect=4)


def plot_particle(darr):
    x, pid = np.meshgrid(darr.time.values, np.arange(darr.sizes['pid']), indexing='ij')
    y = darr.values

    plot = hv.Curve((x.ravel(), y.ravel(), pid.ravel()), kdims='time', vdims=[darr.name, 'pid'])

    return plot.groupby('pid').overlay()
