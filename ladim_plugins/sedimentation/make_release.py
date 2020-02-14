import numpy as np
import xarray as xr


def main(
    location=None, depth=0, start_time='2000-01-01', stop_time='2000-01-01',
    num_particles=0,
):
    # noinspection PyPackageRequirements
    import pandas as pd

    release = pd.DataFrame(
        columns=['active', 'release_time', 'lat', 'lon', 'Z', 'sink_vel', 'group_id'])

    # Handle default arguments
    if location is None:
        location = dict(lat=0., lon=0.)

    # Set parameters
    release['active'] = 1
    release['release_time'] = linspace_time(start_time, stop_time, num_particles)
    release['lat'], release['lon'] = latlon(location, num_particles)
    release['Z'] = depth
    release['sink_vel'] = sinkvel(num_particles)
    release['group_id'] = 0

    return release


def linspace_time(start, stop, num, granularity='ms'):
    start_t = np.datetime64(start)
    stop_t = np.datetime64(stop)
    timediff = (stop_t - start_t) / np.timedelta64(1, granularity)
    dt = np.linspace(0, 1, num) * timediff
    return start_t + dt.astype(f'timedelta64[{granularity}]')


# noinspection PyPackageRequirements
def sinkvel(n):
    from scipy.interpolate import InterpolatedUnivariateSpline
    sinkvel_tab = np.array([.100, .050, .025, .015, .010, .005, 0])
    cumprob_tab = np.array([.000, .662, .851, .883, .909, .937, 1])
    fn = InterpolatedUnivariateSpline(cumprob_tab, sinkvel_tab, k=2)
    return fn(np.random.rand(n))


def latlon(loc, n):
    # If lat/lon is given directly
    if isinstance(loc, dict):
        lat = np.vectorize(to_numeric_latlon)(loc['lat'])
        lon = np.vectorize(to_numeric_latlon)(loc['lon'])

        # If singular point
        if len(lat.shape) == 0 and len(lon.shape) == 0:
            return np.ones(n)*lat, np.ones(n)*lon


def to_numeric_latlon(lat_or_lon):
    if isinstance(lat_or_lon, str):
        if '°' in lat_or_lon:
            deg_str, rest = lat_or_lon.split('°')
            mn = 0
            sec = 0
            if "''" in rest:
                rest = rest.replace("''", "")
                mn_str, sec_str = rest.split("'")
                mn = float(mn_str)
                sec = float(sec_str)
            elif "'" in rest:
                mn = float(rest.replace("'", ""))
            return float(deg_str) + mn/60 + sec/3600
        else:
            return float(lat_or_lon)
    return lat_or_lon


# noinspection PyPackageRequirements
def get_release_data(date, lat, lon, z, distribution, num_particles):
    np.random.seed(0)

    from shapely.wkt import loads as load_wkt

    distribution_shape = load_wkt(distribution)
    if distribution_shape.wkt == 'POINT (0 0)':
        lat_arr = np.repeat(lat, num_particles)
        lon_arr = np.repeat(lon, num_particles)
    elif distribution_shape.wkt.startswith('POLYGON'):
        from oceanviz.crs import transform as crs_transform
        coords = np.array(distribution_shape.exterior.xy).T
        x, y = get_polygon_sample(coords, num_particles)
        lat_arr, lon_arr = crs_transform(x, y, ('local', lat, lon), 'ETRS89')
    else:
        raise ValueError(f'Distribution "{distribution_shape.wkt}" is unknown')

    try:
        zmin = z[0]
        zmax = z[1]
        z_arr = zmin + np.random.rand(num_particles) * (zmax - zmin)
    except TypeError:
        z_arr = np.repeat(z, num_particles)

    date_arr = np.repeat(np.datetime64(date), num_particles)

    return xr.Dataset(
        data_vars=dict(
            release_time=xr.Variable(
                dims='particle',
                data=date_arr,
            ),
            lat=xr.Variable(
                dims='particle',
                data=lat_arr,
            ),
            lon=xr.Variable(
                dims='particle',
                data=lon_arr,
            ),
            Z=xr.Variable(
                dims='particle',
                data=z_arr,
            ),
        )
    )


def _unit_triangle_sample(num):
    xy = np.random.rand(num*2).reshape((2, -1))
    is_upper_triangle = np.sum(xy, axis=0) > 1
    xy[:, is_upper_triangle] = 1 - xy[:, is_upper_triangle]
    return xy


# noinspection PyPackageRequirements
def get_polygon_sample(coords, num):
    from shapely.geometry import Polygon
    from shapely.ops import triangulate
    np.random.seed(0)

    # Triangulate the polygon
    poly = Polygon(coords)
    areas = []
    subcoords = []
    for t in triangulate(poly):
        areas.append(t.area)
        subcoords.append(t.exterior.coords.xy)

    # Distribute the points proportionally among the different triangles
    cumarea = np.cumsum(areas)
    triangle_num = np.searchsorted(cumarea / cumarea[-1], np.random.rand(num))

    # Sample within the triangles
    s, t = _unit_triangle_sample(num)
    (x1, y1), (x2, y2), (x3, y3), _ = np.array(subcoords)[triangle_num].T
    x = (x2 - x1) * s + (x3 - x1) * t + x1
    y = (y2 - y1) * s + (y3 - y1) * t + y1
    return x, y


# noinspection PyPackageRequirements
def get_concave_polygon_sample(coords, num):
    import triangle as tr
    from shapely.geometry import Polygon
    np.random.seed(0)

    # Triangulate the polygon
    sequence = list(range(len(coords)))
    trpoly = dict(vertices=coords,
                  segments=np.array((sequence, sequence[1:] + [0])).T)
    trdata = tr.triangulate(trpoly, 'p')
    subcoords = []
    areas = []
    for tidx in trdata['triangles']:
        t = Polygon(trdata['vertices'][tidx])
        subcoords.append(t.exterior.coords.xy)
        areas.append(t.area)

    # Distribute the points proportionally among the different triangles
    cumarea = np.cumsum(areas)
    triangle_num = np.searchsorted(cumarea / cumarea[-1], np.random.rand(num))

    # Sample within the triangles
    s, t = _unit_triangle_sample(num)
    (x1, y1), (x2, y2), (x3, y3), _ = np.array(subcoords)[triangle_num].T
    x = (x2 - x1) * s + (x3 - x1) * t + x1
    y = (y2 - y1) * s + (y3 - y1) * t + y1
    return x, y
