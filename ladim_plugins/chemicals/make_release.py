import numpy as np
import pandas as pd
import yaml


def main(config, fname=None):
    # Check if input argument is yaml file
    try:
        with open(config, encoding='utf8') as config_file:
            config = yaml.safe_load(config_file)
    except TypeError:
        pass
    except OSError:
        pass

    if isinstance(config, dict):
        frame = single_config(**config)
    else:
        frames = [single_config(**c) for c in config]
        frame = pd.concat(frames)

    if fname is not None:
        frame.to_csv(fname, sep="\t", header=False, index=False,
                     date_format="%Y-%m-%dT%H:%M:%S")

    return frame


def single_config(location, depth, release_time, num_particles, group_id):
    release = pd.DataFrame(
        columns=['release_time', 'lat', 'lon', 'Z', 'group_id'])

    # Set parameters
    release['lat'], release['lon'] = latlon(location, num_particles)
    release['Z'] = _sample(depth, num_particles)
    release['group_id'] = group_id
    release['release_time'] = _sample(_str2time(release_time), num_particles).astype(str)

    return release


def _str2time(strtime):
    if isinstance(strtime, str):
        return np.datetime64(strtime)
    else:
        return [np.datetime64(t) for t in strtime]


def _sample(value_or_range, n):
    try:
        a, b = value_or_range
    except TypeError:
        return np.broadcast_to(value_or_range, (n, ))

    vals = a + np.arange(n) * ((b - a) / (n - 1))
    np.random.shuffle(vals)
    return vals


def latlon(loc, n):
    if isinstance(loc, dict):
        if 'lat' in loc and 'lon' in loc:
            lat = to_numeric_latlon_seq(loc['lat'])
            lon = to_numeric_latlon_seq(loc['lon'])

        elif 'file' in loc:
            lat, lon = latlon_from_file(loc['file'])

        else:
            raise NotImplementedError()

        # If square
        if 'width' in loc:
            return latlon_from_llw(lat, lon, loc['width'], n)

        # If singular point
        elif np.shape(lat) == ():
            return np.ones(n) * lat, np.ones(n) * lon

        # If polygon
        else:
            return latlon_from_poly(lat, lon, n)


def latlon_from_file(fname):
    import json

    with open(fname, 'r', encoding='utf8') as file:
        data = json.load(file)
    if isinstance(data, dict):
        data = [data]

    def get_points_from_layer(layer):
        feats = layer['features']
        return [p for f in feats for p in get_points_from_feature(f)]

    def get_points_from_feature(feature):
        geom = feature['geometry']
        assert geom['type'].upper() == 'MULTIPOLYGON'
        return [np.array(p[0]) for p in geom['coordinates']]

    points = get_points_from_layer(data[0])
    lon = [p[:, 0] for p in points]
    lat = [p[:, 1] for p in points]

    return lat, lon


def latlon_from_llw(lat, lon, width, n):
    (lat1, lat2), (lon1, lon2) = latlon_square(lat, lon, width)
    return latlon_from_poly(
        [lat1, lat1, lat2, lat2], [lon1, lon2, lon2, lon1], n)


def metric_diff_to_degrees(dx, dy, reference_latitude):
    a = 6378137.0
    b = 6356752.314245
    reflat_rad = reference_latitude * np.pi / 180
    lat_cos = np.cos(reflat_rad)
    lat_sin = np.sin(reflat_rad)

    phi_diff = dy / np.sqrt((a*lat_sin)**2 + (b*lat_cos)**2)
    theta_diff = dx / (a * lat_cos)

    lat_diff = phi_diff * 180 / np.pi
    lon_diff = theta_diff * 180 / np.pi

    return lon_diff, lat_diff


def degree_diff_to_metric(lon_diff, lat_diff, reference_latitude):
    a = 6378137.0
    b = 6356752.314245
    reflat_rad = reference_latitude * np.pi / 180
    lat_cos = np.cos(reflat_rad)
    lat_sin = np.sin(reflat_rad)

    phi_diff = lat_diff * np.pi / 180
    theta_diff = lon_diff * np.pi / 180

    dy = np.sqrt((a * lat_sin) ** 2 + (b * lat_cos) ** 2) * phi_diff
    dx = (a * lat_cos) * theta_diff

    return dx, dy


def latlon_square(lat, lon, width):
    lon_width, lat_width = metric_diff_to_degrees(width, width, lat)
    lat_limits = lat + np.array([-0.5, 0.5]) * lat_width
    lon_limits = lon + np.array([-0.5, 0.5]) * lon_width
    return lat_limits, lon_limits


def latlon_from_poly(lat, lon, n):
    # Make multipolygon if given a single polygon
    if np.shape(lat[0]) == ():
        lat = [lat]
        lon = [lon]

    coords = [np.stack((lat_e, lon_e)).T for lat_e, lon_e in zip(lat, lon)]
    triangles = triangulate_nonconvex_multi(coords)
    return get_polygon_sample_triangles(np.array(triangles), n)


def to_numeric_latlon_seq(lat_or_lon_seq):
    # Find nesting level
    dims = 0
    try:
        if not isinstance(lat_or_lon_seq[0], str):
            dims = 1
        if not isinstance(lat_or_lon_seq[0][0], str):
            dims = 2
    except TypeError:
        pass

    if dims == 0:
        return to_numeric_latlon(lat_or_lon_seq)
    elif dims == 1:
        return np.vectorize(to_numeric_latlon)(lat_or_lon_seq)
    else:  # dims == 2
        fn = np.vectorize(to_numeric_latlon)
        return [fn(val) for val in lat_or_lon_seq]


def to_numeric_latlon(lat_or_lon):
    if isinstance(lat_or_lon, str):
        if '°' in lat_or_lon:
            deg_str, rest = lat_or_lon.replace('"', "''").split('°')
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


def _unit_triangle_sample(num):
    xy = np.random.rand(num*2).reshape((2, -1))
    is_upper_triangle = np.sum(xy, axis=0) > 1
    xy[:, is_upper_triangle] = 1 - xy[:, is_upper_triangle]
    return xy


def triangulate_nonconvex_multi(coords):
    import triangle as tr

    # Build flat list of coordinates
    coords_flat = np.concatenate(coords)

    # Build list of segments for multipolygons
    # Multipolygon [[0, 1, 2, 3], [4, 5, 6]] is encoded as
    # segments = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 4]]
    start = np.arange(len(coords_flat))
    stop = start + 1
    coordpos = np.cumsum([0] + [len(c) for c in coords])
    stop[coordpos[1:] - 1] = coordpos[:-1]
    segments = np.stack((start, stop)).T

    # Triangulate and parse output
    trpoly = dict(vertices=coords_flat, segments=segments)
    trdata = tr.triangulate(trpoly, 'p')
    coords = [trdata['vertices'][tidx] for tidx in trdata['triangles']]
    return np.array(coords)


def triangle_areas(triangles):
    a = triangles[..., 1, :] - triangles[..., 0, :]
    b = triangles[..., 2, :] - triangles[..., 0, :]
    return 0.5 * np.abs(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0])


def get_polygon_sample_triangles(triangles, num):
    np.random.seed(0)

    # Triangulate the polygon
    areas = triangle_areas(triangles)

    # Distribute the points proportionally among the different triangles
    cumarea = np.cumsum(areas)
    triangle_num = np.searchsorted(cumarea / cumarea[-1], np.random.rand(num))

    # Sample within the triangles
    s, t = _unit_triangle_sample(num)
    (x1, x2, x3), (y1, y2, y3) = triangles[triangle_num].T
    x = (x2 - x1) * s + (x3 - x1) * t + x1
    y = (y2 - y1) * s + (y3 - y1) * t + y1
    return x, y


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: make_release <config.yaml> <out.rls>')
    elif len(sys.argv) == 2:
        out = main(sys.argv[1])
        print(out)
    else:
        main(sys.argv[1], sys.argv[2])
