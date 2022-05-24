import numpy as np
import typing


def make_release(config, fname=None):
    config = load_config(config)

    # Set seed if specified
    if 'seed' in config:
        np.random.seed(config['seed'])

    # Create release params
    import pandas as pd
    frames = [pd.DataFrame(make_single_release(c)) for c in config['groups']]
    frame = pd.concat(frames).fillna(0)

    # Sort data frame by release time
    frame = frame.sort_values('date')

    # Make column selection if specified
    if 'columns' in config:
        frame = frame[config['columns']]

    if fname:
        frame.to_csv(fname, sep="\t", header=False, index=False)

    return frame.to_dict(orient='list')


def load_config(config):
    # Is this a file name?
    config_str = None
    if isinstance(config, str):
        with open(config, encoding='utf8') as config_file:
            config_str = config_file.read()

    # Is this a stream?
    elif hasattr(config, 'read') and callable(config.read):
        config_str = config.read()

    if config_str is not None:
        import yaml
        try:
            config = yaml.safe_load(config_str)
        except yaml.YAMLError as e:
            raise ValueError(f'Error parsing yaml file: {e}') from e

    # Is this a list? If so, convert to group format
    if isinstance(config, list):
        config = dict(groups=config)

    # Is this not a dict? If not, it wrong format
    if not isinstance(config, dict):
        raise TypeError(f'Not a valid config format: {type(config)}')

    # Is this a flat format? If so, make a single group
    if 'groups' not in config:
        global_params = ['seed', 'columns']
        new_config = {k: v for k, v in config.items() if k in global_params}
        groups = [{k: v for k, v in config.items() if k not in global_params}]
        new_config['groups'] = groups
        config = new_config

    # Does any of the groups lack necessary parameters?
    necessary = ['date', 'location', 'num']
    not_present = [
        [k for k in necessary if k not in params]
        for params in config['groups']
    ]
    if any(m for m in not_present):
        messages = [
            ", ".join(p) + (
                f" in group {i}" if len(config['groups']) > 1 else '')
            for i, p in enumerate(not_present) if p
        ]
        msg = "\n  and ".join(messages)
        raise ValueError("Missing parameters: " + msg)

    # Are any of the date strings wrongly formatted?
    for g in config['groups']:
        np.array(g['date']).astype('datetime64')

    return config


def make_single_release(conf):
    # Enumerate explicit and implicit attributes
    special_keys = ['num', 'date', 'location', 'attrs']
    attrs_default = dict(depth=0.0)
    attrs_explicit = conf.get('attrs', dict())
    attrs_implicit = {k: v for k, v in conf.items() if k not in special_keys}
    attrs_all = {**attrs_default, **attrs_implicit, **attrs_explicit}

    # Compute parameter values
    num = conf['num']
    release_time = date_range(conf['date'], num)
    loc_attrs = get_location(conf['location'], num)
    attrs = get_attrs(attrs_all, num)

    # Assemble return value
    r = dict(date=release_time)
    return {**r, **loc_attrs, **attrs}


def get_location(loc_conf, num):
    loc_attrs = {}

    # If file name: Assume geojson file
    if isinstance(loc_conf, str):
        with open(loc_conf, 'r', encoding='utf-8') as file:
            lon, lat, loc_attrs = get_location_file(file, num)

    # If file object: Assume geojson file
    elif hasattr(loc_conf, 'read'):
        lon, lat, loc_attrs = get_location_file(loc_conf, num)

    # If dict: Assume center/offset config
    elif isinstance(loc_conf, dict) and 'offset' in loc_conf:
        lon, lat = get_location_offset(loc_conf, num)

    # If list of two elements: Assume this is lon/lat
    else:
        lon_spec, lat_spec = loc_conf

        # If single values, assume this is a point specification
        if not hasattr(lon_spec, '__len__'):
            lon = [lon_spec] * num
            lat = [lat_spec] * num

        # Otherwise, assume this is a polygon specification
        else:
            np_lat, np_lon = latlon_from_poly(lat_spec, lon_spec, num)
            lon = np_lon.tolist()
            lat = np_lat.tolist()

    # Finally, assemble return value
    return {**dict(longitude=lon, latitude=lat), **loc_attrs}


def get_location_offset(loc_conf, num):
    clon, clat = loc_conf['center']
    olon, olat = loc_conf['offset']
    dlon, dlat = metric_diff_to_degrees(olon, olat, clat)
    plon = clon + np.array(dlon)
    plat = clat + np.array(dlat)
    slat, slon = latlon_from_poly(plat, plon, num)
    return slon.tolist(), slat.tolist()


def get_location_file(file, num):
    import json

    data = json.loads(file.read())
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
    plon = [p[:-1, 0] for p in points]
    plat = [p[:-1, 1] for p in points]

    slat, slon = latlon_from_poly(plat, plon, num)
    return slon.tolist(), slat.tolist(), {}


def date_range(date_span, num):
    if isinstance(date_span, str) or not hasattr(date_span, '__len__'):
        date_span = [date_span] * 2

    start, stop = [np.datetime64(d) for d in date_span]
    dt = (stop - start).astype('timedelta64[s]')
    drange = start + (np.arange(num) * dt) / (num - 1)
    return drange.astype(str).tolist()


def get_attrs(attrs_conf, num):
    return {k: get_attr(v, num) for k, v in attrs_conf.items()}


def get_attr(v, num):
    if hasattr(v, '__len__') and len(v) == 2 and num != 2:
        v = dict(distribution='uniform', min=v[0], max=v[1])

    if isinstance(v, str):
        import importlib
        mod_name, fn_name = v.rsplit('.', 1)
        mod = importlib.import_module(mod_name)
        v = getattr(mod, fn_name)

    if callable(v):
        v = v(num)

    if not hasattr(v, '__len__'):
        v = [v] * num

    if isinstance(v, dict) and 'distribution' in v:
        v = get_distribution(v, num)

    return list(v)


def get_distribution(v, num):
    if v['distribution'] == 'uniform':
        return np.random.uniform(v['min'], v['max'], num)

    elif v['distribution'] == 'gaussian':
        r = np.random.normal(v['mean'], v['std'], num)
        minimum = v.get('min', -np.inf)
        maximum = v.get('max', np.inf)
        return np.clip(minimum, maximum, r)
    elif v['distribution'] == 'exponential':
        return np.random.exponential(v['mean'], num)
    elif v['distribution'] == 'piecewise':
        from scipy.interpolate import InterpolatedUnivariateSpline
        knots = np.array(v['knots'])
        cdf = np.array(v['cdf'])
        fn = InterpolatedUnivariateSpline(cdf, knots, k=1)
        return fn(np.random.rand(num))
    else:
        raise ValueError(f'Unknown distribution: {v["distribution"]}')


def get_depth(depth_span, num):
    if not hasattr(depth_span, '__len__'):
        depth_span = [depth_span] * 2
    depth = np.linspace(*depth_span, num=num).tolist()  # type: typing.Any
    np.random.shuffle(depth)
    return depth


# --- Triangulation procedures ---


def _unit_triangle_sample(num):
    xy = np.random.rand(num*2).reshape((2, -1))
    is_upper_triangle = np.sum(xy, axis=0) > 1
    xy[:, is_upper_triangle] = 1 - xy[:, is_upper_triangle]
    return xy


def get_polygon_sample(coords, num):
    if is_convex(coords):
        return get_polygon_sample_convex(coords, num)
    else:
        return get_polygon_sample_nonconvex(coords, num)


def triangulate(coords):
    triangles = []
    for i in range(len(coords) - 2):
        idx = [0, i + 1, i + 2]
        triangles.append(coords[idx])
    return np.array(triangles)


def triangulate_nonconvex(coords):
    import triangle as tr

    # Triangulate the polygon
    sequence = list(range(len(coords)))
    trpoly = dict(vertices=coords,
                  segments=np.array((sequence, sequence[1:] + [0])).T)
    trdata = tr.triangulate(trpoly, 'p')
    coords = [trdata['vertices'][tidx] for tidx in trdata['triangles']]
    return np.array(coords)


def point_inside_polygon(coords):
    # Find a point with minimal xy coordinates. Then the point and its neighbours are
    # in the convex hull.
    i = np.lexsort(coords.T)[0]

    # Find the coordinates of the points
    c1 = coords[i - 1]
    c2 = coords[i]
    c3 = coords[i + 1]

    # Find the difference vectors from the central point
    d21 = c1 - c2
    d23 = c3 - c2

    # Walk a small fraction along each of the difference vectors to find a point close
    # to c2 which is still inside the polygon. If the polygon is convex we can choose
    # to walk any distance up to 0.5.
    return c2 + 1e-7 * d21 + 1e-7 * d23


def triangulate_nonconvex_multi(coords):
    import triangle as tr

    # Build flat list of coordinates
    coords_flat = np.concatenate(coords)
    inside_point = [point_inside_polygon(c) for c in coords]

    # Build list of segments for multipolygons
    # Multipolygon [[0, 1, 2, 3], [4, 5, 6]] is encoded as
    # segments = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 4]]
    start = np.arange(len(coords_flat))
    stop = start + 1
    coordpos = np.cumsum([0] + [len(c) for c in coords])
    stop[coordpos[1:] - 1] = coordpos[:-1]
    segments = np.stack((start, stop)).T
    regions = [[x, y, i + 10, 0] for i, (x, y) in enumerate(inside_point)]

    # Triangulate and parse output
    trpoly = dict(vertices=coords_flat, segments=segments, regions=regions)
    trdata = tr.triangulate(trpoly, 'pA')
    coords = [trdata['vertices'][tidx] for tidx in trdata['triangles']]
    return np.array(coords)


def triangle_areas(triangles):
    a = triangles[..., 1, :] - triangles[..., 0, :]
    b = triangles[..., 2, :] - triangles[..., 0, :]
    return 0.5 * np.abs(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0])


def get_polygon_sample_convex(coords, num):
    triangles = triangulate(coords)
    return get_polygon_sample_triangles(triangles, num)


def get_polygon_sample_nonconvex(coords, num):
    triangles = triangulate_nonconvex(coords)
    return get_polygon_sample_triangles(triangles, num)


def get_polygon_sample_triangles(triangles, num):
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


def is_convex(coords):
    # Compute coord differences
    c = np.concatenate([coords, coords[0:2]])
    v = c[:-1, :] - c[1:, :]

    # Compute sign of the cross product
    sgn = v[:-1, 0] * v[1:, 1] > v[:-1, 1] * v[1:, 0]
    # noinspection PyUnresolvedReferences
    return np.all(sgn == sgn[0])


# --- lat / lon procedures ---

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


def latlon_from_poly(lat, lon, n):
    # Make multipolygon if given a single polygon
    if np.shape(lat[0]) == ():
        lat = [lat]
        lon = [lon]

    coords = [np.stack((lat_e, lon_e)).T for lat_e, lon_e in zip(lat, lon)]
    triangles = triangulate_nonconvex_multi(coords)
    return get_polygon_sample_triangles(np.array(triangles), n)


def main():
    import sys

    if len(sys.argv) < 2:
        print("""
    makrel: MAKe RELease files
    
    Usage: makrel <makrel.yaml> <particles.rls>
    
    Sample makrel.yaml file:
    
    # Required attributes
    num: 5                                      # Number of particles
    date: [2000-01-01 01:00, 2000-02-01 01:00]  # Start and stop dates
    location: [5, 60]                           # Release location (lon, lat)
    depth: [0, 10]                              # Release depth range

    # Additional attributes
    region: 0                                   # Constant-valued attribute
    age: [0, 0, 0, 3, 3]                        # Vector-valued attribute
    id: numpy.arange                            # Function-valued attribute
    weight:                                     # Gaussian distribution
      distribution: gaussian
      mean: 40
      std: 10
    length:                                     # Exponential distribution
      distribution: exponential
      mean: 10
    
    
    # Alternative: Release polygon
    # location: [[5, 6, 6, 5], [60, 60, 61, 61]]
    
    # Alternative: Release polygon from .geojson file  
    # location: area.geojson  
    
    # Alternative: Metric offset from center location
    # location: 
    #   center: [5, 60]
    #   offset: [[-50, 50, 50, -50], [-50, -50, 50, 50]]  # 100m x 100m square
    
    """)
    elif len(sys.argv) == 2:
        import pandas as pd
        out = make_release(sys.argv[1])
        print(pd.DataFrame(out))
    else:
        make_release(sys.argv[1], sys.argv[2])
