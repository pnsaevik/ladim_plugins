import numpy as np


def make_release(conf):
    num = conf['num']
    release_time = date_range(conf['date'], num)
    lon, lat = get_location(conf['location'], num)
    attrs = get_attrs(conf.get('attrs', dict()), num)

    r = dict()
    r['release_time'] = release_time
    r['lon'] = lon
    r['lat'] = lat
    r['Z'] = get_depth(conf.get('depth', 0), num)
    r = {**r, **attrs}

    file = conf.get('file', None)
    if file:
        write_to_file(r, file)

    return r


def get_location(loc_conf, num):
    lon, lat = loc_conf

    if not hasattr(lon, '__len__'):
        return [lon] * num, [lat] * num
    else:
        plat, plon = get_polygon_sample_convex(np.array((lat, lon)).T, num)
        return plon.tolist(), plat.tolist()


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
    if isinstance(v, str):
        import importlib
        mod_name, fn_name = v.rsplit('.', 1)
        mod = importlib.import_module(mod_name)
        v = getattr(mod, fn_name)

    if callable(v):
        v = v(num)

    if not hasattr(v, '__len__'):
        v = [v] * num

    return list(v)


def get_depth(depth_span, num):
    if not hasattr(depth_span, '__len__'):
        depth_span = [depth_span] * 2
    return np.linspace(*depth_span, num=num).tolist()


def write_to_file(r, file):
    import pandas as pd
    df = pd.DataFrame(r)
    df.to_csv(file, header=False, sep='\t', index=False)


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


def is_convex(coords):
    # Compute coord differences
    c = np.concatenate([coords, coords[0:2]])
    v = c[:-1, :] - c[1:, :]

    # Compute sign of the cross product
    sgn = v[:-1, 0] * v[1:, 1] > v[:-1, 1] * v[1:, 0]
    # noinspection PyUnresolvedReferences
    return np.all(sgn == sgn[0])
