import numpy as np


def make_release(conf):
    num = conf['num']
    release_time = date_range(conf['date'], num)
    lon, lat = conf['location']
    attrs = get_attrs(conf.get('attrs', dict()), num)

    r = dict()
    r['release_time'] = release_time
    r['lon'] = np.repeat(lon, num).tolist()
    r['lat'] = np.repeat(lat, num).tolist()
    r['Z'] = get_depth(conf.get('depth', 0), num)
    r = {**r, **attrs}

    return r


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
