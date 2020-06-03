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
    r['Z'] = [0] * num
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
    attrs = dict()
    for k, v in attrs_conf.items():
        attrs[k] = [v] * num
    return attrs
