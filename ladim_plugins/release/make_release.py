import numpy as np


def make_release(conf):
    start_date = np.datetime64(conf['date'])
    num = conf['num']
    lon, lat = conf['location']

    r = dict()
    r['release_time'] = np.repeat(start_date, num)
    r['lon'] = np.repeat(lon, num)
    r['lat'] = np.repeat(lat, num)

    return r
