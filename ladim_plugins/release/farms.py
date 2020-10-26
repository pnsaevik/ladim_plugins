def polygon(loknr):
    import re
    import numpy as np
    import requests

    wfs_url = 'https://ogc.fiskeridir.no/wfs.ashx'
    payload = dict(
        service='WFS',
        version='2.0.0',
        request='GetFeature',
        typeName='layer_203',
        maxFeatures=5000000,
        srsName='EPSG:4258'
    )

    r = requests.get(wfs_url, params=payload)
    members = re.findall(r'<wfs:member>(.*?)</wfs:member>', r.text, re.DOTALL)
    member = next(m for m in members if f'<ms:loknr>{loknr}</ms:loknr>' in m)
    pos_list = re.search(r'<gml:posList.*?>(.*?)</gml:posList>', member,
                         re.DOTALL).groups()[0]
    lat, lon = np.array(pos_list.strip().split(" ")).astype('float').reshape(
        (-1, 2)).T
    return lon[:-1], lat[:-1]


def location(loknr):
    import re
    import numpy as np
    import requests

    wfs_url = 'https://ogc.fiskeridir.no/wfs.ashx'
    payload = dict(
        service='WFS',
        version='2.0.0',
        request='GetFeature',
        typeName='layer_262',
        maxFeatures=5000000,
        srsName='EPSG:4258'
    )

    r = requests.get(wfs_url, params=payload)
    members = re.findall(r'<wfs:member>(.*?)</wfs:member>', r.text, re.DOTALL)
    member = next(m for m in members if f'<ms:loknr>{loknr}</ms:loknr>' in m)
    pos_list = re.search(r'<gml:pos.*?>(.*?)</gml:pos>', member,
                         re.DOTALL).groups()[0]
    lat, lon = np.array(pos_list.strip().split(" ")).astype('float')
    return lon, lat
