import pandas as pd
import yaml
from ..release import make_release as mkrl


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
        config = [config]

    config = [convert_single_conf(**c) for c in config]
    return pd.DataFrame(mkrl(config, fname))


def convert_single_conf(location, depth, release_time, num_particles, group_id):
    if 'width' in location:
        w = location['width'] * 0.5
        location = dict(
            center=[location['lon'], location['lat']],
            offset=[[-w, w, w, -w], [-w, -w, w, w]],
        )
    elif 'file' in location:
        location = location['file']
    else:
        location = [location['lon'], location['lat']]

    return dict(
        date=release_time,
        num=num_particles,
        depth=depth,
        location=location,
        attrs=dict(group_id=group_id),
    )
