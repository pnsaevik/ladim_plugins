import pandas as pd


def main(config, fname=None):
    # Check if input argument is yaml file
    try:
        with open(config, encoding='utf8') as config_file:
            import yaml
            config = yaml.safe_load(config_file)
    except TypeError:
        pass
    except OSError:
        pass

    if isinstance(config, dict):
        config = [config]

    results = [apply_single_conf(**c) for c in config]
    frame = pd.concat(results)

    if fname:
        frame.to_csv(fname, sep="\t", header=False, index=False)

    return frame


def apply_single_conf(
    location=None, depth=0, start_time='2000-01-01', stop_time='2000-01-01',
    num_particles=0, group_id=0,
):
    # Handle default arguments
    if location is None:
        location = dict(lat=0., lon=0.)

    return pd.DataFrame(
        dict(
            mycol=[1, 2, 3],
            mynextcol=[4, 5, 6],
        )
    )


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: make_release <config.yaml> <out.rls>')
    elif len(sys.argv) == 2:
        out = main(sys.argv[1])
        print(out)
    else:
        main(sys.argv[1], sys.argv[2])
