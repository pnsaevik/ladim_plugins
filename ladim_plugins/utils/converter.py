def to_sqlite(dset, con):
    cur = con.cursor()

    particle_cols = [k for k, v in dset.variables.items() if v.dims == ('particle', )]
    cmd = (
        "CREATE TABLE particle ("
        + "".join([f'{p} REAL NOT NULL,' for p in particle_cols])
        + "pid INTEGER PRIMARY KEY);"
    )
    cur.execute(cmd)

    instance_cols = [k for k, v in dset.variables.items() if v.dims == ('particle_instance', )]
    cmd = (
        "CREATE TABLE particle_instance ("
        + "".join([f'{c} REAL NOT NULL,' for c in instance_cols])
        + "time REAL NOT NULL);"
    )
    cur.execute(cmd)
