def to_sqlite(dset, con):
    cur = con.cursor()
    cur.execute("CREATE TABLE particle(pid)")
    cur.execute("CREATE TABLE particle_instance(pid)")
