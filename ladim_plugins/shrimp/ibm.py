import numpy as np


class IBM:

    def __init__(self, config):
        # One entry for each pelagic stage
        self.vertical_mixing = np.array(config['ibm']['vertical_mixing'])  # [m2/s]
        self.vertical_speed = np.array(config['ibm']['vertical_speed'])  # [m/s]

        self.maxdepth_day = np.array(config['ibm']['maxdepth_day'])  # [m]
        self.maxdepth_ngh = np.array(config['ibm']['maxdepth_night'])  # [m]
        self.mindepth_day = np.array(config['ibm']['mindepth_day'])  # [m]
        self.mindepth_ngh = np.array(config['ibm']['mindepth_night'])  # [m]

        self.grid = None
        self.state = None
        self.forcing = None
        self.dt = config['dt']

        # Check if active parameter is in state (raises error if not present)
        if 'active' not in config['ibm']['variables']:
            raise KeyError('In ladim.yaml: Add "active" to ibm.variables list')

    def update_ibm(self, grid, state, forcing):
        self.grid = grid
        self.state = state
        self.forcing = forcing

        self.initialize()
        self.growth()
        self.mixing()
        self.diel_migration()

    def initialize(self):
        # Initialize quantile variable
        q = self.state['depth_quantile']
        is_not_initialized = q == 0
        num = np.count_nonzero(is_not_initialized)
        q[is_not_initialized] = np.random.rand(num)
        self.state['depth_quantile'] = q

        # Initialize stage variable
        s = self.state['stage']
        is_not_initialized = s == 0
        if np.any(is_not_initialized):
            self.state['stage'][is_not_initialized] = 1

    def growth(self):
        # Reference paper: Ouellet and Chabot (2005), doi: 10.1007/s00227-005-1625-6

        # Coefficients are re-fitted using Figure 3 in the paper, and values at stage 6:
        # temp[:] = [3.15, 5.31, 7.67]
        # age[:] = [76.1, 53.6, 43.4]
        # (stage - 1) * (alpha + beta * temp) = age * temp

        alpha = 34.98593627
        beta = 4.12176015

        temp = np.clip(self.state['temp'], 3, 8)

        delta_age = self.dt / 86400
        delta_stage = delta_age * temp / (alpha + beta * temp)

        self.state['age'] += delta_age
        self.state['stage'] += delta_stage
        self.state['stage'] = np.clip(self.state['stage'], 1, 6)  # 6 is the maximum stage
        self.state['active'] = self.state['stage'] < 6  # Stage 6 does not move with the currents

        # Compute lengths according to P. Ouellet and J.-P. Allard (2006)
        # doi: 10.1111/j.1365-2419.2005.00394.x
        # length units: mm
        tab_len = [6.371, 7.480, 9.144, 11.433, 12.088, 13.175]
        tab_stg = [1, 2, 3, 4, 5, 6]
        self.state['length'] = np.interp(self.state['stage'], tab_stg, tab_len)

    def mixing(self):
        int_stage = np.minimum(5, np.int32(self.state['stage'])) - 1
        vertmix = self.vertical_mixing[int_stage]

        z = self.state['Z']
        dw = np.random.normal(size=len(z))
        dz = np.sqrt(2 * vertmix * self.dt) * dw
        z += dz
        z[z < 0] *= -1  # Reflective boundary at surface
        self.state['Z'] = z

    def diel_migration(self):
        # Extract state parameters
        if hasattr(self.state, 'timestamp'):
            time = self.state.timestamp
        else:
            time = self.state['time']
        x = self.state['X']
        y = self.state['Y']
        z = self.state['Z']
        q = self.state['depth_quantile']

        # Select parameters based on stage
        int_stage = np.minimum(5, np.int32(self.state['stage'])) - 1
        speed = self.vertical_speed[int_stage]
        maxdepth_day = self.maxdepth_day[int_stage]
        maxdepth_ngh = self.maxdepth_ngh[int_stage]
        mindepth_day = self.mindepth_day[int_stage]
        mindepth_ngh = self.mindepth_ngh[int_stage]

        # Find preferred depth
        lon, lat = self.grid.lonlat(x, y)
        is_day = sunheight(time, lon, lat) > 0
        maxdepth = np.where(is_day, maxdepth_day, maxdepth_ngh)
        mindepth = np.where(is_day, mindepth_day, mindepth_ngh)
        preferred_depth = mindepth + (maxdepth - mindepth) * q

        # Swim towards preferred depth
        # speed_sign = np.zeros(len(z))  # Zero if within preferred range
        # speed_sign[z > maxdepth] = -1    # Upwards if too deep
        # speed_sign[z < mindepth] = 1     # Downwards if too shallow
        speed_sign = np.sign(preferred_depth - z)  # Positive (downwards) if too shallow
        z += self.dt * speed * speed_sign

        self.state['Z'] = z


def sunheight(time, lon, lat):
    RAD_PER_DEG = np.pi / 180.0
    DEG_PER_RAD = 180 / np.pi

    dtime = np.datetime64(time).astype(object)
    lon = np.array(lon)
    lat = np.array(lat)

    time_tuple = dtime.timetuple()
    # day of year, original does not consider leap years
    yday = time_tuple.tm_yday
    # hours in UTC (as output from oceanographic model)
    hours = time_tuple.tm_hour

    phi = lat * RAD_PER_DEG

    # Compute declineation = delta
    a0 = 0.3979
    a1 = 0.9856 * RAD_PER_DEG  # day-1
    a2 = 1.9171 * RAD_PER_DEG
    a3 = 0.98112
    sindelta = a0 * np.sin(a1 * (yday - 80) + a2 * (np.sin(a1 * yday) - a3))
    cosdelta = (1 - sindelta ** 2) ** 0.5

    # True Sun Time [degrees](=0 with sun in North, 15 deg/hour
    # b0 = 0.4083
    # b1 = 1.7958
    # b2 = 2.4875
    # b3 = 1.0712 * rad   # day-1
    # TST = (hours*15 + lon - b0*np.cos(a1*(yday-80)) -
    #        b1*np.cos(a1*(yday-80)) + b2*np.sin(b3*(yday-80)))

    # TST = 15 * hours  # Recover values from the fortran code

    # Simplified formula
    # correct at spring equinox (yday=80) neglecting +/- 3 deg = 12 min
    TST = hours * 15 + lon

    # Sun height  [degrees]
    # sinheight = sindelta*sin(phi) - cosdelta*cos(phi)*cos(15*hours*rad)
    sinheight = sindelta * np.sin(phi) - cosdelta * np.cos(phi) * np.cos(TST * RAD_PER_DEG)
    height = np.arcsin(sinheight) * DEG_PER_RAD

    return height
