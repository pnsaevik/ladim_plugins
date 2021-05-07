import numpy as np
from ladim_plugins.utils import sunheight


class IBM:

    def __init__(self, config):
        stage_duration = np.array(config['ibm']['stage_duration'])  # [days]
        self.devel_rate = 1 / (stage_duration * 24 * 60 * 60)  # [s^-1]

        self.vertical_mixing = np.array(config['ibm']['vertical_mixing'])  # [m2/s]
        self.vertical_speed = np.array(config['ibm']['vertical_speed'])  # [m/s]

        self.maxdepth_day = np.array(config['ibm']['maxdepth_day'])  # [m]
        self.maxdepth_ngh = np.array(config['ibm']['maxdepth_night'])  # [m]
        self.mindepth_day = np.array(config['ibm']['maxdepth_day'])  # [m]
        self.mindepth_ngh = np.array(config['ibm']['maxdepth_night'])  # [m]

        self.dt = config['dt']

    def update_ibm(self, grid, state, forcing):
        # Select parameters based on stage
        int_stage = np.minimum(5, np.int32(state['stage'])) - 1
        devel_rate = self.devel_rate[int_stage]
        vertical_mixing = self.vertical_mixing[int_stage]
        vertical_speed = self.vertical_speed[int_stage]
        maxdepth_day = self.maxdepth_day[int_stage]
        maxdepth_ngh = self.maxdepth_ngh[int_stage]
        mindepth_day = self.mindepth_day[int_stage]
        mindepth_ngh = self.mindepth_ngh[int_stage]

        # --- Larval development ---
        state['age'] += self.dt
        state['stage'] += self.dt * devel_rate
        state['stage'] = np.minimum(state['stage'], 6)  # 6 is the maximum stage
        state['active'] = state['stage'] < 6  # Stage 6 does not move with the currents

        # --- Vertical random migration / turbulent mixing ---
        dw = np.random.normal(size=len(state.X))
        dz = np.sqrt(2 * vertical_mixing * self.dt) * dw
        state['Z'] += dz
        state['Z'][state['Z'] < 0] *= -1  # Reflective boundary at surface

        # --- Diel migration ---
        lon, lat = grid.lonlat(state.X, state.Y)
        is_day = sunheight(state.timestamp, lon, lat) > 0
        maxdepth = np.where(is_day, maxdepth_day, maxdepth_ngh)
        mindepth = np.where(is_day, mindepth_day, mindepth_ngh)

        vspeed_sign = np.zeros(len(state.X))  # Zero if within preferred range
        vspeed_sign[state.Z > maxdepth] = -1    # Upwards if too deep
        vspeed_sign[state.Z < mindepth] = 1     # Downwards if too shallow
        state['Z'] += self.dt * vertical_speed * vspeed_sign
