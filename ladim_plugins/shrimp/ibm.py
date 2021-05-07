import numpy as np


class IBM:

    def __init__(self, config):
        stage_duration = config['ibm'].get('stage_duration', 12)  # [days]
        self.devel_rate = 1 / (stage_duration * 24 * 60 * 60)  # [s^-1]

        self.vertical_mixing = config['ibm'].get('vertical_mixing', 0)  # [m2/s]

        self.dt = config['dt']

    def update_ibm(self, grid, state, forcing):
        # --- Larval development ---
        state['age'] += self.dt
        state['stage'] += self.dt * self.devel_rate
        state['stage'] = np.maximum(state['stage'], 6)  # 6 is the maximum stage
        state['active'] = state['stage'] < 6  # Stage 6 does not move with the currents

        # --- Vertical turbulent mixing ---
        if self.vertical_mixing:
            dw = np.random.normal(size=len(state.X))
            dz = np.sqrt(2 * self.vertical_mixing * self.dt) * dw
            state['Z'] += dz
