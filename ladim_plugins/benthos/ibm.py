import numpy as np


class IBM:
    def __init__(self, config):
        # Submodules
        submodules = {
            name: load_submodule(conf, self)
            for name, conf in config['ibm']['submodules'].items()
        }
        self.submodules = submodules

        # Store time step value to calculate age
        self.dt = config['dt']

        # Reference to other modules
        self.grid = None
        self.forcing = None
        self.state = None

    def update_ibm(self, grid, state, forcing):
        self.grid = grid
        self.forcing = forcing
        self.state = state

        for s in self.submodules.values():
            s.update()


def load_submodule(config, parent):
    import importlib
    package, classname = config['ibm_class'].rsplit('.', 1)
    ibm_class = getattr(importlib.import_module(package), classname)
    return ibm_class(config, parent)


class SubState:
    def __init__(self, parent_state, ptype):
        self._parent_state = parent_state
        self._idx = (parent_state.ptype == ptype)
        self._vars = {}

    def __getattr__(self, item):
        if item not in self._vars:
            self._vars[item] = self._parent_state[item][self._idx]
        return self._vars[item]

    def save_to_parent(self):
        for k, v in self._vars.items():
            self._parent_state[k][self._idx] = self._vars[k]


class Capitella:
    def __init__(self, config, parent):
        # Particle type id
        self.ptype = config['ptype']

        # Reference to other modules
        self.parent = parent
        self.state = None

    def update(self):
        self.state = SubState(self.parent.state, self.ptype)
        self.state.save_to_parent()


class POM:
    def __init__(self, config, parent):
        # Particle type id
        self.ptype = config['ptype']

        # Time before a particle is taken out of the simulation [seconds]
        self.lifespan = config['lifespan']

        # Vertical mixing [m*2/s]
        self.D = config['vertical_mixing']  # 0.001 m2/s -- 0.01 m2/s (?)
        self.taucrit = config.get('taucrit', None)
        self.vertical_diffusion = self.D > 0

        # Reference to other modules
        self.parent = parent
        self.state = None

    def update(self):
        self.state = SubState(self.parent.state, self.ptype)

        self.resuspend()
        self.diffuse()
        self.sink()
        self.bury()
        self.kill_old()

        self.state.save_to_parent()

    def resuspend(self):
        if self.taucrit is None:
            return

        ustar = self.shear_velocity_btm()
        tau = shear_stress_btm(ustar)
        resusp = tau > self.taucrit
        self.state.active[resusp] = True

    def bury(self):
        grid = self.parent.grid
        state = self.state
        a = state.active != 0
        X, Y, Z = state.X[a], state.Y[a], state.Z[a]

        # Define which particles have settled to the bottom and which have not
        H = grid.sample_depth(X, Y)  # Water depth
        at_seabed = Z > H
        Z[at_seabed] = H[at_seabed]

        # Store new vertical position
        state.Z[a] = Z
        state.active[a] = ~at_seabed

    def diffuse(self):
        # Get parameters
        state = self.state
        a = state.active != 0
        x, y, z = state.X[a], state.Y[a], state.Z[a]
        h = self.parent.grid.sample_depth(x, y)

        # Diffusion
        dt = self.parent.dt
        b0 = np.sqrt(2 * self.D)
        dw = np.random.randn(z.size).reshape(z.shape) * np.sqrt(dt)
        z1 = z + b0 * dw

        # Reflexive boundary conditions
        z1[z1 < 0] *= -1  # Surface
        below_seabed = z1 > h
        z1[below_seabed] = 2*h[below_seabed] - z1[below_seabed]

        # Store new vertical position
        state.Z[a] = z1

    def sink(self):
        # Get parameters
        state = self.state
        a = state.active != 0
        z = state.Z[a]
        w = state.sink_vel[a]  # Sink velocity

        # Euler scheme, no boundary conditions
        state.Z[a] = z + self.parent.dt * w

    def kill_old(self):
        state = self.state
        state.age[:] += self.parent.dt
        state.alive[:] = state.alive & (state.age <= self.lifespan)

    def shear_velocity_btm(self):
        # Calculates bottom shear velocity from last computational layer
        # velocity
        # returns: Ustar at bottom cell
        x = self.state.X
        y = self.state.Y
        h = self.parent.grid.sample_depth(x, y)

        u_btm, v_btm = self.parent.forcing.velocity(x, y, h, tstep=0)
        U2 = u_btm*u_btm + v_btm*v_btm
        c = 0.003
        return np.sqrt(c * U2)


def shear_stress_btm(ustar):
    rho = 1000
    return ustar * ustar * rho
