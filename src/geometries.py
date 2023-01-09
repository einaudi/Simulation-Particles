# -*- coding: utf-8 -*-

from copy import copy
import numpy as np


class Box2D():

    def __init__(self, xMin=0, xMax=1, yMin=0, yMax=1):

        self._xMin = xMin
        self._xMax = xMax
        self._yMin = yMin
        self._yMax = yMax

        X = xMax - xMin
        Y = yMax - yMin

        self._xLimMin = xMin - 0.1*np.abs(X)
        self._xLimMax = xMax + 0.1*np.abs(X)
        self._yLimMin = yMin - 0.1*np.abs(Y)
        self._yLimMax = yMax + 0.1*np.abs(Y)

        self._dx = (self._xMax - self._xMin) / 1e4
        self._dy = (self._yMax - self._yMin) / 1e4

        self._U = None

    # Geometry parameters
    def get_volume(self):

        V = (self._xMax - self._xMin) * (self._yMax - self._yMin)

        return V

    def get_area(self):

        Ax = self._xMax - self._xMin
        Ay = self._yMax - self._yMin

        return Ax, Ay

    def get_limits(self):

        return (copy(self._xLimMin), copy(self._xLimMax)), (copy(self._yLimMin), copy(self._yLimMax))

    # Walls collisions and momentum
    def detect_collision_wall(self, particles_list):

        pressure_x = 0
        pressure_y = 0

        for p in particles_list:
            ps = p.get_ps()
            r = p.radius()
            vs = p.get_vs()

            if (ps[0] - r < self._xMin and vs[0] < 0) or (ps[0] + r > self._xMax and vs[0] > 0):
                vs[0] = -vs[0]
                pressure_x += 2*np.abs(vs[0])*p.mass()
            if (ps[1] - r < self._yMin and vs[1] < 0) or (ps[1] + r > self._yMax and vs[1] > 0):
                vs[1] = -vs[1]
                pressure_y += 2*np.abs(vs[0])*p.mass()

            p.change_vs(vs)

        Ax, Ay = self.get_area()

        return pressure_x/Ax + pressure_y/Ay

    # Potential
    def add_potential(self, U_func):

        self._U = U_func

    def force_potential(self, particle, **kwargs):

        if self._U is None:
            return np.zeros(particle.dim)
        else:
            x, y = particle.get_ps()
            U_x = (self._U(x+self._dx, y) - self._U(x-self._dx, y)) / (2*self._dx)
            U_y = (self._U(x, y+self._dy) - self._U(x, y-self._dy)) / (2*self._dy)

            return -np.array([U_x, U_y])

    def hasPotential(self):

        if self._U is None:
            return False
        else:
            return True

    # Plotting
    def plot_boundaries(self, ax, c='white'):

        ax.plot(
            [self._xMin, self._xMax],
            [self._yMin, self._yMin],
            c=c
        )
        ax.plot(
            [self._xMin, self._xMax],
            [self._yMax, self._yMax],
            c=c
        )
        ax.plot(
            [self._xMin, self._xMin],
            [self._yMin, self._yMax],
            c=c
        )
        ax.plot(
            [self._xMax, self._xMax],
            [self._yMin, self._yMax],
            c=c
        )

    def plot_set_axis_limits(self, ax):

        ax.set_xlim(self._xLimMin, self._xLimMax)
        ax.set_ylim(self._yLimMin, self._yLimMax)


if __name__ == '__main__':

    from utils import save_csv
    from alive_progress import alive_bar

    T = 10
    fps = 60
    dt = 1/fps
    NT = T*fps

    N = 30
    r = 0.03

    x1 = 0
    x2 = 5
    y1 = 0
    y2 = 5

    ps = np.zeros((NT, N, 2))
    vs = np.zeros((NT, N, 2))

    sim = Box2D(x1, x2, y1, y2, dt)
    # sim.add_particle(0, 0, 1, 0.5, r=0.1)
    for i in range(N):
        ps_particle = np.random.rand(2)
        ps_particle[0] = ps_particle[0]*(x2-x1)+x1
        ps_particle[1] = ps_particle[1]*(y2-y1)+y1
        sim.add_particle(
            ps_particle,
            vs=np.random.rand(2),
            m=1,
            r=r
        )

    dataX = {}
    dataY = {}
    dataVX = {}
    dataVY = {}
    with alive_bar(NT) as bar:
        for i in range(NT):
            ps = sim.get_ps().reshape((N, 2))
            vs = sim.get_vs().reshape((N, 2))
            dataX['{}'.format(i)] = ps[:, 0]
            dataY['{}'.format(i)] = ps[:, 1]
            dataVX['{}'.format(i)] = vs[:, 0]
            dataVY['{}'.format(i)] = vs[:, 1]

            # sim.update(np.array([[0,-0.5]]))
            sim.sim_step()

            bar()

    save_csv(
        dataX,
        './data/test/px.csv',
        {
            'dt': dt,
            'x1': x1,
            'x2': x2,
            'y1': y1,
            'y2': y2
        },
        index=False
    )    
    save_csv(
        dataY,
        './data/test/py.csv',
        {
            'dt': dt,
            'x1': x1,
            'x2': x2,
            'y1': y1,
            'y2': y2
        },
        index=False
    )    
    save_csv(
        dataVX,
        './data/test/vx.csv',
        {
            'dt': dt,
            'x1': x1,
            'x2': x2,
            'y1': y1,
            'y2': y2
        },
        index=False
    )
    save_csv(
        dataVY,
        './data/test/vy.csv',
        {
            'dt': dt,
            'x1': x1,
            'x2': x2,
            'y1': y1,
            'y2': y2
        },
        index=False
    )