# -*- coding: utf-8 -*-

import numpy as np

from src.sim_particles import Sim, cmap_temperature


def compress_cos(t, f, c, geometry, axis='x'):

    bounds = geometry.get_initial_boundaries()

    if axis == 'x':
        xmax = 'xMax'
        xmin = 'xMin'
    else:
        xmax = 'yMax'
        xmin = 'yMin'

    L = bounds[xmax] - bounds[xmin]
    A = L*(1-c)/2
    offset = bounds[xmax] - A

    if axis == 'x':
        ret = [
            bounds['xMin'],
            A*np.cos(2*np.pi*f*t) + offset,
            bounds['yMin'],
            bounds['yMax']
        ]
    else:
        ret = [
            bounds['xMin'],
            bounds['xMax'],
            bounds['yMin'],
            A*np.cos(2*np.pi*f*t) + offset
        ]

    geometry.change_boundaries(*ret)


# Menu
fps = 60
spf = 2

f = 0.05
compression = 0.1

# g = -3
g = 0

N = 100
r = 0.05

bounds = {
    'xMin': 0,
    'xMax': 2.5,
    'yMin': 0,
    'yMax': 2.5
}

vmin = -2.5
vmax = 2.5

compress_func = lambda t, geometry, _: compress_cos(t, f, compression, geometry)

# Initialisation

sim = Sim(fps, spf, bounds=bounds)

sim.add_time_dependent_function(compress_func)

for i in range(N):
    ps_particle = np.random.rand(2)
    ps_particle[0] = ps_particle[0]*(bounds['xMax']-bounds['xMin'])+bounds['xMin']
    ps_particle[1] = ps_particle[1]*(bounds['yMax']-bounds['yMin'])+bounds['yMin']
    sim.add_particle(
        ps_particle,
        vs=np.random.rand(2)*(vmax-vmin)+vmin,
        m=1,
        r=r
    )

# Animation
acs = np.zeros((sim.N, 2))
acs[:, 1] = g*np.ones(sim.N)

sim.animate_pVT(acs, cmap=cmap_temperature)
