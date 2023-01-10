# -*- coding: utf-8 -*-

import numpy as np

from src.sim_particles import Sim, cmap_temperature


def heat(t, geometry, particles, coef, f):

    c = coef*np.sin(2*np.pi*f*t) + 1
    print(c)

    vs = particles.get_vs()
    vs *= c

    particles.change_vs(vs)


# Menu
fps = 60
spf = 2

f = 0.1
heat_coef = 0.005

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

heat_func = lambda t, geometry, particles: heat(t, geometry, particles, heat_coef, f)

# Initialisation

sim = Sim(fps, spf, bounds=bounds)

sim.add_time_dependent_function(heat_func)

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
