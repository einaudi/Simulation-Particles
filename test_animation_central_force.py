# -*- coding: utf-8 -*-

import numpy as np

from src.sim_particles import Sim, cmap_temperature


def force_central(p, p0, G):

    dr = p0 - p
    dr_norm2 = np.sum(np.power(dr, 2))

    ret = G*dr / np.power(dr_norm2, 1.5)

    return ret

def force_central_limited(p, p0, G, R):

    dr = p0 - p
    dr_norm2 = np.sum(np.power(dr, 2))

    if dr_norm2 < R**2:
        return G/R**2 * dr / np.sqrt(dr_norm2)
    else:
        return G*dr / np.power(dr_norm2, 1.5)

def potential_central(x, y, x0, y0, G):

    dx2 = np.power(x - x0, 2)
    dy2 = np.power(y - y0, 2)
    dr = np.sqrt(dx2 + dy2)

    ret = G / dr

    return ret

# Menu
fps = 60
spf = 20

# g = -3
g = 0

N = 10
r = 0.1

bounds = {
    'xMin': -3,
    'xMax': 3,
    'yMin': -3,
    'yMax': 3
}

vmin = -2.5
vmax = 2.5

p0 = np.array([
    (bounds['xMax'] + bounds['xMin'])/2,
    (bounds['yMax'] + bounds['yMin'])/2
])
G = -0.1

# force = lambda p: force_central(p, p0, G)
# force = lambda p: force_central_limited(p, p0, G, 0.1)
force = lambda x, y: potential_central(x, y, p0[0], p0[1], G)

# Initialisation
sim = Sim(fps, spf, bounds=bounds)

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

# sim.geometry.add_potential_force_func(force)
sim.geometry.add_potential(force)

# Animation
acs = np.zeros((sim.N, 2))
acs[:, 1] = g*np.ones(sim.N)

sim.animate(acs, cmap=cmap_temperature)
