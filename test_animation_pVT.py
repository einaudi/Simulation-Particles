# -*- coding: utf-8 -*-

import numpy as np

from src.sim_particles import Sim, cmap_temperature
   

# Menu
fps = 60
spf = 1

# g = -3
g = 0

Cr = 1

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

# Initialisation

sim = Sim(fps, spf, bounds=bounds)

for i in range(N):
    ps_particle = np.random.rand(2)
    ps_particle[0] = ps_particle[0]*(0.5*bounds['xMax']-bounds['xMin'])+bounds['xMin']
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

sim.animate_blit_pVT(acs, cmap=cmap_temperature, Cr=Cr, interval=1)
