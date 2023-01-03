# -*- coding: utf-8 -*-

from copy import copy

from src.particles import Particles
from src.geometries import *

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

blue = (65, 105, 225)
red = (199, 1, 57)

def cmap_linear(color1, color2, name, N=256):    

    rgb = np.zeros((N, 3))
    rgb[:, 0] = np.linspace(0, 1, N)*(color2[0] - color1[0]) + color1[0]
    rgb[:, 1] = np.linspace(0, 1, N)*(color2[1] - color1[1]) + color1[1]
    rgb[:, 2] = np.linspace(0, 1, N)*(color2[2] - color1[2]) + color1[2]

    rgb /= 256

    cmap = ListedColormap(rgb, name=name, N=N)

    return cmap

cmap_temperature = cmap_linear(blue, red, 'temperature')

class Sim(Particles):

    def __init__(self, dt, geometry='box', bounds={}, collisions='KDTree'):

        print('Initialising simulation...')

        super().__init__(collisions=collisions)
       
        if geometry == 'box':
            self.geometry = Box2D(**bounds)
        else:
            print('Choose proper geometry!')
            quit()

        self.dt = dt
        self.bounds = bounds

        self.pressure = []
        self.volume = []
        self.temperature = []

        print('Simulation initialised')

    def sim_step(self, acs=None, pVT=False):

        pressure = self.geometry.detect_collision_wall(self.particles_list)

        if acs is None:
            acs = np.zeros((self.N, self.dim))

        if self.geometry.hasPotential():
            for i, p in enumerate(self.particles_list):
                acs[i,:] += self.geometry.force_potential(p)/p.mass()

        # Fs = self.geometry.force_potential(self.get_ps())
        # print(Fs)
        # acs = Fs
        # acs[:,0] /= self._m
        # acs[:,1] /= self._m

        self.update(acs, self.dt)

        if pVT:
            self.pressure.append(pressure/self.dt)
            self.volume.append(self.geometry.get_volume())
            self.temperature.append(self.get_temperature())

    def animate(self, acs=None, cmap='bwr', interval=1):

        print('Total number of particles: ', self.N)
        print('Running simulation...')

        fig = plt.figure(facecolor='black', figsize=(6,6))
        ax = fig.add_subplot(111)
        ax.axis('equal')
        ax.set_facecolor('black')

        xCenter = (self.bounds['xMax'] + self.bounds['xMin'])/2
        yCenter = (self.bounds['yMax'] + self.bounds['yMin'])/2

        def animation(i):
            self.sim_step(acs=acs)

            points = self.get_ps()
            E_kin = self.get_energy_kinetic()

            ax.clear()
            self.geometry.plot_set_axis_limits(ax)
            self.geometry.plot_boundaries(ax)

            ax.scatter(
                points[:, 0],
                points[:, 1],
                c=E_kin,
                cmap=cmap
            )
            ax.scatter(
                xCenter,
                yCenter,
                c='white'
            )

        anim = FuncAnimation(fig, animation, interval=interval)
        plt.show()

    def animate_pVT(self, acs=None, cmap='bwr', interval=1):

        print('Total number of particles: ', self.N)
        print('Running simulation...')

        fig = plt.figure(facecolor='black', figsize=(6,6))
        ax = fig.add_subplot(211)
        ax.axis('equal')
        ax.set_facecolor('black')

        ax_pVT = fig.add_subplot(212)
        ax_pVT.set_facecolor('black')
        ax_pVT.spines['bottom'].set_color('white')
        ax_pVT.spines['top'].set_color('white') 
        ax_pVT.spines['right'].set_color('white')
        ax_pVT.spines['left'].set_color('white')
        ax_pVT.tick_params(axis='x', colors='white')
        ax_pVT.tick_params(axis='y', colors='white')

        xCenter = (self.bounds['xMax'] + self.bounds['xMin'])/2
        yCenter = (self.bounds['yMax'] + self.bounds['yMin'])/2

        def animation(i):
            self.sim_step(acs=acs, pVT=True)

            points = self.get_ps()
            E_kin = self.get_energy_kinetic()

            ax.clear()
            ax_pVT.clear()
            self.geometry.plot_set_axis_limits(ax)
            self.geometry.plot_boundaries(ax)

            ax.scatter(
                points[:, 0],
                points[:, 1],
                c=E_kin,
                cmap=cmap
            )
            ax.scatter(
                xCenter,
                yCenter,
                c='white'
            )

            while len(self.pressure) > 200:
                self.pressure = self.pressure[1:]
                self.volume = self.volume[1:]
                self.temperature = self.temperature[1:]

            p_norm = np.array(self.pressure)
            p_norm /= np.amax(p_norm)
            V_norm = np.array(self.volume, dtype=float)
            V_norm /= np.amax(V_norm)
            T_norm = np.array(self.temperature)
            T_norm /= np.amax(T_norm)

            ax_pVT.plot(p_norm, label='p')
            ax_pVT.plot(V_norm, label='V')
            ax_pVT.plot(T_norm, label='T')
            ax_pVT.legend(loc=0)

        anim = FuncAnimation(fig, animation, interval=interval)
        plt.show()