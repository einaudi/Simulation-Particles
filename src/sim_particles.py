# -*- coding: utf-8 -*-

from copy import copy

from src.particles import Particles
from src.geometries import *

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

    def __init__(self, fps, spf=1, geometry='box', bounds={}, collisions='KDTree'):

        print('Initialising simulation...')

        super().__init__(collisions=collisions)
       
        if geometry == 'box':
            self.geometry = Box2D(**bounds)
        else:
            print('Choose proper geometry!')
            quit()

        self.fps = fps  # frames per second
        self.spf = spf  # steps per frame

        self.dt = 1/fps/spf  # s
        self.bounds = bounds

        # Kinetic energy bounds for coloring
        self._Ekin_min = 100
        self._Ekin_max = 0

        self.pressure = []
        self.volume = []
        self.temperature = []

        print('Simulation initialised')

    def sim_step_single(self, acs=None, depth=1, **kwargs):

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

        self.update(acs, self.dt, depth=depth, **kwargs)

        return pressure

    def sim_step_multi(self, acs=None, pVT=False, depth=1, **kwargs):

        pressure = 0

        for _ in range(self.spf):
            pressure += self.sim_step_single(acs, depth, **kwargs)

        if pVT:
            self.pressure.append(pressure/(self.dt*self.spf))
            self.volume.append(self.geometry.get_volume())
            self.temperature.append(self.get_temperature())

    def _get_markersizes(self, ax, fig):

        xLim, _ = self.geometry.get_limits()

        self.geometry.plot_set_axis_limits(ax)
        s = (ax.get_window_extent().width  / (xLim[1]-xLim[0]) * 72./fig.dpi)
        s *= self.radius()
        s = np.power(s, 2)

        return s

    # Animation
    def _animation_base_function(self, axParticles, acs, depth, s, cmap, pVT=False, **kwargs):

        self.sim_step_multi(acs=acs, depth=depth, pVT=pVT, **kwargs)

        points = self.get_ps()

        # Kinetic energy
        E_kin = self.get_energy_kinetic()
        E_min = np.amin(E_kin)
        E_max = np.amax(E_kin)
        if self._Ekin_min > E_min:
            self._Ekin_min = E_min
        if self._Ekin_max < E_max:
            self._Ekin_max = E_max

        E_kin -= self._Ekin_min
        E_kin /= self._Ekin_max

        axParticles.clear()
        self.geometry.plot_set_axis_limits(axParticles)
        self.geometry.plot_boundaries(axParticles)

        axParticles.scatter(
            points[:, 0],
            points[:, 1],
            c=E_kin,
            s=s,
            cmap=cmap
        )

    def animate(self, acs=None, cmap='bwr', interval=1, dpi=150, figsize=(4, 4)):

        print('Total number of particles: ', self.N)
        print('Particle packing: {:.4f} %'.format(self.collective_volume()/self.geometry.get_volume()*100))
        print('Running simulation...')

        fig = plt.figure(facecolor='black', figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)
        ax.axis('equal')
        ax.set_facecolor('black')

        xCenter = (self.bounds['xMax'] + self.bounds['xMin'])/2
        yCenter = (self.bounds['yMax'] + self.bounds['yMin'])/2

        # KDTree depth calculation
        depth = np.log2(self.N) - 4
        if depth < 1:
            depth = 1

        # Markersizes with respect to axis
        s = self._get_markersizes(ax, fig)

        def animation(i):
            self._animation_base_function(ax, acs, depth, s, cmap)
            
            ax.scatter(
                xCenter,
                yCenter,
                c='white'
            )

        anim = FuncAnimation(fig, animation, interval=interval)
        plt.show()

    def animate_pVT(self, acs=None, cmap='bwr', interval=1, dpi=150, figsize=(3, 4), **kwargs):

        print('Total number of particles: ', self.N)
        print('Running simulation...')

        fig = plt.figure(facecolor='black', figsize=figsize, dpi=dpi)

        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
              
        ax = plt.subplot(gs[0])
        ax.axis('equal')
        ax.set_facecolor('black')

        ax_pVT = plt.subplot(gs[1])
        ax_pVT.set_facecolor('black')
        ax_pVT.spines['bottom'].set_color('white')
        ax_pVT.spines['top'].set_color('white') 
        ax_pVT.spines['right'].set_color('white')
        ax_pVT.spines['left'].set_color('white')
        ax_pVT.tick_params(axis='x', colors='white')
        ax_pVT.tick_params(axis='y', colors='white')

        xCenter = (self.bounds['xMax'] + self.bounds['xMin'])/2
        yCenter = (self.bounds['yMax'] + self.bounds['yMin'])/2

         # KDTree depth calculation
        depth = np.log2(self.N) - 4
        if depth < 1:
            depth = 1

        # Markersizes with respect to axis
        s = self._get_markersizes(ax, fig)

        def animation(i):
            self._animation_base_function(ax, acs, depth, s, cmap, pVT=True, **kwargs)
            
            ax.scatter(
                xCenter,
                yCenter,
                c='white'
            )

            while len(self.pressure) > 1000:
                self.pressure = self.pressure[1:]
                self.volume = self.volume[1:]
                self.temperature = self.temperature[1:]

            p_norm = np.array(self.pressure)
            if not np.amax(p_norm) == 0:
                p_norm /= np.amax(p_norm)
            V_norm = np.array(self.volume, dtype=float)
            V_norm /= np.amax(V_norm)
            T_norm = np.array(self.temperature)
            T_norm -= np.amin(T_norm)
            T_norm /= np.amax(T_norm)

            ax_pVT.clear()
            ax_pVT.plot(p_norm, label='p')
            ax_pVT.plot(V_norm, label='V')
            ax_pVT.plot(T_norm, label='T')
            # ax_pVT.legend(loc=0)

        anim = FuncAnimation(fig, animation, interval=interval)
        plt.show()

    def animate_pVT_compress(self, acs=None, cmap='bwr', interval=1, dpi=150, figsize=(3, 4), compression_func=None):

        print('Total number of particles: ', self.N)
        print('Running simulation...')

        fig = plt.figure(facecolor='black', figsize=figsize, dpi=dpi)

        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
              
        ax = plt.subplot(gs[0])
        ax.axis('equal')
        ax.set_facecolor('black')

        ax_pVT = plt.subplot(gs[1])
        ax_pVT.set_facecolor('black')
        ax_pVT.spines['bottom'].set_color('white')
        ax_pVT.spines['top'].set_color('white') 
        ax_pVT.spines['right'].set_color('white')
        ax_pVT.spines['left'].set_color('white')
        ax_pVT.tick_params(axis='x', colors='white')
        ax_pVT.tick_params(axis='y', colors='white')

        xCenter = (self.bounds['xMax'] + self.bounds['xMin'])/2
        yCenter = (self.bounds['yMax'] + self.bounds['yMin'])/2

         # KDTree depth calculation
        depth = np.log2(self.N) - 4
        if depth < 1:
            depth = 1

        # Markersizes with respect to axis
        s = self._get_markersizes(ax, fig)

        def animation(i):
            if compression_func is not None:
                t = i*self.dt
                self.geometry.change_boundaries(*compression_func(t))

            self._animation_base_function(ax, acs, depth, s, cmap, pVT=True)
            
            ax.scatter(
                xCenter,
                yCenter,
                c='white'
            )

            while len(self.pressure) > 1000:
                self.pressure = self.pressure[1:]
                self.volume = self.volume[1:]
                self.temperature = self.temperature[1:]

            p_norm = np.array(self.pressure)
            if not np.amax(p_norm) == 0:
                p_norm /= np.amax(p_norm)
            V_norm = np.array(self.volume, dtype=float)
            V_norm /= np.amax(V_norm)
            T_norm = np.array(self.temperature)
            T_norm -= np.amin(T_norm)
            T_norm /= np.amax(T_norm)

            ax_pVT.clear()
            ax_pVT.plot(p_norm, label='p')
            ax_pVT.plot(V_norm, label='V')
            ax_pVT.plot(T_norm, label='T')
            # ax_pVT.legend(loc=0)

        anim = FuncAnimation(fig, animation, interval=interval)
        plt.show()