# -*- coding: utf-8 -*-

from copy import copy

import numpy as np

from src.KDTree import KDTree


def norm(v):

    ret = np.sqrt(
        np.sum(
            np.power(v, 2)
        )
    )

    return ret

def check_particle_overlap(p1, p2):

    d = np.power(p2.get_ps() - p1.get_ps(), 2)
    d = np.sum(d)
    d = np.sqrt(d)

    r = p1.radius() + p2.radius()

    if d <= r:
        return True
    else:
        return False

def elastic_collision(p1, p2):

    v2 = p2.get_vs()
    v1 = p1.get_vs()

    n = p2.get_ps() - p1.get_ps()
    n /= norm(n)
    
    t = np.random.random(p1.dim)
    t -= np.dot(t, n) * n
    t /= norm(t)

    v2n = np.dot(v2, n) * n
    v2t = np.dot(v2, t) * t
    v1n = np.dot(v1, n) * n
    v1t = np.dot(v1, t) * t

    # check if particles are moving apart
    if np.dot(v2n - v1n, n) > 0:
        return v1, v2
    else:
        v1n_after = v1n*(p1.mass()-p2.mass()) + 2*p2.mass()*v2n
        v1n_after /= p2.mass()+p1.mass()

        v2n_after = v2n*(p2.mass()-p1.mass()) + 2*p1.mass()*v1n
        v2n_after /= p2.mass()+p1.mass()

        v1 = v1n_after + v1t
        v2 = v2n_after + v2t

        return v1, v2

def general_collision(p1, p2, Cr=1):

    v2 = p2.get_vs()
    v1 = p1.get_vs()

    n = p2.get_ps() - p1.get_ps()
    n /= norm(n)
    
    t = np.random.random(p1.dim)
    t -= np.dot(t, n) * n
    t /= norm(t)

    v2n = np.dot(v2, n) * n
    v2t = np.dot(v2, t) * t
    v1n = np.dot(v1, n) * n
    v1t = np.dot(v1, t) * t

    # check if particles are moving apart
    if np.dot(v2n - v1n, n) > 0:
        return v1, v2
    else:
        v1n_after = Cr*p2.mass()*(v2n-v1n) + p1.mass()*v1n + p2.mass()*v2n
        v1n_after /= p2.mass()+p1.mass()

        v2n_after = Cr*p1.mass()*(v1n-v2n) + p1.mass()*v1n + p2.mass()*v2n
        v2n_after /= p2.mass()+p1.mass()

        v1 = v1n_after + v1t
        v2 = v2n_after + v2t

        return v1, v2



class Particle():

    __slots__ = '_ps', '_vs', '_m', '_r', 'dim'

    def __init__(self, ps, vs, m=1, r=1):

        self._ps = np.array(
            ps,
            dtype=float
        )
        self._vs = np.array(
            vs,
            dtype=float
        )

        self._m = m
        self._r = r

        self.dim = self._ps.size

    def __str__(self):

        ret = 'px = {:5.2f}\tpy = {:5.2f}\tvx = {:5.2f}\tvy = {:5.2f}\tm = {:5.2f}\tr = {:5.2f}'.format(
            *self._ps,
            *self._vs,
            self._m,
            self._r
        )

        return ret

    # private attributes
    def mass(self):

        return copy(self._m)

    def radius(self):

        return copy(self._r)

    def volume(self):

        if self.dim == 2:
            return np.pi*self._r**2
        elif self.dim == 3:
            return 4./3*np.pi*self._r**3

    # get attributes
    def get_ps(self):

        return copy(self._ps)

    def get_vs(self):

        return copy(self._vs)

    def get_energy_kinetic(self):

        ret = np.power(self._vs, 2)
        ret = np.sum(ret)
        ret *= self._m/2

        return ret

    # change arbitrarily attributes
    def change_ps(self, ps):

        self._ps = ps

    def change_vs(self, vs):

        self._vs = vs

    # shift attributes
    def shift_ps(self, dp):

        self._ps += dp

    def shift_vs(self, dv):

        self._vs += dv

    # update dynamics
    def update_ps(self, dt):

        self._ps += self._vs*dt

    def update_vs(self, acs, dt):

        self._vs += acs*dt


class Particles():

    __slots__ = 'particles_list', 'N', 'dim', '_m', '_r', '_detect_collisions'

    def __init__(self, collisions='KDTree'):

        self.particles_list = []

        self.N = 0
        self.dim = 0
        self._m = []
        self._r = []

        # choose collisions detection algorithm
        if collisions == 'KDTree':
            self._detect_collisions = self._detect_collisions_KDTree
        else:
            print('Wrong collision detection algorithm chosen!')
            quit()

    def __str__(self):

        ret = 'Particles:\n'
        for p in self.particles_list:
            ret += str(p) + '\n'

        return ret[:-1]

    def add_particle(self, ps, vs, m=1, r=1):

        self.particles_list.append(
            Particle(ps, vs, m, r)
        )
        self._m.append(m)
        self._r.append(r)

        self.N = len(self.particles_list)
        self.dim = ps.size

    # private attributes
    def mass(self):

        ret = copy(np.array(self._m))

        return ret

    def radius(self):

        ret = copy(np.array(self._r))

        return ret

    def collective_volume(self):

        ret = 0
        for p in self.particles_list:
            ret += p.volume()

        return ret

    # get attributes
    def get_ps(self):

        ret = np.zeros((self.N, 2))

        for i, p in enumerate(self.particles_list):
            ret[i, :] = p.get_ps().reshape((1, 2))

        return ret

    def get_vs(self):

        ret = np.zeros((self.N, 2))

        for i, p in enumerate(self.particles_list):
            ret[i, :] = p.get_vs().reshape((1, 2))

        return ret

    def get_energy_kinetic(self):

        ret = np.zeros(self.N)
        for i, p in enumerate(self.particles_list):
            ret[i] = p.get_energy_kinetic()

        return ret

    def get_temperature(self):

        E = self.get_energy_kinetic()

        return np.average(E)

    # change arbitrarily attributes
    def change_ps(self, ps):

        for i, p in enumerate(self.particles_list):
            p.change_ps(ps[i, :])

    def change_vs(self, vs):

        for i, p in enumerate(self.particles_list):
            p.change_vs(vs[i, :])

    # shift attributes
    def shift_ps(self, dp):

        for i, p in enumerate(self.particles_list):
            p.shift_ps(dp[i, :])

    def shift_vs(self, dv):

        for i, p in enumerate(self.particles_list):
            p.shift_vs(dv[i, :])
    
    # collision handling
    def _detect_collisions_KDTree(self, *args, Cr=1, **kwargs):

        tree = KDTree(self)
        if 'depth' in kwargs.keys():
            depth = kwargs['depth']
        else:
            depth = 1

        for p in self.particles_list:
            nearest_guess = tree.find_nearest(p.get_ps(), depth)
            # print(nearest_guess)
            for pnn in nearest_guess:
                if p == pnn:
                    continue
                if check_particle_overlap(p, pnn):
                    # print('x')
                    # v, vnn = elastic_collision(p, pnn)
                    v, vnn = general_collision(p, pnn, Cr)
                    p.change_vs(v)
                    pnn.change_vs(vnn)

    # dynamics update
    def update_ps(self, dt):

        for p in self.particles_list:
            p.update_ps(dt)

    def update_vs(self, acs, dt):

        for i, p in enumerate(self.particles_list):
            p.update_vs(acs[i, :], dt)

    def update(self, acs, dt, **kwargs):

        self._detect_collisions(**kwargs)

        self.update_vs(acs, dt)
        self.update_ps(dt)


if __name__ == '__main__':

    ps = Particles()
    for i in range(10):
        ps.add_particle(
            ps=np.random.rand(2),
            vs=np.random.rand(2)
        )

    print(ps)