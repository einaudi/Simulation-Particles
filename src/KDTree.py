# -*- coding: utf-8 -*-

import numpy as np


class KDNode():

    __slots__ = 'splitAxis', 'splitValue', 'particles', 'left', 'right'

    def __init__(self, splitAxis, splitValue, particles, left, right):

        self.splitAxis = splitAxis
        self.splitValue = splitValue
        self.particles = particles
        self.left = left
        self.right = right

    def __str__(self):

        ret = ''
        for particle in self.particles:
            ret += str(particle) + '\n'

        return ret[:-1]

    def get_splits(self, splitsDict={}):

        if self.splitAxis in splitsDict.keys():
            splitsDict[self.splitAxis].append(self.splitValue)
        else:
            splitsDict[self.splitAxis] = [self.splitValue]

        if self.left is not None:
            self.left.get_splits(splitsDict)
        if self.right is not None:
            self.right.get_splits(splitsDict)

        return splitsDict

    def isLeaf(self):

        if self.left is None or self.right is None:
            return True
        else:
            return False


class KDTree():

    __slots__ = '_root', 'dim'

    def __init__(self, particles):

        def nk2(splitAxis, subparticles):

            if not subparticles:
                return None

            # split particles set at the median
            subparticles.sort(key=lambda p: p.get_ps()[splitAxis])
            m = len(subparticles) // 2
            splitParticle = subparticles[m]

            # cycle through dimensions
            splitNext = (splitAxis + 1) % splitParticle.dim

            # create additional nodes if required
            return KDNode(
                splitAxis,
                splitParticle.get_ps()[splitAxis],
                subparticles,
                nk2(
                    splitNext,
                    subparticles[:m]
                ),
                nk2(
                    splitNext,
                    subparticles[m+1:]
                )
            )

        self._root = nk2(0, particles.particles_list)
        self.dim = particles.particles_list[0].dim

    def get_splits(self):

        return self._root.get_splits()

    def find_node(self, point):

        node = self._root
        axis = 0

        while not node.isLeaf():
            if point[axis] < node.splitValue:
                node = node.left
            else:
                node = node.right

            axis = (axis + 1) % self.dim

        return node.particles[0]

    def find_nearest(self, point, depth=1):

        node = self._root
        axis = 0
        i = 0

        while i < depth:
            if point[axis] < node.splitValue:
                node = node.left
            else:
                node = node.right

            axis = (axis + 1) % self.dim
            i += 1

            if node.isLeaf():
                return node
        
        return node.particles


if __name__ == '__main__':

    import src.particles as particles
    import matplotlib.pyplot as plt

    ps = particles.Particles()
    for i in range(11):
        ps.add_particle(
            px=np.random.rand(),
            py=np.random.rand(),
            vx=np.random.rand(),
            vy=np.random.rand()
        )

    points = ps.get_ps()

    tree = KDTree(ps)
    
    point = np.array([0.5, 0.5])
    nearest_points = tree.find_nearest(point, 2)
    print(nearest_points)

    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.scatter(
        points[:,0],
        points[:,1]
    )
    for key, value in tree.get_splits().items():
        for v in value:
            if key == 0:
                ax.axvline(v)
            elif key == 1:
                ax.axhline(v)

    plt.show()


