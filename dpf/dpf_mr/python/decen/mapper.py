#!/usr/bin/env python

import sys
import numpy as np
import models
import resample

class DeBfMapper(object):
    def __init__(self, modelClass, mapconfigurefile):
        self.model = modelClass
        self.mapconfigurefile = mapconfigurefile

    def map(self, value):
        particle = self.model.propagate(self.step * self.timestep, value)
        weight = self.model.obpdf(self.y, particle)
        self.mapCache.append([particle, weight])

    def setup(self):
        self.mapCache=[]
        self.localnumber = 0
        with open(self.mapconfigurefile, 'r') as configure:
            obfilename = configure.readline().rstrip()
            self.timestep = float(configure.readline().rstrip())
            self.step = int(configure.readline().rstrip())
        n = 0
        with open(obfilename, 'r') as obfile:
            for line in obfile:
                if n < self.step:
                    n += 1
                if n == self.step:
                    self.y = float(line.rstrip())
                    break

    def cleanup(self):
        N_l = len(self.mapCache)
        particles, weights = np.array(zip(*self.mapCache))
        W_l = weights.sum()
        weights /= W_l
        resamplecounts = resample.systematic(N_l, weights)
        weight = W_l/N_l
        for particle, count in zip(particles, resamplecounts):
            n = count
            while n > 0:
                print('%s\t%s' % ('reducer1', str(particle)+' '+str(weight)))
                n -= 1

    def run(self):
        self.setup()
        for line in sys.stdin:
            val = line.rstrip().split(' ')
            particle = float(val[0])
            self.map(particle)
        self.cleanup()

if __name__ == '__main__':
    maptask = DeBfMapper(models.BenchmarkModel, 'benchmarkmapconf.txt')
    maptask.run()
