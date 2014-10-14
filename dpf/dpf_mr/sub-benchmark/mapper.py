#!/usr/bin/env python

import sys
import numpy as np
import models
import resample


class SubBfMapper(object):
    def __init__(self, modelClass, mapconfigurefile):
        self.model = modelClass
        self.mapconfigurefile = mapconfigurefile

    def map(self, value):
        particle = self.model.propagate(self.step * self.timestep, value)
        weight = self.model.obpdf(self.y, particle) # not multiply by previous weight since resample every iteration
        self.mapCache.append([particle, weight])

    def setup(self):
        self.mapCache = []
        with open(self.mapconfigurefile, 'r') as configure:
            obfilename = configure.readline().rstrip()
            self.timestep = float(configure.readline().rstrip())
            self.step = int(configure.readline().rstrip())
            N_red = int(configure.readline().rstrip())
        self.N_sub = N_red
        n = 0
        with open(obfilename, 'r') as obfile:
            for line in obfile:
                if n < self.step:
                    n += 1
                if n == self.step:
                    self.y = float(line.rstrip())
                    break

    def cleanup(self):
        N_m = len(self.mapCache)
        N_s = N_m / self.N_sub + 1
        for l in range(self.N_sub):
            for j in range(N_s):
                k = np.random.randint(N_m)
                print('%s\t%s' % ('reducer'+str(l), str(self.mapCache[k][0])+' '+str(self.mapCache[k][1])))

    def run(self):
        self.setup()
        for line in sys.stdin:
            val = line.rstrip().split(' ')
            particle = float(val[0])
            self.map(particle)
        self.cleanup()

if __name__ == '__main__':
    maptask = SubBfMapper(models.BenchmarkModel, 'benchmarkmapconf.txt')
    maptask.run()
