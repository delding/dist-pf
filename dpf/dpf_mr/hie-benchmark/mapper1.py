#!/usr/bin/env python

import sys
import numpy as np
import models
import resample

class HieBfMapper1(object):
    def __init__(self, modelClass, mapconfigurefile):
        self.model = modelClass
        self.mapconfigurefile = mapconfigurefile

    def map(self, value):
        particle = self.model.propagate(self.step * self.timestep, value)
        weight = self.model.obpdf(self.y, particle)
        print(str(particle)+' '+str(weight))

    def setup(self):
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
        pass

    def run(self):
        self.setup()
        for line in sys.stdin:
            val = line.rstrip().split(' ')
            particle = float(val[0])
            self.map(particle)
        self.cleanup()

if __name__ == '__main__':
    maptask = HieBfMapper1(models.BenchmarkModel, 'benchmarkmapconf.txt')
    maptask.run()
