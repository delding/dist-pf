#!/usr/bin/env python

import sys
import numpy as np
import models
import resample

class Reducer(object):
    pass

class CenBfReducer(Reducer):
    def __init__(self, redconfigurefile):
        self.redconfigurefile = redconfigurefile

    def setup(self):
        with open(self.redconfigurefile, 'r') as configure:
            self.particlenumber = int(configure.readline().rstrip())
            self.resamplenumber = int(configure.readline().rstrip())
        self.particles = np.arange(self.particlenumber, dtype = 'float64')
        self.weights = np.arange(self.particlenumber, dtype = 'float64')

    def reduce(self, particles, weights):
        weights = weights / weights.sum()
        resamplecounts = resample.systematic(self.resamplenumber, weights)
        weight = 1. / self.resamplenumber
        for particle, count in zip(particles, resamplecounts):
            n = count
            while n > 0:
                print(str(particle)+' '+str(weight))
                n -= 1

    def cleanup(self):
        pass

    def run(self):
        self.setup()
        for index, line in enumerate(sys.stdin):
            key, val = line.rstrip().split('\t')
            value = val.split(' ')
            self.particles[index] = float(value[0])
            self.weights[index] = float(value[1])
        self.reduce(self.particles, self.weights)
        self.cleanup()

if __name__ == '__main__':
    reducetask = CenBfReducer('benchmarkredconf.txt')
    reducetask.run()
