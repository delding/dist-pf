#!/usr/bin/env python

import sys
import numpy as np
import resample

class SubBfReducer(object):
    def __init__(self, redconfile):
        self.redconfile=redconfile

    def setup(self):
        with open(self.redconfile, 'r') as configure:
            self.N=int(configure.readline().rstrip())
            self.N_red = int(configure.readline().rstrip())

    def reduce(self, values):
        particles, weights = np.array(zip(*values))
        W_r = weights.sum()
        weights /= W_r
        N_r = self.N / self.N_red
        resamplecounts = resample.systematic(N_r, weights)
        weight = 1. / self.N
        for particle, count in zip(particles, resamplecounts):
            n = count
            while n > 0:
                print(str(particle)+' '+str(weight))
                n -= 1

    def cleanup(self):
        pass

    def run(self):
        self.setup()
        pre_key,val = sys.stdin.readline().rstrip().split('\t')
        value = val.split(' ')
        values = [[float(value[0]), float(value[1])]]
        for line in sys.stdin:
            key, val = line.rstrip().split('\t')
            value = val.split(' ')
            if key == pre_key:
                values.append([float(value[0]), float(value[1])])
            else:
                self.reduce(values)
                values=[[float(value[0]), float(value[1])]]
                pre_key=key
        # make a call for last key, for which test pre_key==key is always true
        self.reduce(values)
        self.cleanup()

if __name__ == '__main__':
    reducetask = SubBfReducer('redconf.txt')
    reducetask.run()
