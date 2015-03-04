#!/usr/bin/env python

import sys
import numpy as np

class DeBfReducer(object):
    def __init__(self):
        pass

    def setup(self):
        pass
    def reduce(self, values):
        particles, weights = np.array(zip(*values))
        W_g = weights.sum()
        weights /= W_g
        for particle, weight in zip(particles, weights):
            print(str(particle)+' '+str(weight))

    def cleanup(self):
        pass

    def run(self):
        self.setup()
        values=[]
        for line in sys.stdin:
            key, val = line.rstrip().split('\t')
            value = val.split(' ')
            values.append([float(value[0]), float(value[1])])
        self.reduce(values)
        self.cleanup()

if __name__ == '__main__':
    reducetask = DeBfReducer()
    reducetask.run()
