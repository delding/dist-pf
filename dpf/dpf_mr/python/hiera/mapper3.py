#!/usr/bin/env python

import sys
import numpy as np
import resample

class HieBfMapper3(object):
    def __init__(self, distributedCache):
        self.cachefile = distributedCache

    def setup(self):
        self.disCache = {}
        self.mapCache = []
        with open(self.cachefile, 'r') as cache:
            for line in cache:
                w, c = line.rstrip().split('\t')
                try:
                    if not self.disCache.has_key(w):
                        self.disCache[w] = int(c)
                    else:
                        raise Exception()
                except Exception:
                    print('Found two equal local weight sum')
                    sys.exit()
        self.particlenumber = 0
        for number in self.disCache.values():
            self.particlenumber += number

    def map(self, value):
        self.mapCache.append([value[0], value[1]])

    def cleanup(self):
        particles, weights = np.array(zip(*self.mapCache))
        W_l = weights.sum()
        N_resample = self.disCache[str(W_l)]
        weights /= W_l
        resamplecounts = resample.systematic(N_resample, weights)
        weight = 1.0 / self.particlenumber
        for particle, count in zip(particles, resamplecounts):
            n = count
            while n > 0:
                print(str(particle)+' '+str(weight))
                n -= 1

    def run(self):
        self.setup()
        for line in sys.stdin:
            val = line.rstrip().split(' ')
            value = float(val[0]), float(val[1])
            self.map(value)
        self.cleanup()

if __name__ == '__main__':
    maptask3=HieBfMapper3('red2out.txt')
    maptask3.run()
