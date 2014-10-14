#!/usr/bin/env python

import sys
import numpy as np
import resample

class HieBfReducer2(object):
    def __init__(self, red2confile):
        self.red2confile = red2confile

    def setup(self):
        with open(self.red2confile, 'r') as configure:
            self.particlenumber=int(configure.readline().rstrip())

    def reduce(self, values):
        mapweights = np.array(values)
        weights = mapweights / mapweights.sum()
        mapresamplecounts = resample.multinomial(self.particlenumber, weights)
        for w, c in zip(mapweights, mapresamplecounts):
            print('%s\t%s' % (str(w), str(c)))
        # for cat test only
        #with open('red2out.txt','w') as file:
        #    for w,c in zip(mapweights, mapresamplecounts):
        #        file.write(str(w)+'\t'+str(c)+'\n')

    def cleanup(self):
        pass

    def run(self):
        self.setup()
        values=[]
        for line in sys.stdin:
            key, val = line.rstrip().split('\t')
            values.append(float(val))
        self.reduce(values)
        self.cleanup()

if __name__ == '__main__':
    reducetask2 = HieBfReducer2('red2confile.txt')
    reducetask2.run()
