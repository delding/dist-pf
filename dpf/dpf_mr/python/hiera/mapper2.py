#!/usr/bin/env python

import sys

class HieBfMapper2(object):
    def __init__(self):
        pass

    def map(self, value):
        self.W_l += value

    def setup(self):
       self.W_l = 0.

    def cleanup(self):
        print('%s\t%s' % ('reducer2', str(self.W_l)))

    def run(self):
        self.setup()
        for line in sys.stdin:
            val = line.rstrip().split(' ')
            weight = float(val[1])
            self.map(weight)
        self.cleanup()

if __name__ == '__main__':
    maptask = HieBfMapper2()
    maptask.run()
