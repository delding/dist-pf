import numpy as np
from scipy.stats import norm

class Model(object):
    pass

class BenchmarkModel(Model):
    def __init__(self):
        pass

    @classmethod
    def initialize(cls):
        return np.sqrt(10) * np.random.randn()

    @classmethod
    def propagate(cls, n, x):
        return x / 2 + 25 * x / (1 + x * x) + 8 * np.cos(1.2 * n) + np.sqrt(10) * np.random.randn()

    @classmethod
    def observe(cls, x):
        return x * x / 20 + np.random.randn()

    @classmethod
    def tranpdf(cls, n, x, pre_x):
        return norm.pdf(x, pre_x / 2 + 25 * pre_x / (1 + pre_x * pre_x) + 8 * np.cos(1.2 * n), np.sqrt(10))

    @classmethod
    def obpdf(cls, y, x):
        return norm.pdf(y, x * x / 20, 1)

    @classmethod
    def generatedata(cls, length, timestep, obfilename, statefilename):
        x = cls.initialize()
        obfile = open(obfilename, 'w')
        statefile = open(statefilename, 'w')
        try:
            step = 0
            while step < length:
                x = cls.propagate((step + 1) * timestep, x)
                statefile.write(str(x)+'\n')
                y = cls.observe(x)
                obfile.write(str(y)+'\n')
                step += 1
        finally:
            obfile.close()
            statefile.close()
    @classmethod
    def initializeparticles(cls, number, initfilename):
        with open(initfilename, 'w') as initfile:
            n = 0
            weight = 1.0 / number
            while n < number:
                particle = cls.initialize()
                initfile.write(str(particle)+' '+str(weight)+'\n')
                n += 1

if __name__ == '__main__':
    #BenchmarkModel.generatedata(100, 1.0, 'benchmarkob.txt', 'benchmarkstate.txt')
    BenchmarkModel.initializeparticles(500000, 'benchmarkinit500k.txt')
