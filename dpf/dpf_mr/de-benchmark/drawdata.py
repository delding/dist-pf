import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    particlenumber = int(sys.argv[2])
    particles = np.arange(particlenumber, dtype='float64')
    normedweights = np.arange(particlenumber, dtype='float64')
    with open(sys.argv[1]) as file:
        for index, line in enumerate(file.readlines()):
            particle, normedweight = line.rstrip().split(' ')
            particles[index], normedweights[index] = float(particle), float(normedweight)
    plt.hist(particles, bins=50, weights=normedweights, normed=True, label='Histogram')
    plt.show()
