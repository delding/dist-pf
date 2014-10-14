import sys
import numpy as np
import matplotlib.pyplot as plt


def draw_hist(file_name, num_particles, num_bins=50, with_weights=True):
    if with_weights is True:
        particles = np.arange(num_particles, dtype='float64')
        normedweights = np.arange(num_particles, dtype='float64')
        with open(file_name) as file:
            for index, line in enumerate(file.readlines()):
                normedweight, particle = line.rstrip().split(' ')
                particles[index], normedweights[index] = \
                    float(particle), float(normedweight)
        plt.hist(particles, bins=num_bins, weights=normedweights,
                 normed=True, label='Histogram')
        plt.show()
    else:
        particles = np.arange(num_particles, dtype='float64')
        with open(file_name) as file:
            for index, line in enumerate(file.readlines()):
                particle = line.rstrip()
                particles[index] = float(particle)
        plt.hist(particles, bins=num_bins, normed=True, label='Histogram')
        plt.show()

# argv[1] filename, argv[2] particleNumber
if __name__ == '__main__':
    file_name = sys.argv[1]
    num_particles = int(sys.argv[2])
    num_bins = int(sys.argv[3])
    with_weights = bool(int(sys.argv[4]))
    draw_hist(file_name, num_particles, num_bins, with_weights)
