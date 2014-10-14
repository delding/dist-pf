import numpy as np


if __name__ == '__main__':
    w = []
    file = open('/data/parBeforeResp')
    for line in file.readlines():
        weight, particle = line.rstrip().split(' ')
        w.append(float(weight))
    index = []
    index_rep = []
    for j in range(10000):
        index.append(np.random.random_integers(0, 10000 - 1))
    for k in range(10000):
        index_rep.append(index.count(k))
    weights = np.array(w)
    reps = np.array(index_rep)
    x1 = 1 / np.inner(weights, reps)
