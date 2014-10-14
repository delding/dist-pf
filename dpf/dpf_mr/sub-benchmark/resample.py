import numpy as np

def multinomial(resamplenumber, weightarray):
    return np.random.multinomial(resamplenumber, weightarray)

def systematic(resamplenumber, weightarray):
    resamplecounts = np.arange(len(weightarray))
    rand = np.random.random_sample() / resamplenumber
    weight_cumulative = 0.
    for k, w in enumerate(weightarray):
        count = 0
        weight_cumulative += w
        while weight_cumulative > rand:
            count += 1
            rand += 1. / resamplenumber
        resamplecounts[k] = count
    return resamplecounts
