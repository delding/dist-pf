import numpy as np
import resample
import models
import sys
from particle import Particle


class DPF(object):
    # [partition[Par,Par,...,Par], partition, ..., partition]
    def __init__(self, model, num_particles, num_partitions):
        self.model = model
        self.num_particles = num_particles
        self.num_partitions = num_partitions

    # partition is list(Particle)
    def propagate(self, t, y, partition):
        propagated = []
        for par in partition:
            x = self.model.propagate(t, par.state)
            w = self.model.obpdf(y, par.state) * par.weight
            propagated.append(Particle(w, x))
        return propagated

    # selection function, shuffled is a list of buckets,
    # bucket is a list of particles
    def shuffle(self, partition):
        num_per_split = len(partition) / self.num_partitions
        shuffled = []
        for i in range(self.num_partitions):
            split = []
            index = []
            index_rep = []
            for j in range(num_per_split):
                index.append(np.random.random_integers(0, len(partition) - 1))
            for k in range(len(partition)):
                index_rep.append(index.count(k))
            for rep, par in zip(index_rep, partition):
                while rep > 0:
                    split.append(par)
                    rep -= 1
            shuffled.append(split)
        return shuffled

    # shuffle function, repartition_arr is a list of partitions
    def repartition(self, shuffled_arr):
        repartition_arr = []
        for i in range(self.num_partitions):
            partition = []
            for shuffled in shuffled_arr:
                # shuffled[i] is ith split
                for par in shuffled[i]:
                    partition.append(par)
            repartition_arr.append(partition)
        return repartition_arr

    def resample(self, partition):
        resampled = []
        weights = []
        for par in partition:
            weights.append(par.weight)
        weights = np.array(weights)
        weights /= weights.sum()
        index_rep = resample.systematic(len(partition), weights)
        weight = 1. / self.num_particles
        for par, rep in zip(partition, index_rep):
            while rep > 0:
                resampled.append(Particle(weight, par.state))
                rep -= 1
        return resampled

    def run(self, t, y, partition_arr):
        propagated_arr = []
        for partition in partition_arr:
            propagated_arr.append(self.propagate(t, y, partition))
        shuffled_arr = []
        for partition in propagated_arr:
            shuffled_arr.append(self.shuffle(partition))
        repartitioned_arr = self.repartition(shuffled_arr)
        resampled_arr = []
        for partition in repartitioned_arr:
            resampled_arr.append(self.resample(partition))
        return resampled_arr

    def eval_mean(self, partition_arr):
        weight_sum = 0.0
        mean = 0.0
        for partition in partition_arr:
            for par in partition:
                weight_sum += par.weight
                mean += par.weight * par.state
        return mean / weight_sum


class PartialDPF(object):
    def __init__(self, model, num_particles, num_partitions):
        self.model = model
        self.num_particles = num_particles
        self.num_partitions = num_partitions

    def resample(self, partition, resample_num, weight):
        resampled = []
        weights = []
        for par in partition:
            weights.append(par.weight)
        weights = np.array(weights)
        weights /= weights.sum()
        index_rep = resample.systematic(resample_num, weights)
        for par, rep in zip(partition, index_rep):
            while rep > 0:
                resampled.append(Particle(weight, par.state))
                rep -= 1
        return resampled

    def cen_resample(self, partition_arr):
        particles = []
        for partition in partition_arr:
            for par in partition:
                particles.append(par)
        weights = []
        for par in particles:
            weights.append(par.weight)
        weights = np.array(weights)
        weights /= weights.sum()
        index_rep = resample.systematic(len(particles), weights)
        weight = 1. / self.num_particles
        resampled = []
        for par, rep in zip(particles, index_rep):
            while rep > 0:
                resampled.append(Particle(weight, par.state))
                rep -= 1
        resampled_arr = []
        # append() returns NoneType, can not be returned directly
        resampled_arr.append(resampled)
        return resampled_arr

    def p_dpf1_resample(self, partition_arr):
        partition_weight_sums = []
        for partition in partition_arr:
            weight_sum = 0.0
            for par in partition:
                weight_sum += par.weight
            partition_weight_sums.append(weight_sum)
        partition_weight_sums = np.array(partition_weight_sums)
        partition_weight_sums /= partition_weight_sums.sum()
        resampled_arr = []
        for partition, weight in zip(partition_arr, partition_weight_sums):
            resampled_arr.append(self.resample(partition,
                                               len(partition), weight))
        return resampled_arr

    def p_dpf2_resample(self, partition_arr):
        partition_weight_sums = []
        for partition in partition_arr:
            weight_sum = 0.0
            for par in partition:
                weight_sum += par.weight
            partition_weight_sums.append(weight_sum)
        partition_weight_sums = np.array(partition_weight_sums)
        partition_weight_sums /= partition_weight_sums.sum()
        partition_resp_nums = resample.systematic(self.num_particles,
                                                  partition_weight_sums)
        resampled_arr = []
        weight = 1.0 / self.num_particles
        for resample_num, partition in zip(partition_resp_nums,
                                           partition_arr):
            resampled_arr.append(self.resample(partition,
                                               resample_num, weight))
        return resampled_arr

    def p_dpf3_resample(self, partition_arr):
        partition_weight_sums = []
        for partition in partition_arr:
            weight_sum = 0.0
            for par in partition:
                weight_sum += par.weight
            partition_weight_sums.append(weight_sum)
        partition_weight_sums = np.array(partition_weight_sums)
        partition_weight_sums /= partition_weight_sums.sum()
        partition_rep_nums = resample.systematic(self.num_partitions,
                                                 partition_weight_sums)
        resampled_arr = []
        weight = 1.0 / self.num_particles
        for partition, rep in zip(partition_arr, partition_rep_nums):
            while rep > 0:
                resampled_arr.append(self.resample(partition,
                                                   len(partition), weight))
                rep -= 1
        return resampled_arr


def generate_sample(sample_file_name, num_partitions, num_particles,
                    t, y):
    sample_file = open(sample_file_name, 'w')
    partition_arr = []
    for i in range(num_partitions):
        partition = []
        for j in range(num_particles / num_partitions):
            x = models.BenchmarkModel.propagate(
                t, models.BenchmarkModel.initialize())
            w = models.BenchmarkModel.obpdf(y, x)
            partition.append(Particle(w, x))
            sample_file.write(str(w) + " " + str(x) + "\n")
        partition_arr.append(partition)
    sample_file.close()


def read_sample(sample_file_name, num_partitions, num_particles):
    partition_arr = []
    sample_file = open(sample_file_name)
    # read particle sample into partition_arr
    for num in range(num_partitions):
        part = []
        for num_par in range(num_particles / num_partitions):
            w, x = sample_file.readline().rstrip().split(' ')
            part.append(Particle(float(w), float(x)))
        partition_arr.append(part)
    sample_file.close()
    return partition_arr


# test the influence of selection step on estimation result
# use shuffled particle set, because the estimation result of
# shuffled set is as same as selected set
def test_selection(shuffle_mean_file, sim_number, partition_arr, dpf):
    shuffled_mean_file = open(shuffle_mean_file, 'w')
    for sim_num in range(sim_number):
        selected_arr = []
        for partition in partition_arr:
            selected_arr.append(dpf.shuffle(partition))
        shuffled_arr = dpf.repartition(selected_arr)
        shuffled_mean = dpf.eval_mean(shuffled_arr)
        shuffled_mean_file.write(str(shuffled_mean) + "\n")
    shuffled_mean_file.close()


def test_resample(resample_mean_file, local_weight_sum_file,
                  sim_number, partition_arr, dpf):
    with open(resample_mean_file, 'w') as resampled_mean_file, open(
            local_weight_sum_file, 'w') as local_weight_sum:
        for sim_num in range(sim_number):
            selected_arr = []
            for part in partition_arr:
                selected_arr.append(dpf.shuffle(part))
            shuffled_arr = dpf.repartition(selected_arr)
            resampled_arr = []
            for part in shuffled_arr:
                resampled_arr.append(dpf.resample(part))
            # write means
            resampled_mean = dpf.eval_mean(resampled_arr)
            resampled_mean_file.write(str(resampled_mean) + '\n')
            # write local weight sums
            for part in shuffled_arr:
                weight_sum = .0
                for par in part:
                    weight_sum += par.weight
                local_weight_sum.write(str(weight_sum) + ' ')
            local_weight_sum.write('\n')


if __name__ == '__main__':
    print(sys.argv[0])
    #argv[1]: num_particles, argv[2]: num_partitions
    num_particles = int(sys.argv[1])
    num_partitions = int(sys.argv[2])
    dpf = DPF(models.BenchmarkModel, num_particles, num_partitions)
    pdpf = PartialDPF(models.BenchmarkModel, num_particles, num_partitions)
    t = 1.0
    y = 5.60923390197
    # read selected particles into selected_arr
    # selected_arr = []
    # for num in range(num_partitions):
    #     part = []
    #     for num_b in range(num_partitions):
    #         bucket = []
    #         for par in range(num_particles / num_partitions
    #                          / num_partitions):
    #             w, x = sample_file.readline().rstrip().split(' ')
    #             bucket.append(Particle(float(w), float(x)))
    #         part.append(bucket)
    #     selected_arr.append(part)

    # selection_mean_file = open("selection_mean", 'w')
    # shuffle_mean_file = open("shuffle_mean", 'w')
    # shuffle_weight_sum_file = open('shuffle_weight_sum', 'w')
    # resampled_mean_file = open('resample_mean', 'w')
    # cen_mean_file = open("cen_mean", 'w')
    # pdpf1_mean_file = open("pdpf1_mean", 'w')
    # pdpf2_mean_file = open("pdpf2_mean", 'w')
    # pdpf3_mean_file = open("pdpf3_mean", 'w')
        #dpf_mean = dpf.eval_mean(resampled_arr)
        # #cenbf
        # cen_mean = dpf.eval_mean(pdpf.cen_resample(partition_arr))
        # cen_mean_file.write(str(cen_mean) + "\n")
        # #pdpf1
        # pdpf1_mean = dpf.eval_mean(pdpf.p_dpf1_resample(partition_arr))
        # pdpf1_mean_file.write(str(pdpf1_mean) + "\n")
        # #pdpf2
        # pdpf2_mean = dpf.eval_mean(pdpf.p_dpf2_resample(partition_arr))
        # pdpf2_mean_file.write(str(pdpf2_mean) + "\n")
        # #pdpf3
        # pdpf3_mean = dpf.eval_mean(pdpf.p_dpf3_resample(partition_arr))
        # pdpf3_mean_file.write(str(pdpf3_mean) + "\n")
    #selection_mean_file.close()
    #shuffle_weight_sum_file.close()
    #resampled_mean_file.close()
    # cen_mean_file.close()
    # pdpf1_mean_file.close()
    # pdpf2_mean_file.close()
    # pdpf3_mean_file.close()
    partition_arr = read_sample('particle_sample_6400', 8, 6400)
    #test_selection('selection_mean_6400par_5000sim', 5000, partition_arr, dpf)
    #generate_sample('particle_sample_6400', 8, 6400, t, y)
    test_resample('resample_mean_6400par_5000sim',
                  'shuffle_weight_sum_6400par', 5000, partition_arr, dpf)
