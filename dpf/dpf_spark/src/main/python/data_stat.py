import sys


def particle_mean(file_name):
    sum = 0.0
    weight_sum = 0.0
    with open(file_name) as file:
        for line in file.readlines():
            w, x = line.rstrip().split(" ")
            sum += float(w) * float(x)
            weight_sum += float(w)
    return sum / weight_sum


def mean(file_name):
    sum = 0.0
    num = 0
    with open(file_name) as file:
        for line in file.readlines():
            sum += float(line.rstrip())
            num += 1
    return sum / num


def var(file_name, mean):
    sum = 0.0
    num = 0
    with open(file_name) as file:
        for line in file.readlines():
            sum += (float(line.rstrip()) - mean) * (
                float(line.rstrip()) - mean)
            num += 1
    return sum / num

if __name__ == '__main__':
    sample_file_name = sys.argv[1]
    sample_mean = particle_mean(sample_file_name)
    print(sample_mean)
    mean_file_name = sys.argv[2]
    temp_mean = mean(mean_file_name)
    print("mean: " + str(temp_mean))
    print("variance: " + str(var(mean_file_name, temp_mean)))
