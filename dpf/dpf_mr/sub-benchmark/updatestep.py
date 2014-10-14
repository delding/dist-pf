import sys
#change 4th parameter calling writelines method when number of reducers is changed
if __name__ == '__main__':
    with open('benchmarkmapconf.txt', 'w') as file:
        file.writelines(['benchmarkob.txt\n','1.0\n',str(sys.argv[1])+'\n'+'1\n'])
