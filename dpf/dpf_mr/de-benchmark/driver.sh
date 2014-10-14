#!/usr/bin/env bash

source ~/hadooprc
DATA_DIR=debf/500k/step
iternumber=2
n=0
maptasks=2
reducetasks=1
while (( n < iternumber )); do
    INPUT_DIR=$DATA_DIR$n
    n=$((n+1))
    python updatestep.py $n
    OUTPUT_DIR=$DATA_DIR$n
    hadoop jar $HADOOP_PATH/contrib/streaming/hadoop-streaming-1.2.1.jar \
    -D mapred.map.tasks=${maptasks} -D mapred.reduce.tasks=${reducetasks} \
    -input $INPUT_DIR -output $OUTPUT_DIR \
    -mapper mapper.py -reducer reducer.py \
    -file mapper.py -file reducer.py \
    -file benchmarkmapconf.txt \
    -file benchmarkob.txt -file models.py -file resample.py
done

