#!/usr/bin/env bash

source ~/hadooprc
DATA_DIR=hdfs://localhost:9000/user/erli/hiebf/500k/step
iternumber=2
n=0
maptasks=2
reducetasks=1
while (( n < iternumber )); do
    INPUT_DIR=$DATA_DIR$n/data
    n=$((n+1))
    python updatestep.py $n
    OUTPUT_DIR_MAP1=$DATA_DIR$n/map1
    hadoop jar $HADOOP_PATH/contrib/streaming/hadoop-streaming-1.2.1.jar \
    -D mapred.map.tasks=${maptasks} -D mapred.reduce.tasks=0 \
    -input $INPUT_DIR -output $OUTPUT_DIR_MAP1 \
    -mapper mapper1.py \
    -file mapper1.py \
    -file benchmarkmapconf.txt \
    -file benchmarkob.txt -file models.py -file resample.py

     INPUT_DIR=$OUTPUT_DIR_MAP1
     OUTPUT_DIR_RED2=$DATA_DIR$n/red2
     hadoop jar $HADOOP_PATH/contrib/streaming/hadoop-streaming-1.2.1.jar \
    -D mapred.map.tasks=${maptasks} -D mapred.reduce.tasks=${reducetasks} \
    -input $INPUT_DIR -output $OUTPUT_DIR_RED2 \
    -mapper mapper2.py -reducer reducer2.py \
    -file mapper2.py -file reducer2.py \
    -file red2confile.txt \
    -file resample.py

     rm red2out.txt
     hadoop fs -get ${DATA_DIR}${n}/red2/part-00000 red2out.txt
     INPUT_DIR=$OUTPUT_DIR_MAP1
     CACHE_DIR=$OUTPUT_DIR_RED2
     OUTPUT_DIR_MAP3=$DATA_DIR$n/data
     hadoop jar $HADOOP_PATH/contrib/streaming/hadoop-streaming-1.2.1.jar \
    -D mapred.map.tasks=${maptasks} -D mapred.reduce.tasks=0 \
    -input $INPUT_DIR -output $OUTPUT_DIR_MAP3 \
    -mapper mapper3.py \
    -file mapper3.py \
    -file resample.py -file red2out.txt
    #-file "hdfs://localhost:9000/user/erli/hiebfbenchmark/step${n}/red2/part-00000 "#'red2out.txt'
done

