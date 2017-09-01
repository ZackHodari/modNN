#!/bin/bash

######
# USAGE
# bash submit.sh <job_name> <task_set> <begin_id> <end_id>

qsub \
	-N $1 \
	-t $3-$4:1 \
	-tc 2 \
	-o results/stdout \
	-e results/stdout \
	run.sh $2

