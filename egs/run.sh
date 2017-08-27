#!/bin/bash
# qsub options:
#$ -l gpu=1
#$ -q gpgpu
#$ -cwd
# First option informs scheduler that the job requires a gpu.
# Second ensures the job is put in the batch job queue, not the interactive queue

# Set up the CUDA environment
export CUDA_HOME=/opt/cuda-8.0.44
export CUDNN_HOME=/opt/cuDNN-5.1
export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH
export CPATH=${CUDNN_HOME}/include:$CPATH
export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

# This finds the UUID of the GPU with the lowest memory used, and makes sure that is the one you use:
# export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=gpu_uuid,memory.used --format=csv,noheader,nounits | sort -n --key=2 | cut -d "," -f 1 | head -n 1)
source ~/get_free_gpu.sh
echo "Chosen gpu:" $CUDA_VISIBLE_DEVICES

# Activate the relevant virtual environment
source ~/miniconda2/bin/activate py2
export MPLBACKEND="agg"

# run the python program
if [ -z "$1" ]; then
	# if using an array job, no task_id supplied
    python $HOME/modNN/src/run.py $1
else
	# $3 in [0, num_tasks] (task_id)
	python $HOME/modNN/src/run.py $1 $2
fi


