#!/bin/bash

echo "Go to scratch"
#cd $SCRATCH

echo "Prepare modules"
module load daint-gpu
module load craype-accel-nvidia60
module load pycuda
module load pycuda/2016.1.2-CrayGNU-2016.11-Python-3.5.2-cuda-8.0

# Enable CUDNN
export CUDNN_BASE=/users/phwindis/cuda
export LD_LIBRARY_PATH=$CUDNN_BASE/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDNN_BASE/include:$CPATH
export LIBRARY_PATH=$CUDNN_BASE/lib64:$LD_LIBRARY_PATH

# avoid lock-issues 
export THEANO_FLAGS="mode=FAST_RUN,device=gpu,lib.cnmem=1,floatX=float32,base_compiledir=$SCRATCH/theano.NOBACKUP"

. ./run.sh
