#!/bin/bash

echo "Go to scratch"
#cd $SCRATCH

echo "Prepare modules"
module load daint-gpu
module load craype-accel-nvidia60
module load pycuda/2016.1.2-CrayGNU-2016.11-Python-3.5.2-cuda-8.0

# Enable CUDNN
export CUDNN_BASE=/users/phwindis/cuda
export LD_LIBRARY_PATH=$CUDNN_BASE/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDNN_BASE/include:$CPATH
export LIBRARY_PATH=$CUDNN_BASE/lib64:$LD_LIBRARY_PATH

# avoid lock-issues 
export THEANO_FLAGS="mode=FAST_RUN,device=gpu,lib.cnmem=1,floatX=float32,base_compiledir=$SCRATCH/theano.NOBACKUP"

echo "Go back home"
JOB_ID=lstm64_3layers_singlestep
JOB_DESC="lstm with 3 layers, 64 nodes each, only one epoch per batch"
OUTDIR=$HOME/BTagger/RNN_out/${JOB_ID}
mkdir -p $OUTDIR
echo $OUTDIR
echo $JOB_DESC
echo $JOB_DESC > $OUTDIR/desc.txt

echo "Starting TrainClassifiers.py"

#python -m cProfile /users/phwindis/BTagger/RNNClassifier-HDF.py `echo "$OUTDIR"` &> log.txt
python /users/phwindis/BTagger/RNNClassifier-HDF.py `echo "$OUTDIR"` > $OUTDIR/log.txt
