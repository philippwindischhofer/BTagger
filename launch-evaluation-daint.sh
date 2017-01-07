srun --time=100 --nodes=1 --gres=gpu:1 -C gpu --partition=normal ./evaluateRNNClassifier-daint.sh ./RNN_out/lstm128_1layer_singlestep_250_2
