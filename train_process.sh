#!/bin/bash
#SBATCH --cpus-per-task=4               # Ask for 6 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=12G                        # Ask for 32 GB of RAM
#SBATCH -o /network/tmp1/schmidtv/munit/disen_v1_shift_single/slurm-%j.out  # Write the log in $SCRATCH

. /network/home/schmidtv/anaconda3/etc/profile.d/conda.sh
conda activate munit
export COMET_API_KEY="5jmii3mKCc5WbSWZ6w9L6gzy0"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ai/modules/cuda-10.0/lib64/:/usr/local/cuda-9.0/lib64/

exec $@