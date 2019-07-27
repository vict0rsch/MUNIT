#!/bin/bash
#SBATCH --account=rpp-bengioy            # Yoshua pays for your job
#SBATCH --cpus-per-task=6               # Ask for 6 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=24G                        # Ask for 32 GB of RAM
#SBATCH -o /home/vsch/scratch/munit/outputs/disen_v1_shift_single/slurm-%j.out  # Write the log in $SCRATCH

#. /network/home/schmidtv/anaconda3/etc/profile.d/conda.sh
#conda activate munit
export COMET_API_KEY="5jmii3mKCc5WbSWZ6w9L6gzy0"
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ai/modules/cuda-10.0/lib64/:/usr/local/cuda-9.0/lib64/

source ~/MUNIT/munitenv/bin/activate

exec $@