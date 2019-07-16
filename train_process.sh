#!/bin/bash
#SBATCH --account=rpp-bengioy            # Yoshua pays for your job
#SBATCH --cpus-per-task=4               # Ask for 6 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=12G                        # Ask for 32 GB of RAM
#SBATCH -o /scratch/schmidtv/slurm-%j.out  # Write the log in $SCRATCH
module load python/3.6
module load 
source $HOME/venv/env/bin/activate

exec $@