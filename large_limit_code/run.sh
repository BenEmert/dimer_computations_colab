#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=1:00:00   # walltime
#SBATCH --n=5
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err

base_dir="/groups/astuart/mlevine/dimer_computations_colab/results/bigSims"

srun -n1 python script.py --n_monomers 9 --base_dir $base_dir &
srun -n1 python script.py --n_monomers 10 --base_dir $base_dir &
srun -n1 python script.py --n_monomers 11 --base_dir $base_dir &
srun -n1 python script.py --n_monomers 12 --base_dir $base_dir &
srun -n1 python script.py --n_monomers 13 --base_dir $base_dir
wait
