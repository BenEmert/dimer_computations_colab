#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=60:00:00   # walltime
#SBATCH -J "benbio"   # job name
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=1

# module load python3/3.7.0
# module load gcc/9.2.0

round_dir="/groups/astuart/mlevine/dimer_computations_colab/results/3bumps_eps1e-16"

echo "Sending results to $round_dir"
srun --ntasks=1 python3 bump_inference_tests_ainner_3bumpLibrary.py --m 3 --maxiter_O 10 --popsize_O 3 --maxiter_K 10 --popsize_K 25 --base_dir "$round_dir/a-inner_w-inner_m3/inner10x3_outer10x25" &
srun --ntasks=1 python3 bump_inference_tests_ainner_3bumpLibrary.py --m 4 --maxiter_O 10 --popsize_O 3 --maxiter_K 10 --popsize_K 25 --base_dir "$round_dir/a-inner_w-inner_m4/inner10x3_outer10x25" &
srun --ntasks=1 python3 bump_inference_tests_ainner_3bumpLibrary.py --m 5 --maxiter_O 10 --popsize_O 3 --maxiter_K 10 --popsize_K 25 --base_dir "$round_dir/a-inner_w-inner_m5/inner10x3_outer10x25" &
srun --ntasks=1 python3 bump_inference_tests_ainner_3bumpLibrary.py --m 6 --maxiter_O 10 --popsize_O 3 --maxiter_K 10 --popsize_K 25 --base_dir "$round_dir/a-inner_w-inner_m6/inner10x3_outer10x25"
wait
