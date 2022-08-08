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

round_dir="/groups/astuart/mlevine/dimer_computations_colab/results/bumps"

echo "Sending results to $round_dir"
srun --ntasks=1 python3 bump_inference_tests_ainner.py --bump_center 0.8 --bump_width 2 --maxiter_O 10 --popsize_O 3 --maxiter_K 10 --popsize_K 25 --base_dir "$round_dir/center0.8_width2/inner10x3_outer10x25" &
srun --ntasks=1 python3 bump_inference_tests_ainner.py --bump_center 0.5 --bump_width 2 --maxiter_O 10 --popsize_O 3 --maxiter_K 10 --popsize_K 25 --base_dir "$round_dir/center0.5_width2/inner10x3_outer10x25" &
srun --ntasks=1 python3 bump_inference_tests_ainner.py --bump_center 0.2 --bump_width 2 --maxiter_O 10 --popsize_O 3 --maxiter_K 10 --popsize_K 25 --base_dir "$round_dir/center0.2_width2/inner10x3_outer10x25" &
srun --ntasks=1 python3 bump_inference_tests_ainner.py --bump_center 0.8 --bump_width 1 --maxiter_O 10 --popsize_O 3 --maxiter_K 10 --popsize_K 25 --base_dir "$round_dir/center0.8_width1/inner10x3_outer10x25" &
srun --ntasks=1 python3 bump_inference_tests_ainner.py --bump_center 0.5 --bump_width 1 --maxiter_O 10 --popsize_O 3 --maxiter_K 10 --popsize_K 25 --base_dir "$round_dir/center0.5_width1/inner10x3_outer10x25" &
srun --ntasks=1 python3 bump_inference_tests_ainner.py --bump_center 0.2 --bump_width 1 --maxiter_O 10 --popsize_O 3 --maxiter_K 10 --popsize_K 25 --base_dir "$round_dir/center0.2_width1/inner10x3_outer10x25"
wait
