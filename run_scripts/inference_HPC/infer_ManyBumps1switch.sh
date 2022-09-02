#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=72:00:00   # walltime
#SBATCH -J "benbio"   # job name
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=1

# module load python3/3.7.0
# module load gcc/9.2.0

round_dir="/groups/astuart/mlevine/dimer_computations_colab/results/ManyBumps1switch_singlebeta_constrained"

echo "Sending results to $round_dir"
srun --ntasks=1 python3 bump_inference_tests_ainner_ManyBumps.py --n_switches 1 --single_beta 1 --w_opt outer --maxiter_O 10 --popsize_O 3 --maxiter_K 10 --popsize_K 25 --base_dir "$round_dir/m3_inner10x3_outer10x25" &
srun --ntasks=1 python3 bump_inference_tests_ainner_ManyBumps.py --n_switches 1 --single_beta 1 --w_opt outer --maxiter_O 10 --popsize_O 3 --maxiter_K 25 --popsize_K 25 --base_dir "$round_dir/m3_inner10x3_outer25x25" &
srun --ntasks=1 python3 bump_inference_tests_ainner_ManyBumps.py --n_switches 1 --single_beta 1 --w_opt outer --maxiter_O 10 --popsize_O 10 --maxiter_K 10 --popsize_K 25 --base_dir "$round_dir/m3_inner10x10_outer10x25" &
srun --ntasks=1 python3 bump_inference_tests_ainner_ManyBumps.py --n_switches 1 --single_beta 1 --w_opt outer --maxiter_O 10 --popsize_O 3 --maxiter_K 10 --popsize_K 25 --base_dir "$round_dir/m3_inner10x3_outer10x25" &
srun --ntasks=1 python3 bump_inference_tests_ainner_ManyBumps.py --n_switches 1 --single_beta 1 --w_opt outer --maxiter_O 10 --popsize_O 3 --maxiter_K 25 --popsize_K 25 --base_dir "$round_dir/m3_inner10x3_outer25x25" &
srun --ntasks=1 python3 bump_inference_tests_ainner_ManyBumps.py --n_switches 1 --single_beta 1 --w_opt outer --maxiter_O 10 --popsize_O 10 --maxiter_K 10 --popsize_K 25 --base_dir "$round_dir/m3_inner10x10_outer10x25"
wait
