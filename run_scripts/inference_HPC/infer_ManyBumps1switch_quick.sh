#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=72:00:00   # walltime
#SBATCH -J "benbio"   # job name
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1

# module load python3/3.7.0
# module load gcc/9.2.0

round_dir="/groups/astuart/mlevine/dimer_computations_colab/results/ManyBumps1switch_singlebeta_constrained"

echo "Sending results to $round_dir"
srun --ntasks=1 python3 bump_inference_tests_ainner_ManyBumps.py --n_switches 1 --single_beta 1 --w_opt outer --m 3 --maxiter_O 10 --popsize_O 6 --maxiter_K 10 --popsize_K 15 --base_dir "$round_dir/m3_inner10x3_outer10x15" &
srun --ntasks=1 python3 bump_inference_tests_ainner_ManyBumps.py --n_switches 1 --single_beta 1 --w_opt outer --m 4 --maxiter_O 10 --popsize_O 6 --maxiter_K 10 --popsize_K 15 --base_dir "$round_dir/m4_inner10x3_outer10x15" &
srun --ntasks=1 python3 bump_inference_tests_ainner_ManyBumps.py --n_switches 1 --single_beta 1 --w_opt outer --m 5 --maxiter_O 10 --popsize_O 6 --maxiter_K 10 --popsize_K 15 --base_dir "$round_dir/m5_inner10x3_outer10x15" &
srun --ntasks=1 python3 bump_inference_tests_ainner_ManyBumps.py --n_switches 1 --single_beta 1 --w_opt outer --m 6 --maxiter_O 10 --popsize_O 6 --maxiter_K 10 --popsize_K 15 --base_dir "$round_dir/m6_inner10x3_outer10x15" &
srun --ntasks=1 python3 bump_inference_tests_ainner_ManyBumps.py --n_switches 1 --single_beta 1 --w_opt outer --m 3 --maxiter_O 10 --popsize_O 3 --maxiter_K 10 --popsize_K 15 --base_dir "$round_dir/m3_inner10x3_outer10x15" &
srun --ntasks=1 python3 bump_inference_tests_ainner_ManyBumps.py --n_switches 1 --single_beta 1 --w_opt outer --m 4 --maxiter_O 10 --popsize_O 3 --maxiter_K 10 --popsize_K 15 --base_dir "$round_dir/m4_inner10x3_outer10x15" &
srun --ntasks=1 python3 bump_inference_tests_ainner_ManyBumps.py --n_switches 1 --single_beta 1 --w_opt outer --m 5 --maxiter_O 10 --popsize_O 3 --maxiter_K 10 --popsize_K 15 --base_dir "$round_dir/m5_inner10x3_outer10x15" &
srun --ntasks=1 python3 bump_inference_tests_ainner_ManyBumps.py --n_switches 1 --single_beta 1 --w_opt outer --m 6 --maxiter_O 10 --popsize_O 3 --maxiter_K 10 --popsize_K 15 --base_dir "$round_dir/m6_inner10x3_outer10x15"
wait
