#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=60:00:00   # walltime
#SBATCH -J "benbio"   # job name
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1

# module load python3/3.7.0
# module load gcc/9.2.0

round_dir="/groups/astuart/mlevine/dimer_computations_colab/results/6bumps_mixed_eps1e-16_singlebeta_percentiles_v2"

echo "Sending results to $round_dir"
srun --ntasks=1 python3 bump_inference_tests_ainner_6bump_mixed_Library.py --single_beta 1 --maxiter_O 10 --popsize_O 3 --maxiter_K 10 --popsize_K 25 --base_dir "$round_dir/a-inner_w-inner_m3/inner10x3_outer10x25" &
srun --ntasks=1 python3 bump_inference_tests_ainner_6bump_mixed_Library.py --single_beta 1 --maxiter_O 10 --popsize_O 3 --maxiter_K 20 --popsize_K 50 --base_dir "$round_dir/a-inner_w-inner_m3/inner10x3_outer20x50" &
srun --ntasks=1 python3 bump_inference_tests_ainner_6bump_mixed_Library.py --single_beta 1 --maxiter_O 10 --popsize_O 10 --maxiter_K 20 --popsize_K 25 --base_dir "$round_dir/a-inner_w-inner_m3/inner10x10_outer20x25" &
srun --ntasks=1 python3 bump_inference_tests_ainner_6bump_mixed_Library.py --single_beta 1 --maxiter_O 10 --popsize_O 25 --maxiter_K 10 --popsize_K 25 --base_dir "$round_dir/a-inner_w-inner_m3/inner10x25_outer10x25" &
srun --ntasks=1 python3 bump_inference_tests_ainner_6bump_mixed_Library.py --single_beta 1 --maxiter_O 10 --popsize_O 25 --maxiter_K 20 --popsize_K 25 --base_dir "$round_dir/a-inner_w-inner_m3/inner10x25_outer20x25" &
srun --ntasks=1 python3 bump_inference_tests_ainner_6bump_mixed_Library.py --single_beta 1 --maxiter_O 10 --popsize_O 25 --maxiter_K 50 --popsize_K 25 --base_dir "$round_dir/a-inner_w-inner_m3/inner10x25_outer50x25" &
srun --ntasks=1 python3 bump_inference_tests_ainner_6bump_mixed_Library.py --single_beta 1 --maxiter_O 10 --popsize_O 25 --maxiter_K 20 --popsize_K 50 --base_dir "$round_dir/a-inner_w-inner_m3/inner10x25_outer20x50" &
srun --ntasks=1 python3 bump_inference_tests_ainner_6bump_mixed_Library.py --single_beta 1 --maxiter_O 15 --popsize_O 50 --maxiter_K 20 --popsize_K 50 --base_dir "$round_dir/a-inner_w-inner_m3/inner15x50_outer20x50"
wait
