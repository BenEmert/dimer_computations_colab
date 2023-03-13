import os, sys
import time
import numpy as np
import subprocess

job_directory = 'inference_HPC/randomK_jacobTargets_perDimer_autogen'
os.makedirs(job_directory, exist_ok=True)

sbatch_str = """#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=24:00:00   # walltime
#SBATCH --array=0-1000      # how many tasks in the array
#SBATCH -J "randomK_jacobTargets_perID_m{m}_{offset}"   # job name
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --mem-per-cpu={mem}G

base_dir="/groups/astuart/mlevine/dimer_computations_colab/results/BoundedBumps_randomK_jacobTarget_perDimer_8.1.0"

id=$(( {offset} + $SLURM_ARRAY_TASK_ID))
srun python bump_inference_tests_ainner_BoundedBumpsSeparate_main_randomK_jacobTargets_perDimer.py --id $id --base_dir $base_dir --grid_dir badname --m {m}
"""

# sleep_secs = 60*60 # length of time (secs) to wait before trying to submit more jobs. Using 1 hour.
num_Ks = 10
n_per_batch_submission = 1000
n_target_dict = {m: np.load('../data/voxel_averages/{}M_voxel_averages.npy'.format(m)).shape[0] for m in range(3,13)}
n_dimers_dict = {m: m*(m+1)/2}
mem_dict = {3: 5, 4: 10, 5: 20, 10: 20}
for m in [3, 4, 5, 10]:
    max_jobs = num_Ks * n_target_dict[m] * n_dimers_dict[m]
    for offset in np.arange(0, max_jobs, 1000):
        job_file = os.path.join(job_directory, "m{}_offset{}.job".format(m, offset))

        with open(job_file, 'w') as fh:
            fh.writelines(sbatch_str.format(m=m, offset=offset, mem=mem_dict[m]))
