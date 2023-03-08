#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=65:00:00   # walltime
#SBATCH -J "metaAutogen"   # job name
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err

srun python randomK_scriptGen.py
