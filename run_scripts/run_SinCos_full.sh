#!/usr/bin/env bash
set -Eeuo pipefail

base_dir='results_SinCos_full'
TF='../data/hc_3M_metaClusterBasis_thresh3.npy'
m=3 # Total number of monomers
N=40 # Number of values to titrate the input monomer species. Values spaced evenly on a log10 scale
TL='SinCos' # Which Target Library to fit to
# TL='MetaClusters' # Which Target Library to fit to

# Optimization settings for differential evolution over binding affinities K
MI_K=3 # Maximum iterations
PI_K=15 # Population Size (number of particles = 2*dim(K)*PI_K)
W_K=-1 # number of workers (set to -1 to use maximum available workers)
PO_K=1 # Boolean: whether to Polish the optimization using scipy.optimize.minimize (trust-constr to obey bounds)

# Optimization settings for differential evolution over accessory monomers within a single choice of K
# MI_O=3 # Maximum iterations
# PI_O=15 # Population Size  (number of particles = 2*(m-1)*PI_O)
MI_O=3 # Maximum iterations
PI_O=15 # Population Size  (number of particles = 2*(m-1)*PI_O)
W_O=1 # number of workers (set to -1 to use maximum available workers)
PO_O=1 # Boolean: whether to Polish the optimization using scipy.optimize.minimize (trust-constr to obey bounds)



# one accessory concentration shared across all targets
# different output weights for each target
python ../code/run_script.py \
    --acc_opt 'outer' \
    --w_opt 'inner' \
    --target_lib_name $TL \
    --target_lib_file $TF \
    --maxiter_K $MI_K \
    --popsize_K $PI_K \
    --polish_K $PO_K \
    --workers_K $W_K \
    --maxiter_O $MI_O \
    --popsize_O $PI_O \
    --polish_O $PO_O \
    --workers_O $W_O \
    --base_dir $base_dir \
    --m $m \
    --n_input_samples $N \
\

# different accessory concentrations for each target
# one output weighting shared across all targets
python ../code/run_script.py \
    --acc_opt 'inner' \
    --w_opt 'outer' \
    --target_lib_name $TL \
    --target_lib_file $TF \
    --maxiter_K $MI_K \
    --popsize_K $PI_K \
    --polish_K $PO_K \
    --workers_K $W_K \
    --maxiter_O $MI_O \
    --popsize_O $PI_O \
    --polish_O $PO_O \
    --workers_O $W_O \
    --base_dir $base_dir \
    --m $m \
    --n_input_samples $N \
\

# different accessory concentrations for each target
# different output weights for each target
python ../code/run_script.py \
    --acc_opt 'inner' \
    --w_opt 'inner' \
    --target_lib_name $TL \
    --target_lib_file $TF \
    --maxiter_K $MI_K \
    --popsize_K $PI_K \
    --polish_K $PO_K \
    --workers_K $W_K \
    --maxiter_O $MI_O \
    --popsize_O $PI_O \
    --polish_O $PO_O \
    --workers_O $W_O \
    --base_dir $base_dir \
    --m $m \
    --n_input_samples $N \
\
