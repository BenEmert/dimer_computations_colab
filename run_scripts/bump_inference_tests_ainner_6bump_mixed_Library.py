import os, sys

import numpy as np

sys.path.append('../code')

from utilities import *
from makefuncs import *
from opt_utils import AnalyzePymoo, minimize_wrapper, analyze_convergence
from pymoo.algorithms.soo.nonconvex.de import DE

# from scipy.optimize import minimize, brute, differential_evolution
# from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.problems.functional import FunctionalProblem
from pymoo.optimize import minimize
from pymoo.util.termination.default import SingleObjectiveDefaultTermination
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from tuning import TuneK
from pdb import set_trace as bp
import argparse
from time import time

parser0 = argparse.ArgumentParser()
## Run settings
parser0.add_argument('--n_rounds', default=1, type=int)
# parser0.add_argument('--n_targets_per_round', default=1, type=int)
parser0.add_argument('--bump_center', default=0.8, type=float) # relative across logX input domain
parser0.add_argument('--bump_width', default=1, type=int) # units of discretization
FLAGS_run, __ = parser0.parse_known_args()

parser1 = argparse.ArgumentParser()
## Settings for TuneK
parser1.add_argument('--base_dir', default='../results/bumps_mixed_6library/a-inner_w-outer/', type=str) # base directory for output
parser1.add_argument('--target_lib_name', default='SinCos', type=str) # Name for target library
parser1.add_argument('--target_lib_file', default='../data/metaclusters/hc_3M_metaClusterBasis_thresh3.npy', type=str) # file for reading target functions
parser1.add_argument('--m', default=3, type=int) #number of total monomers
parser1.add_argument('--n_input_samples', default=40, type=int) #Number of values to titrate the input monomer species. Values spaced evenly on a log10 scale
parser1.add_argument('--acc_opt', default="inner", type=str)
parser1.add_argument('--w_opt', default="inner", type=str)
parser1.add_argument('--single_beta', default=0, type=int)
parser1.add_argument('--polish', default=0, type=int)
parser1.add_argument('--lsq_linear_method', default='bvls', type=str)
parser1.add_argument('--plot_inner_opt', default=0, type=int)
FLAGS_tune, __ = parser1.parse_known_args()

# Settings for Differential Evolution optimizer within TuneK (i.e. for a single call of loss(K))
parser3 = argparse.ArgumentParser()
parser3.add_argument('--maxiter_O', default=15, type=int)
parser3.add_argument('--popsize_O', default=50, type=int)
parser3.add_argument('--polish_O', default=1, type=int)
parser3.add_argument('--workers_O', default=1, type=int) # default is to use 1 worker (not paralleized). -1 uses all available workers!
FLAGS_optsetts, __ = parser3.parse_known_args()
tune_dict = FLAGS_tune.__dict__
tune_dict['opt_settings_outer'] = make_opt_settings(FLAGS_optsetts.__dict__)

# Settings for Differential Evolution optimizer over binding affinity K
parser2 = argparse.ArgumentParser()
parser2.add_argument('--maxiter_K', default=3, type=int)
parser2.add_argument('--popsize_K', default=20, type=int)
parser2.add_argument('--polish_K', default=0, type=int)
parser2.add_argument('--workers_K', default=1, type=int) # default is to use 1 worker (not paralleized). -1 uses all available workers!
FLAGS_diffev, __ = parser2.parse_known_args()
opt_setts_K = make_opt_settings(FLAGS_diffev.__dict__)

def opt_wrapper(opt_setts, n_rounds=1, bump_center=0.5, bump_width=2, **kwargs):

    FOO = TuneK(**tune_dict)
    # n_input_samples = FOO.n_input_samples
    # F = np.zeros((1, n_input_samples))
    # i_mid = int(bump_center*n_input_samples)
    # i_low = max(0, i_mid - bump_width)
    # i_high = min(n_input_samples-1, i_mid + bump_width)
    # F[0,i_low:i_high] = 1 # assign the bump
    F1 = bump_on(bump_starts=[0.2, 0.5, 0.8], n_input_samples=FOO.n_input_samples)
    F2 = bump_targets(bump_centers=[0.2, 0.5, 0.8], bump_width=2, n_input_samples=FOO.n_input_samples)

    F = np.vstack((F1,F2))

    # set the target
    FOO.set_target(F) #wipes other targets and focuses on this one (interpolates, then re-discretizes)

    # run this after setting the target (only ESSENTIAL when doing acc_opt="inner")
    FOO.set_opts()

    for c in range(n_rounds):
        # next, perform full optimization over K and a, theta
        FOO.optimize_binding(seed=c, **opt_setts)
    return

######
if __name__ == '__main__':

    # analyze_convergence('../results/default/07252022105441_acc-outer_weights-inner')
    np.random.seed(0)
    t0 = time()
    opt_wrapper(opt_setts_K, **FLAGS_run.__dict__)
    # run overall analysis
    # analyze_convergence(FOO.output_dir)
    print('# Total run time = {}'.format(sec2friendly(time()-t0)))
