import os, sys

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import pickle
import functools
import numpy as np
import pandas as pd
import scipy.stats
import itertools
from scipy.optimize import minimize, brute, differential_evolution

from pymoo.problems.functional import FunctionalProblem
from pymoo.optimize import minimize
from opt_utils import minimize_wrapper
from pymoo.util.termination.default import SingleObjectiveDefaultTermination
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS

from opt_utils import AnalyzePymoo

# %load_ext watermark

from pdb import set_trace as bp
import time

# If you need to install eqtk, you can uncomment and run the cell below.
# Or check out https://eqtk.github.io/getting_started/eqtk_installation.html.
# It can take several seconds to load the package into python.
# Additionally, the first time you run the equilibrium solver (eqtk.solve)
# may take longer than subsequent calls as the program compiles the code with Numba
#!{sys.executable} -m pip install --upgrade eqtk
import eqtk

#load some utility functions
# pwd = os.path.abspath('..')
# sys.path.append(os.path.join(pwd, 'code/'))
from utilities import *
from makefuncs import set_target_library

plt.rcParams.update({'font.size': 22, 'legend.fontsize': 12,
                'legend.facecolor': 'white', 'legend.framealpha': 0.8,
                'legend.loc': 'upper left', 'lines.linewidth': 4.0})

default_colors = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])

class TuneK:
    def __init__(self,
                    base_dir = '../optimization_results',
                    target_lib_name = 'SinCos',
                    target_lib_file = None,
                    acc_opt = 'inner',
                    w_opt = 'inner',
                    opt_settings_outer = {}, # use default settings for differential evolution
                    m = 3,
                    n_input_samples = 40, #discretization of first monomer...time complexity scales linearly w.r.t. this parameter
                    n_accessory_samples = 75, # number of latin-hypercube samples from the space of initial accessory concentrations...time complexity scales linearly w.r.t. this parameter
                    input_lb = -3,
                    input_ub = 3,
                    acc_lb = -3,
                    acc_ub = 3,
                    centered = True,
                    param_lb = -6,
                    param_ub = 6,
                    plot_inner_opt = True,
                    **kwargs):
        """
        Run simulations for dimer networks of size m and input titration size t
        with k different parameter universes.

        Parameters
        ----------
        m : int.
            Number of monomer species.
        n_input_samples : int. Default 10.
            Number of values to titrate the input monomer species.
            Values spaced evenly on a log10 scale
        n_accessory_samples : int. Default 10.
            Number of values to titrate the accessory monomer species concentrations.
            This is performed with either LatinHypercube sampling
            OR
            denotes the maximum evaluations of an optimization over accessory concentrations.
            Values spaced evenly on a log10 scale.
        input_lb : int. Default -3
            lower bound for titrating the input monomer species. log10 scale
        input_ub : int. Default 3
            upper bound for titrating the input monomer species. log10 scale
        param_lb : int. Default -6
            Lower bounds for binding affinity parameters.
        param_ub : int. Default 6
            Upper bounds for binding affinity parameters.
        centered : bool. Default True.
            Indicates whether to center the draws from the latin-hypercube sampler
        base_dir : string. Default 'results'
            relative path for saving figure.
        target_lib_name : string. Default 'SinCos'
            Name of specially designed library to which we will fit.
        """
        self.base_dir = base_dir
        self.m = m
        self.n_input_samples = n_input_samples
        self.n_accessory_samples = n_accessory_samples
        self.input_lb = input_lb
        self.input_ub = input_ub
        self.acc_lb = acc_lb
        self.acc_ub = acc_ub
        self.centered = centered

        self.param_lb = param_lb
        self.param_ub = param_ub
        self.opt_settings_outer = opt_settings_outer

        self.plot_inner_opt = plot_inner_opt

        self.setup()

        self.acc_opt = acc_opt
        self.w_opt = w_opt
        self.set_opts()

        self.make_output_dir()

        self.f_targets = set_target_library(n_input_samples=n_input_samples, target_lib_name=target_lib_name, target_lib_file=target_lib_file)
        self.n_targets = self.f_targets.shape[0]

    def mse(self, f_true, f_pred):
        '''Mean Squared Error'''
        return np.mean( (f_true - f_pred)**2 )

    def make_output_dir(self):
        t = time.localtime()
        timestamp = time.strftime('%m%d%Y%H%M%S', t)
        base_nm = self.make_experiment_name()
        self.output_dir = os.path.join(self.base_dir, timestamp + '_' + base_nm)
        os.makedirs(self.output_dir)

    def make_experiment_name(self):
        foo = {'acc': self.acc_opt, 'weights': self.w_opt}
        mystr = '_'.join([key + '-' + str(foo[key]) for key in foo])
        return mystr

    def setup(self):
        self.Knames = np.array(make_Kij_names(n_input=1, n_accesory=self.m-1))
        self.N = make_nXn_stoich_matrix(self.m)
        self.num_rxns = self.N.shape[0]
        self.M0_min = [self.input_lb] + [0] * (self.m-1)
        self.M0_max = [self.input_ub] + [0] * (self.m-1)
        self.num_conc = [self.n_input_samples] + [1] * (self.m-1)

        self.C0 = make_C0_grid(self.m, M0_min=self.M0_min, M0_max=self.M0_max, num_conc=self.num_conc)

        self.num_params = self.num_rxns + (self.m-1)

        self.acc_monomer_ind = np.arange(1,self.m)

        #Parameter bounds for sampling and for titration
        if self.param_lb is None:
            self.param_lb = [-6]*self.num_rxns + [-3]*(self.m-1)

        if self.param_ub is None:
            self.param_ub = [6]*self.num_rxns + [3]*(self.m-1)

        self.acc_lb_list = [self.acc_lb]*(self.m-1)
        self.acc_ub_list = [self.acc_ub]*(self.m-1)
        # self.acc_list = sample_concentrations(self.m-1, self.n_accessory_samples, self.acc_lb_list, self.acc_ub_list, do_power=False) # 75 (number of draws) x 2 (number of accessories)

        self.param_lb = [self.param_lb]*self.num_rxns
        self.param_ub = [self.param_ub]*self.num_rxns


    def optimize_binding(self, popsize=15, maxiter=10, acc_opt="outer", w_opt="inner"):
        self.opt = differential_evolution(self.loss_k,
                disp = True,
                bounds = np.vstack((self.param_lb, self.param_ub)).T,
                maxiter = maxiter,
                popsize = popsize,
                workers = 1)
        self.k_opt = self.opt.x
        self.k_opt_loss = self.opt.fun

        # make plots and check the solution
        print('\n\n#### Now running/plotting final optimal values... ####')
        self.loss_k(self.k_opt, final_run=True)

        return foo

    def loss_k(self, K, final_run=False, verbose=False, normalize_plot=False):
        mse_best, mse_list_best, c0_acc_best, theta_best = self.outer_opt(K)
        if verbose or final_run:
            print('MSE:', mse_best)
            print('MSE per target:', mse_list_best)
            print('Accessory concentrations:', c0_acc_best)
            print('log(K):', K)
            print('theta:', theta_best)

        if final_run:
            N_plot_dimers = min(10, len(K))
            if N_plot_dimers == len(K):
                dimer_inds = np.arange(N_plot_dimers)
            else:
                dimer_inds = np.random.choice(np.arange(len(K)), size=N_plot_dimers, replace=False)

            N_plot_targets = min(10, self.n_targets)
            fig, axs = plt.subplots(nrows=3, ncols=N_plot_targets, figsize = [N_plot_targets*10,20])
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
            if N_plot_targets == self.n_targets:
                target_inds = np.arange(N_plot_targets)
            else:
                target_inds = np.random.choice(np.arange(self.n_targets), size=N_plot_targets, replace=False)
            for cc in range(N_plot_targets):
                j = target_inds[cc]
                if self.acc_opt=='inner':
                    c0_acc = c0_acc_best[j]
                else:
                    c0_acc = c0_acc_best

                if self.w_opt=='inner':
                    theta = theta_best[j]
                else:
                    theta = theta_best

                f_hat = self.predict(K, c0_acc, theta)
                axs[0,cc].plot(f_hat, label='Fit')
                axs[0,cc].plot(self.f_targets[j], label='Target', color='black')
                axs[0,cc].set_title('Fitting target-{}: MSE {}'.format(j, round(self.mse(f_hat, self.f_targets[j]), 3)))
                axs[0,cc].legend()

                # compute dimers
                dimer_names = self.Knames[dimer_inds]
                output_dimers = self.g1(c0_acc, K)[:,dimer_inds]
                max_od = np.max(output_dimers, axis=0)

                if normalize_plot:
                    axs[1,cc].plot(output_dimers/max_od, label=dimer_names)
                    axs[1,cc].set_title('Output Dimers (Max-Normalized)')
                    y = theta[dimer_inds]*max_od
                else:
                    axs[1,cc].plot(output_dimers, label=dimer_names)
                    axs[1,cc].set_title('Output Dimers')
                    axs[1,cc].set_yscale('log')
                    y = theta[dimer_inds]

                axs[1,cc].legend()

                x = np.flipud(np.argsort(y))
                y_sorted = y[x]
                xseq = np.arange(len(x))
                axs[2,cc].bar(xseq, y_sorted, color=default_colors[x])
                axs[2,cc].set_xticks(xseq)
                axs[2,cc].set_xticklabels(dimer_names[x])
                if normalize_plot:
                    axs[2,cc].set_title('Normalized Dimer Weights')
                else:
                    axs[2,cc].set_title('Dimer Weights')
                    axs[2,cc].set_yscale('log')

            fig.savefig(os.path.join(self.output_dir, 'optimization_results.pdf'), format='pdf')

            # plot output dimers
            if self.acc_opt=='outer':
                fig, axs = plt.subplots(nrows=1, ncols=1, figsize = [10,10])
                output_dimers = self.g1(c0_acc, K)
                axs.plot(output_dimers)
                axs.set_title('Output Dimers')
                axs.set_yscale('log')
                fig.savefig(os.path.join(self.output_dir, 'output_dimers.pdf'), format='pdf')

            # plot accessory values
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize = [14,8])
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.6)
            sns.heatmap(ax=axs, data=np.array(c0_acc_best).reshape(-1, self.m-1), cmap="viridis", vmin=self.acc_lb, vmax=self.acc_ub)
            axs.set_xticklabels(np.arange(2,self.m+1))
            axs.set_ylabel('Target Function Index')
            axs.set_xlabel('Accessory Monomer Index')
            axs.set_title('Optimized Accessory Monomer Concentrations (log10)')
            fig.savefig(os.path.join(self.output_dir, 'accessory_concentrations.pdf'), format='pdf')

            ## plot optimal K
            mask = np.tril(np.ones(self.m),-1) # creating mask
            Kmatrix = make_K_matrix(K, self.m)
            # plotting a triangle correlation heatmap
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize = [10,10])
            sns.heatmap(ax=axs, data=Kmatrix, cmap="viridis", mask=mask, vmin=self.param_lb[0], vmax=self.param_ub[0])
            axs.set_xticklabels(np.arange(1,self.m+1))
            axs.set_yticklabels(np.arange(1,self.m+1))
            axs.set_title('Optimized Binding Affinity Matrix (log10)')
            fig.savefig(os.path.join(self.output_dir, 'logK.pdf'), format='pdf')

            # save results
            out = {'MSE': mse_best,
                    'MSE-per-target': mse_list_best,
                    'a': c0_acc_best,
                    'theta': theta_best,
                    'logK': K,
                    'K': np.float_power(10, K)}

            with open(os.path.join(self.output_dir, 'info.pkl'), 'wb') as f:
                pickle.dump(out, f)

        return mse_best

    def set_opts(self):

        self.termination = SingleObjectiveDefaultTermination(
                            x_tol=1e-6,
                            cv_tol=0.0,
                            f_tol=1e-6,
                            nth_gen=5,
                            n_last=20,
                            n_max_gen=self.opt_settings_outer['maxiter'])

        if self.acc_opt=='outer':
            n_var = self.m-1
        elif self.acc_opt=='inner':
            n_var = self.n_targets * (self.m-1)

        self.algorithm = DE(
            pop_size=self.opt_settings_outer['popsize']*n_var,
            sampling=LHS(),
            variant="DE/rand/1/bin",
            CR=0.9,
            dither="vector",
            jitter=False)


        if self.acc_opt=='inner' and self.w_opt=='inner':
            # SUM_j min_(a_j, theta_j) |F_j - G(k; a_j, theta_j)|
            def outer_opt(K):
                theta_star = np.zeros((self.n_targets, len(K)))
                c0_acc_star = np.zeros((self.n_targets, self.m-1))
                mse_total = 0
                mse_list = [0 for j in range(self.n_targets)]
                for j in range(self.n_targets):
                    problem = FunctionalProblem(
                        n_var=n_var,
                        objs=lambda x: self.inner_opt(x, K, j)[1],
                        xl=self.acc_lb_list,
                        xu=self.acc_ub_list)

                    opt = minimize_wrapper(
                        problem,
                        self.algorithm,
                        self.termination,
                        save_history=True,
                        verbose=True,
                        plot_dirname=os.path.join(self.output_dir,'inner_opt'))

                    c0_acc_star_j = opt.X
                    c0_acc_star[j] = c0_acc_star_j
                    # rerun the best run to get more details
                    theta_star_j, mse_j = self.inner_opt(c0_acc_star_j, K, j)
                    theta_star[j] = theta_star_j
                    mse_list[j] = mse_j
                    mse_total += mse_j
                return mse_total, mse_list, c0_acc_star, theta_star

            def inner_opt(c0_acc, K, j_target):
                '''Compute the optimal weights for each target function given fixed K and accessorry concentrations'''
                dimers = self.g1(c0_acc, K) # 40 x 9
                theta_star_j, errs_j = scipy.optimize.nnls(dimers, self.f_targets[j_target]) # returns L2 errors (sqrt of sum of squares)
                mse_j = errs_j**2 / self.n_input_samples
                return theta_star_j, mse_j


        elif self.acc_opt=='inner' and self.w_opt=='outer':
            # min_theta SUM_j min_(a_j) |F_j - G(k; a_j, theta)|
            ## or, equivalently ##
            # min_A SUM_j |F_j - G(k; A_j, theta(A))|
            def inner_opt(c0_acc_list, K):
                '''Compute the optimal weights for across all target functions given fixed K and per-target accessorry concentrations'''
                dimers_all = []
                for j in range(self.n_targets):
                    i_low = j*(self.m-1)
                    i_high = i_low + (self.m-1)
                    dimers_j = self.g1(c0_acc_list[i_low:i_high], K) # 40 x 9
                    dimers_all += [dimers_j]

                targets_all = self.f_targets.reshape(-1) #(40*n,)
                dimers_all = np.vstack(dimers_all)
                # theta_star, errs_all = scipy.optimize.nnls(dimers_all, targets_all) # returns L2 errors (sqrt of sum of squares)
                # instead of NNLS, use lsq_linear because it returns pointwise residuals automatically, rather than just overall MSE.
                foo = scipy.optimize.lsq_linear(dimers_all, targets_all, bounds=(0, np.Inf), method='bvls')
                theta_star = foo.x
                mse_total = np.sum(foo.fun**2) / self.n_input_samples

                # compute listed MSEs
                mse_list = [0 for j in range(self.n_targets)]
                for j in range(self.n_targets):
                    i_low = j*(self.m-1)
                    i_high = i_low + (self.m-1)
                    mse_list[j] = np.sum(foo.fun[i_low:i_high]**2) / self.n_input_samples

                return theta_star, mse_total, mse_list

            def outer_opt(K):
                '''Optimize accessory concentrations for the fixed K.'''

                problem = FunctionalProblem(
                    n_var=n_var,
                    objs=lambda x: self.inner_opt(x, K)[1],
                    xl=[self.acc_lb]*(self.m-1)*self.n_targets,
                    xu=[self.acc_ub]*(self.m-1)*self.n_targets)

                opt = minimize_wrapper(
                    problem,
                    self.algorithm,
                    self.termination,
                    save_history=True,
                    verbose=True,
                    plot_dirname=os.path.join(self.output_dir,'inner_opt'))

                # opt = differential_evolution(lambda x: self.inner_opt(x, K)[1],
                #         bounds = np.vstack(([self.acc_lb]*(self.m-1)*self.n_targets, [self.acc_ub]*(self.m-1)*self.n_targets)).T,
                #         **self.opt_settings_outer)
                c0_acc_best = opt.X
                # rerun the best run to get more details
                theta_best, mse_total_best, mse_list_best = self.inner_opt(c0_acc_best, K)

                # convert c0_acc_best to a shaped array
                c0_acc_best = c0_acc_best.reshape(self.n_targets,-1)
                return mse_total_best, mse_list_best, c0_acc_best, theta_best

        elif self.acc_opt=='outer' and self.w_opt=='inner':
            # min_a SUM_j min_(theta_j) |F_j - G(k; a, theta_j)|
            # perhaps simplest computationally?
            def inner_opt(c0_acc, K):
                '''Compute the optimal weights for each target function given fixed K and accessorry concentrations'''
                dimers = self.g1(c0_acc, K) # 40 x 9
                theta_star = np.zeros((self.n_targets, dimers.shape[1]))
                mse_total = 0
                mse_list = [0 for j in range(self.n_targets)]
                for j in range(self.n_targets):
                    opt_j, errs_j = scipy.optimize.nnls(dimers, self.f_targets[j]) # returns L2 errors (sqrt of sum of squares)
                    theta_star[j] = opt_j
                    mse_j = errs_j**2 / self.n_input_samples
                    mse_list[j] = mse_j
                    mse_total += mse_j # errs_j is l2 norm, so square it, then divide by N to get MSE
                return theta_star, mse_total, mse_list

            def outer_opt(K):
                '''Optimize accessory concentrations for the fixed K.'''

                problem = FunctionalProblem(
                    n_var=n_var,
                    objs=lambda x: self.inner_opt(x, K)[1],
                    xl=self.acc_lb_list,
                    xu=self.acc_ub_list)

                opt = minimize_wrapper(
                    problem,
                    self.algorithm,
                    self.termination,
                    save_history=True,
                    verbose=True,
                    plot_dirname=os.path.join(self.output_dir,'inner_opt'))

                c0_acc_best = opt.X
                # rerun the best run to get more details
                theta_best, mse_total_best, mse_list_best = self.inner_opt(c0_acc_best, K)
                return mse_total_best, mse_list_best, c0_acc_best, theta_best

        elif self.acc_opt=='outer' and self.w_opt=='outer':
            # min_(a, theta) SUM_j  |F_j - G(k; a, theta)|
            raise('Cannot do an outer-optimization for both accessories and weights. Quitting.')
        else:
            raise('Inner/Outer optimizations not correctly specified.')

        # allocate
        self.inner_opt = inner_opt
        self.outer_opt = outer_opt

        return

    def g1(self, c0_acc, K, apply_power_K=True, apply_power_c0=True):
        # for 1d -> 1d predictions, we have each row of C0 being the same EXCEPT in its first column,
        # where we modulate the input monomor concentration over a pre-defined range.
        # Note: evaluation of g1 scales linearly with number of rows in C0...eqtk must treat each row of C0 independently.
        if apply_power_K:
            K = np.float_power(10, K)
        if apply_power_c0:
            c0_acc = np.float_power(10, c0_acc)

        C0 = self.C0.copy()
        C0[:,self.acc_monomer_ind] = c0_acc # each row should get c0_acc
        sols = eqtk.solve(c0=C0, N=self.N, K=K)

        dimers = sols[:,self.m:]

        return dimers

    def g2(self, dimers, theta):
        f_out = dimers @ theta
        return f_out

    def predict(self, K, c0_acc, theta):
        dimers = self.g1(c0_acc, K)

        f_hat = self.g2(dimers, theta)
        return f_hat
