import os, sys
sys.path.append('../code')
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pymoo.optimize import minimize
import scipy.optimize
from utilities import *
from pdb import set_trace as bp

plt.rcParams.update({'font.size': 22, 'legend.fontsize': 12,
                'legend.facecolor': 'white', 'legend.framealpha': 0.8,
                'legend.loc': 'upper left', 'lines.linewidth': 4.0})

default_colors = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])


def minimize_wrapper(problem, algorithm, termination, seed=None, save_history=True, verbose=True,
                    n_starts=2,
                    plot_convergence=True,
                    plot_comparisons=True,
                    plot_dirname='optplots',
                    polish=False,
                    truth = []):

    opt_list = []
    for n in range(n_starts):
        opt = minimize(
            problem,
            algorithm,
            termination,
            seed=seed,
            save_history=True,
            verbose=True)

        # optionally run a local optimizer to polish the optimization
        if polish:
            result = scipy.optimize.minimize(problem.objs[0],
                              opt.X,
                              method='trust-constr',
                              bounds=np.array([problem.xl, problem.xu]).T)
            opt.X = result.x
            opt.F = result.fun

        opt_list += [opt]

    new_plot_dirname = make_new_dir(plot_dirname)
    ap = AnalyzePymoo(opt_list, truth=truth)
    ap.make_plots(new_plot_dirname)

    return opt_list

class AnalyzePymoo:
    def __init__(self, opt_list,
                    xnames = None,
                    truth = None,
                    **kwargs):
        self.opt_list = opt_list
        self.xnames = xnames

        self.truth = truth

        self.extract_info()

    def extract_info(self):

        # get opt sequence
        self.X = []
        self.F = []
        self.n_params = self.opt_list[0].opt.get("X").shape[1]
        for n in range(len(self.opt_list)):
            history = self.opt_list[n].history
            self.X += [np.vstack([e.opt.get("X") for e in history])]
            self.F += [np.array([e.opt[0].F for e in history])]

        # get index for iterations
        self.n_evals = np.array([e.evaluator.n_eval for e in history]) # only need 1, and they are identical

        # convert to np array (1st coord indexes the independent optimization).
        self.X = np.array(self.X)
        self.F = np.array(self.F)

        if self.X.ndim==2:
            self.X = np.expand_dims(self.X, 0)
            self.F = np.expand_dims(self.F, 0)

        if self.xnames is None:
            self.xnames = ['P{}'.format(c) for c in range(self.X.shape[-1])]

    def write_info(self, writedir):
        dump = {'X': self.X, 'F': self.F, 'n_evals': self.n_evals, 'truth': self.truth}
        dump_data(dump, os.path.join(writedir, 'opt_info.pkl'))


    def make_plots(self, plotdir):
        os.makedirs(plotdir, exist_ok=True)

        # write monitoring data to file
        self.write_info(plotdir)

        # compare across optimizations
        self.plot_loss(plotdir)

        self.compare_params(plotdir)

        # plot per-optimization:
        for n in range(len(self.opt_list)):
            pdir = os.path.join(plotdir, 'run_{}'.format(n))
            os.makedirs(pdir, exist_ok=True)
            self.plot_params(self.X[n], pdir)


    def compare_params(self, plotdir, nm='a0'):
        fig, axs = plt.subplots(nrows=1, ncols=self.n_params, figsize = [self.n_params*10,8])
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.6)

        for n in range(len(self.opt_list)):
            for c in range(self.X.shape[-1]):
                axs[c].plot(self.n_evals, self.X[n,:,c], label='Run {}'.format(n))
                axs[c].set_title("Parameter {}".format(c))
                axs[c].set_xlabel("Iters")

        for c in range(self.X.shape[-1]):
            if len(self.truth):
                axs[c].axhline(y=self.truth[c], color='black', linestyle='--', label='True')
            axs[c].legend()

        # axs.set_title("Convergence")
        # axs.plot(self.n_evals, X, label=self.xnames)
        # axs.legend()
        fig.savefig(os.path.join(plotdir, 'X_convergence_comparison.pdf'), format='pdf')
        plt.close()

    def plot_loss(self, plotdir):
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize = [14,8])
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.6)

        for n in range(len(self.opt_list)):
            axs.plot(self.n_evals, self.F[n], "--", label="Run {}".format(n))

        axs.legend()
        axs.set_title("Convergence")
        fig.savefig(os.path.join(plotdir, 'fval_convergence.pdf'), format='pdf')
        plt.close()

    def plot_params(self, X, plotdir):
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize = [14,8])
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.6)

        axs.set_title("Convergence")
        axs.plot(self.n_evals, X, label=self.xnames)

        if len(self.truth):
            for c in range(self.X.shape[-1]):
                axs.axhline(y=self.truth[c], color=default_colors[c], linestyle='--', label='True')


        axs.legend()
        fig.savefig(os.path.join(plotdir, 'X_convergence.pdf'), format='pdf')
        plt.close()


def analyze_convergence(inputdir):

    dirlist = ['inner_opt_{}'.format(c) for c in range(10)]
    X = []
    F = []
    Fstd = []
    Xstd = []
    for dir in dirlist:
        res = load_data(os.path.join(inputdir, dir, "opt_info.pkl"))
        X += [res['X']]
        F += [res['F']]
        Fstd += [np.std(res['F'],axis=0)]
        Xstd += [np.sum(np.std(res['X'],axis=0), axis=1)]
    n_evals = res['n_evals']

    X = np.vstack(X)
    F = np.vstack(F).squeeze()
    Fstd = np.hstack(Fstd).T
    Xstd = np.vstack(Xstd)

    # plot F optimization variancex
    df = pd.DataFrame(Fstd.T)
    df['iteration'] = n_evals
    df = df.melt(id_vars='iteration')
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize = [14,8])
    sns.lineplot(ax=axs, data=df, x='iteration', y='value')
    fig.savefig(os.path.join(inputdir, 'Fstd_convergence.pdf'), format='pdf')

    # plot X optimization variancex
    df = pd.DataFrame(Xstd.T)
    df['iteration'] = n_evals
    df = df.melt(id_vars='iteration')
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize = [14,8])
    sns.lineplot(ax=axs, data=df, x='iteration', y='value')
    fig.savefig(os.path.join(inputdir, 'Xstd_convergence.pdf'), format='pdf')

    # plot F optimization
    df = pd.DataFrame(F.T)
    df['iteration'] = n_evals
    df = df.melt(id_vars='iteration')
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize = [14,8])
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.6)
    sns.lineplot(ax=axs, data=df, x='iteration', y='value')
    fig.savefig(os.path.join(inputdir, 'F_convergence.pdf'), format='pdf')

    # plot X optimization
    dfX = pd.DataFrame()
    for p in range(X.shape[-1]):
        pname = 'A{}'.format(p+1)
        for iter in range(X.shape[1]):
            for rn in range(X.shape[0]):
                foo = pd.DataFrame([{'iteration':iter, 'rn': rn, 'parameter': pname, 'value': X[rn,iter,p]}])
                dfX = pd.concatenate(dfX, foo)
    dfX = dfX.reset_index()
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize = [14,8])
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.6)
    sns.lineplot(ax=axs, data=dfX, x='iteration', y='value', hue='parameter', ci='sd')
    fig.savefig(os.path.join(inputdir, 'X_convergence.pdf'), format='pdf')
