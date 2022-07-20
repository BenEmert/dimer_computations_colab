import os, sys
sys.path.append('../code')
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from pdb import set_trace as bp

plt.rcParams.update({'font.size': 22, 'legend.fontsize': 12,
                'legend.facecolor': 'white', 'legend.framealpha': 0.8,
                'legend.loc': 'upper left', 'lines.linewidth': 4.0})

default_colors = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])

class AnalyzePymoo:
    def __init__(self, result,
                    output_dir = '../optimization_results/defaults',
                    xnames = None,
                    **kwargs):
        self.history = result.history
        self.output_dir = output_dir
        self.xnames = xnames

    def make_plots(self):
        self.plot_loss()
        self.plot_params()

    def plot_params(self):
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize = [14,8])
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.6)

        n_evals = np.array([e.evaluator.n_eval for e in self.history])
        X = np.vstack([e.opt.get("X") for e in self.history])

        axs.set_title("Convergence")
        axs.plot(n_evals, X, label=self.xnames)
        axs.legend()
        fig.savefig(os.path.join(self.output_dir, 'X_convergence.pdf'), format='pdf')
        plt.close()

    def plot_loss(self):
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize = [14,8])
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.6)

        n_evals = np.array([e.evaluator.n_eval for e in self.history])
        opt = np.array([e.opt[0].F for e in self.history])

        axs.set_title("Convergence")
        axs.plot(n_evals, opt, "--")
        fig.savefig(os.path.join(self.output_dir, 'fval_convergence.pdf'), format='pdf')
        plt.close()

    def plot_x(self):
        pass
