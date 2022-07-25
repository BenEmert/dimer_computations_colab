# import os, sys
# import matplotlib
# import matplotlib.pyplot as plt
# plt.rcParams.update({'font.size': 22, 'legend.fontsize': 12,
#                 'legend.facecolor': 'white', 'legend.framealpha': 0.8,
#                 'legend.loc': 'upper left', 'lines.linewidth': 4.0})
# import pickle
import scipy.interpolate
import numpy as np
from pdb import set_trace as bp

def set_target_library(n_input_samples=10, target_lib_name="SinCos", target_lib_file=None):
    '''Generate a library of functions to which we will fit or use to measure expressivity.'''
    if target_lib_name=='SinCos':
        x = np.linspace(0, 2*np.pi, n_input_samples)
        target_function_sin = 0.5*(np.sin(x)+1)
        target_function_cos = 0.5*(np.cos(x)+1)

        f_targets = np.vstack((target_function_sin, target_function_cos))
    elif target_lib_name=='MetaClusters':
        with open(target_lib_file, 'rb') as f:
            F = np.load(f) # n_clusters x discretization
            f_targets = interp_target(n_input_samples, F)
    else:
        raise('library name "{}" not recognized. Quitting.'.format(target_lib_name))

    return f_targets

def interp_target(n_input_samples, F):
    ''''''
    n_clusters, n_pts = F.shape
    grid = np.linspace(0, 1, n_pts)
    target_grid = np.linspace(0, 1, n_input_samples)
    f_interp = scipy.interpolate.interp1d(grid, F, kind='cubic')
    f_targets = f_interp(target_grid)
    return f_targets
