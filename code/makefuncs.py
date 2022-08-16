# import os, sys
# import matplotlib
# import matplotlib.pyplot as plt
# plt.rcParams.update({'font.size': 22, 'legend.fontsize': 12,
#                 'legend.facecolor': 'white', 'legend.framealpha': 0.8,
#                 'legend.loc': 'upper left', 'lines.linewidth': 4.0})
import scipy.interpolate
import numpy as np
from pdb import set_trace as bp

def set_target_library(n_input_samples=10, target_lib_name="SinCos", target_lib_file=None, bump_centers=[0.2, 0.4, 0.6], bump_width=2):
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
    elif target_lib_name=='bump':
        # this function uses relative bump centers over the input domain and bump width in units of the discretization points.
        # SO, if you change n_input_samples, the bump function will change (e.g. finer discretization => narrower bump).
        # to alleviate this, can make the bump_width wider.
        # in the future, should define bumps in real coordinates, THEN interpolate and determine discretizations afterwards.
        f_targets = bump_targets(bump_centers, bump_width, n_input_samples)
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

def bump_targets(bump_centers, bump_width, n_input_samples):
    n_targets = len(bump_centers)
    f_targets = np.zeros((n_targets, n_input_samples))
    for n in range(n_targets):
        i_mid = int(bump_centers[n]*n_input_samples)
        i_low = max(0, i_mid - bump_width)
        i_high = min(n_input_samples-1, i_mid + bump_width)
        f_targets[n,i_low:i_high] = 1 # assign the bump
    return f_targets

def bump_on(bump_starts, n_input_samples):
    n_targets = len(bump_starts)
    f_targets = np.zeros((n_targets, n_input_samples))
    for n in range(n_targets):
        i_low = int(bump_starts[n]*n_input_samples)
        f_targets[n,i_low:] = 1 # assign the bump
    return f_targets
