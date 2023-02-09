import os, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from tqdm import tqdm
import argparse

# If you need to install eqtk, you can uncomment and run the cell below.
# Or check out https://eqtk.github.io/getting_started/eqtk_installation.html.
# It can take several seconds to load the package into python.
# Additionally, the first time you run the equilibrium solver (eqtk.solve)
# may take longer than subsequent calls as the program compiles the code with Numba
#!{sys.executable} -m pip install --upgrade eqtk
import eqtk

from utilities import dump_data, load_data, make_tanh_features, DimerNets, my_regressor
from pdb import set_trace as bp


parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default='.', type=str) # base directory for output
parser.add_argument("--job_id", default=-1, type=int)
parser.add_argument("--n_input_dim", default=2, type=int)
parser.add_argument("--n_input_samples", default=1000, type=int)
parser.add_argument("--n_tanh_features", default=100, type=int)
parser.add_argument("--n_monomers", default=10, type=int)
FLAGS = parser.parse_args()

def make_plots(x, y, savepath, y_pred=None, features=None):

    print('x.shape:', x.shape)
    print('y.shape:', y.shape)

    # plot the target function
    ax.scatter(x[:,0], x[:,1], c=y, alpha=0.5)

    # plot the fit
    if y_pred is not None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.scatter(x[:,0], x[:,1], c=y_pred, alpha=0.5)
        ax.set_title('Fit')
        plt.close()

        # plot residuals of fit
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.scatter(x[:,0], x[:,1], c=y_pred - y, alpha=0.5)
        ax.set_title('Residuals of fit')
        plt.close()

## basic parameters
def main(n_input_dim=2, n_input_samples=1000, n_tanh_features=1000, n_monomers=10, base_dir = '.', **kwargs):

    output_dir = os.path.join(base_dir, './inputDim-{}_samplesX-{}_m-{}_tanhN-{}'.format(n_input_dim, n_input_samples, n_monomers, n_tanh_features))
    os.makedirs(output_dir, exist_ok=True)

    pkl_path = os.path.join(output_dir, 'model_info.pkl')
    try:
        data = load_data(pkl_path)
        print('Successfully loaded model data!\n')
    except:
        print('Could not load existing data. Generating data and features...\n')
        ## Pick your domain discretization here
        x = np.random.uniform(low=0.1,high=2, size=(n_input_samples, n_input_dim)) # sample the input domain

        ## Pick your target function here
        y = np.sin(np.sum(x**2, axis=1))**2 # target function is sin(||x||)^2 (note this holds for any dimensional x)

        ## Make tanh random features
        F_tanh = make_tanh_features(n_input_dim, n_tanh_features, x)

        ## Make Dimer-based random features
        dn = DimerNets(n_monomers=n_monomers, n_input=n_input_dim)
        dn.generate_random_params()
        # dn.sparsify(num_edges=110)
        t0 = time()
        F_dimers = dn.make_dimer_features(x)
        tf = time()-t0
        print('Took {} seconds to generate data-to-dimer features (m={})'.format(tf, n_monomers))
        print('F_dimers.shape:',F_dimers.shape)
        print('K:', dn.K)
        print('acc:', dn.c0_acc)

        ## save the data
        data = {'x': x,
                'y': y,
                'F_dimers': F_dimers,
                'F_tanh': F_tanh,
                'tf': tf}
        dump_data(data, pkl_path)


    ### Run regressions on Tanh features
    alpha = 1e-5
    print('Running tanh-RF regression with alpha={}'.format(alpha))
    # plotdir = os.path.join(output_dir, 'tanh_plots_alpha{}'.format(alpha))
    clf_tanh, y_pred_tanh, residuals_tanh = my_regressor(data['F_tanh'], data['y'], alpha)
    print('mean of coeffs:', np.mean(clf_tanh.coef_))
    print('sd of coeffs:', np.std(clf_tanh.coef_))
    # make_plots(x, y, plotdir, y_pred=None, features=None)

    ### Run regressions on Dimer features
    alpha = 1e-5
    for alpha_exp in range(1,10):
        alpha = np.float_power(10, -alpha_exp)
        print('Running dimer-RF regression with alpha={}'.format(alpha))
        clf_dimers, y_pred_dimers, residuals_dimers = my_regressor(data['F_dimers'], data['y'], alpha, positive=False)
        print('\tmean of coeffs:', np.mean(clf_dimers.coef_))
        print('\tsd of coeffs:', np.std(clf_dimers.coef_))


if __name__ == "__main__":
    if FLAGS.job_id >= 0:
        m_list = [3,5,10,20,30,100,200,500,1000]
        FLAGS.n_monomers = m_list[FLAGS.job_id]

    print('### Running for m = {}'.format(FLAGS.n_monomers))
    main(**FLAGS.__dict__)
