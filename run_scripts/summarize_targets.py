import os
import pickle
import numpy as np

base = '../results/BoundedBumps_1.0_m3/'
dir = 'nswitches-2_nswitchlocs-5_a-inner_w-inner_singleBeta-1_scaleType-per-dimer_m-3_plotInner-0_targetID-{}/maxiterO-10_popsizeO-25_polishO-1_maxiterK-25_popsizeK-10_polishK-1_start-both/10182022230141_acc-inner_weights-inner/FinalK_{}/run0/model_info.pkl'

D = 40 # number of input monomer concentrations
N = 883 # number of targets
S = 2 # number of optimizations per target

targets = np.zeros((N,D))
fits = np.zeros((N,D,S))

for n in range(N):
    for s in range(S):
        fname = os.path.join(base, dir.format(n, s))
        # try:
        with open(fname, 'rb') as f:
            x = pickle.load(f)

        if n==0:
            targets[n] = x['f_target']
        fits[n,:,s] = x['f_fit']
        # except:
        #     print('ID {}, round {} not available.'.format(n,s))
        #     pass

data = {'targets': targets, 'fits': fits}
summary_fname = os.path.join(base, 'fit_summary.pkl')
with open(summary_fname, 'a') as f:
    pickle.dump(data, f)
