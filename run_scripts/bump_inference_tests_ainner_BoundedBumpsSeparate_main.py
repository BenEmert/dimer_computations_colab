import os, sys
import argparse
import pandas as pd
sys.path.append('../code')
from utilities import dict_combiner
from bump_inference_tests_ainner_ManyBumps import opt_wrapper
from pdb import set_trace as bp

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default='../results/manybumps1switch_dev', type=str) # base directory for output
parser.add_argument("--dev_run", default=0, type=int)
parser.add_argument("--run_all", default=0, type=int)
parser.add_argument("--id", default=0, type=int)
parser.add_argument("--dothreading", default=0, type=int)
FLAGS = parser.parse_args()

mydict = {
    "base_dir": [FLAGS.base_dir],
    "target_lib_name": ["bumps_all"],
    "dimer_eps": [1e-16],
    "n_switches": [1],# 2],
    "n_switch_points": [3],
    "start": ["both"], #["on", "off", "both"],
    "id_target": [i for i in range(168)],
    "m": [3, 4, 5, 10, 20],
    "acc_opt": ["inner"],
    "w_opt": ["inner"],
    "single_beta": [1],
    "scale_type": ["per-dimer"], #"per-target"],
    "plot_inner_opt": [0],
    "maxiter_O": [10],
    "popsize_O": [50],
    "polish_O": [0],
    "maxiter_K": [1, 10, 25],
    "popsize_K": [100],
    "polish_K": [1],
    "nstarts_K": [2],
}

if FLAGS.dev_run:
    mydict['maxiter_O'] = [2]
    mydict['maxiter_K'] = [3]
    mydict['popsize_O'] = [20]
    mydict['popsize_K'] = [15]
    mydict['polish_K'] = [0]
    mydict['polish_O'] = [0]


EXPERIMENT_LIST = dict_combiner(mydict)

MAX_ITERS = 2e6

def namemaker(x):

    foo = [x[k] for k in ['n_switches', 'n_switch_points', 'acc_opt', 'w_opt', 'single_beta', 'scale_type', 'm', 'plot_inner_opt', 'id_target']]
    dirname = 'nswitches-{}_nswitchlocs-{}_a-{}_w-{}_singleBeta-{}_scaleType-{}_m-{}_plotInner-{}_targetID-{}'.format(*foo)

    goo = [x[k] for k in ['maxiter_O', 'popsize_O', 'polish_O', 'maxiter_K', 'popsize_K', 'polish_K', 'start']]
    fname = 'maxiterO-{}_popsizeO-{}_polishO-{}_maxiterK-{}_popsizeK-{}_polishK-{}_start-{}'.format(*goo)

    nm = os.path.join(dirname, fname)
    return nm

def run_main(sett, dothreading):
    sett['base_dir'] = os.path.join(sett['base_dir'], namemaker(sett))

    K_names = ['maxiter_K', 'popsize_K', 'polish_K', 'nstarts_K']
    sett_K = {k.split('_')[0]: sett[k] for k in K_names}
    sett_K['dothreading'] = dothreading

    tune_names = ['maxiter_O', 'popsize_O', 'polish_O']
    sett['opt_settings_outer'] = {k.split('_')[0]: sett[k] for k in tune_names}

    n_iters = sett_K['maxiter']*sett_K['popsize']*sett['opt_settings_outer']['maxiter']*sett['opt_settings_outer']['popsize']
    if n_iters <= MAX_ITERS:
        opt_wrapper(sett_K, tune_dict=sett)


if __name__ == "__main__":
    print('{} total experiments available'.format(len(EXPERIMENT_LIST)))
    if FLAGS.run_all:
        df = pd.DataFrame()
        for sett in EXPERIMENT_LIST:
            try:
                run_main(sett, dothreading=FLAGS.dothreading)
                status = "FINISHED"
            except:
                status = "FAILED"
                # print('The following setting failed:')
                # print(sett)
            sett["status"] = status
            sett_df = pd.DataFrame([sett])
            df = pd.concat([df, sett_df])

            print(df.sort_values(by="status"))
    else:
        settings = EXPERIMENT_LIST[FLAGS.id]
        run_main(settings, dothreading=FLAGS.dothreading)
