import sys
import argparse
import pandas as pd
sys.path.append('../code')
from utilities import dict_combiner
from bump_inference_tests_ainner_ManyBumps import opt_wrapper
from pdb import set_trace as bp

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default='../results/manybumps1switch', type=str) # base directory for output
parser.add_argument("--run_all", default=0, type=int)
parser.add_argument("--id", default=0, type=int)
FLAGS = parser.parse_args()

mydict = {
    "base_dir": [FLAGS.base_dir],
    "target_lib_name": ["bumps_all"],
    "start": ["on", "off"],
    "m": [3, 5],
    "acc_opt": ["inner"],
    "w_opt": ["inner", "outer"],
    "single_beta": [1],
    "one_scale": [1],
    "plot_inner_opt": [0],  # [0, 1],
    "maxiter_O": [10, 20],
    "popsize_O": [5, 10],
    "polish_O": [1],
    "maxiter_K": [10, 25],
    "popsize_K": [10, 25, 50],
    "polish_K": [1],
    "nstarts_K": [2],
}

EXPERIMENT_LIST = dict_combiner(mydict)

MAX_ITERS = 3e4

def run_main(sett):
    K_names = ['maxiter_K', 'popsize_K', 'polish_K', 'nstarts_K']
    sett_K = {k.split('_')[0]: sett[k] for k in K_names}

    tune_names = ['maxiter_K', 'popsize_K', 'polish_K']
    sett['opt_settings_outer'] = {k.split('_')[0]: sett[k] for k in tune_names}

    n_iters = sett_K['maxiter']*sett_K['popsize']*sett['opt_settings_outer']['maxiter']*sett['opt_settings_outer']['popsize']
    if n_iters <= MAX_ITERS:
        opt_wrapper(sett_K, tune_dict=sett)


if __name__ == "__main__":
    if FLAGS.run_all:
        df = pd.DataFrame()
        for sett in EXPERIMENT_LIST:
            try:
                run_main(sett)
                status = "FINISHED"
            except:
                status = "FAILED"
                # print('The following setting failed:')
                # print(sett)
            sett["status"] = status
            sett_df = pd.DataFrame([sett])
            df = pd.concat([df, sett_df])

            print(df.sort_values(by="status"))
        bp()
    else:
        settings = EXPERIMENT_LIST[FLAGS.id]
        run_main(settings)
