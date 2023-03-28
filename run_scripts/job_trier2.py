import os, sys
import time
import numpy as np
import pickle
import shutil
import subprocess
import pandas as pd
from pdb import set_trace as bp

output_dir = "/groups/astuart/mlevine/dimer_computations_colab/results/BoundedBumps_randomK_jacobTarget_perDimer_9.0.0_devrun/maxiterO-2_popsizeO-2_polishO-0_maxiterK-1_popsizeK-1_polishK-0"
os.makedirs(output_dir, exist_ok=True)
log_file = os.path.join(output_dir, 'my.log')
master_file = os.path.join(output_dir, 'master_file.pkl')

sleep_secs = 3*60 # length of time (secs) to wait before trying to submit more jobs. Using 30min.

job_list = ['inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset53000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset52000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset51000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset50000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset49000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset48000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset47000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset46000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset45000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset44000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset43000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset42000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset41000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset40000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset39000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset38000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset37000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset36000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset35000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset34000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset33000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset32000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset31000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset30000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset29000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset28000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset27000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset26000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset25000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset24000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset23000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset22000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset21000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset20000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset19000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset18000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset17000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset16000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset15000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset14000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset13000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset12000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset11000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset10000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset9000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset8000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset7000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset6000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset5000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset4000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset3000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset2000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset1000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m12_offset0.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset45000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset44000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset43000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset42000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset41000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset40000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset39000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset38000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset37000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset36000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset35000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset34000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset33000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset32000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset31000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset30000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset29000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset28000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset27000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset26000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset25000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset24000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset23000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset22000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset21000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset20000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset19000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset18000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset17000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset16000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset15000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset14000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset13000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset12000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset11000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset10000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset9000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset8000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset7000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset6000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset5000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset4000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset3000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset2000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset1000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m11_offset0.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset37000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset36000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset35000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset34000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset33000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset32000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset31000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset30000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset29000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset28000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset27000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset26000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset25000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset24000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset23000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset22000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset21000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset20000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset19000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset18000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset17000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset16000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset15000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset14000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset13000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset12000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset11000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset10000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset9000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset8000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset7000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset6000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset5000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset4000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset3000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset2000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset1000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset0.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m9_offset29000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m9_offset28000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m9_offset27000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m9_offset26000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m9_offset25000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m9_offset24000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m9_offset23000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m9_offset22000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m9_offset21000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m9_offset20000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m9_offset19000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m9_offset18000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m9_offset17000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m9_offset16000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m9_offset15000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m9_offset14000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m9_offset13000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m9_offset12000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m9_offset11000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m9_offset10000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m9_offset9000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m9_offset8000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m9_offset7000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m9_offset6000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m9_offset5000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m9_offset4000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m9_offset3000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m9_offset2000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m9_offset1000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m9_offset0.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m8_offset22000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m8_offset21000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m8_offset20000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m8_offset19000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m8_offset18000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m8_offset17000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m8_offset16000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m8_offset15000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m8_offset14000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m8_offset13000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m8_offset12000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m8_offset11000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m8_offset10000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m8_offset9000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m8_offset8000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m8_offset7000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m8_offset6000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m8_offset5000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m8_offset4000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m8_offset3000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m8_offset2000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m8_offset1000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m8_offset0.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m7_offset17000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m7_offset16000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m7_offset15000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m7_offset14000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m7_offset13000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m7_offset12000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m7_offset11000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m7_offset10000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m7_offset9000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m7_offset8000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m7_offset7000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m7_offset6000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m7_offset5000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m7_offset4000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m7_offset3000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m7_offset2000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m7_offset1000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m7_offset0.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m6_offset12000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m6_offset11000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m6_offset10000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m6_offset9000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m6_offset8000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m6_offset7000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m6_offset6000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m6_offset5000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m6_offset4000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m6_offset3000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m6_offset2000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m6_offset1000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m6_offset0.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m5_offset8000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m5_offset7000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m5_offset6000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m5_offset5000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m5_offset4000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m5_offset3000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m5_offset2000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m5_offset1000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m5_offset0.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m4_offset4000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m4_offset3000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m4_offset2000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m4_offset1000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m4_offset0.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m3_offset2000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m3_offset1000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m3_offset0.job']

def run_cleanup(master_file, output_dir):
    if not os.path.exists(output_dir):
        print(output_dir, 'NOT YET CREATED. SKIPPING RUN CLEANUP.')
        return

    try:
        with open(master_file, 'rb') as f_master:
            master = pickle.load(f_master)
    except:
        # only to initialize (first time)
        master = {}

    # look for completed runs, then consolidate their output and delete the original run data.
    for run_dir in os.listdir(output_dir):
        info_file = os.path.join(output_dir, run_dir, 'model_info.pkl')
        experiment_key = os.path.split(run_dir)[-1]
        try:
            with open(info_file, 'rb') as f:
                # read experiment info
                model_info = pickle.load(f)

                #save experiment info to master dict
                master[experiment_key] = model_info

                # write master to file
                with open(master_file, 'wb') as f_master:
                    pickle.dump(master, f_master)

                # delete original dir
                shutil.rmtree(os.path.join(output_dir, run_dir))
        except:
            pass
    return

try:
    df = pd.read_csv(log_file)
except:
    df = pd.DataFrame(job_list,columns=['name'])
    df['SUBMITTED'] = 0
    df['id'] = np.arange(len(df))
    df.to_csv(log_file)

while any(df.SUBMITTED==0):

    run_cleanup(master_file, output_dir)

    one_job = df[df.SUBMITTED==0].iloc[0]
    cmd = ['sbatch', one_job['name']]
    status = 1
    while status!=0:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            # check for successful run and print the error
            status = proc.returncode
            out = proc.stdout
        except:
            status = np.random.choice([0,1])
            out = 'EXCEPTION'
        if status!=0:
            my_str = 'Job submission FAILED: {} {}'.format(out, cmd)
            my_str += '\n Will try again in {} mins'.format(sleep_secs/60)
            print(my_str)
            time.sleep(sleep_secs)
    new_str = 'Job submitted: {}'.format(' '.join(cmd))
    print(new_str)
    df.loc[df.id==one_job['id'], 'SUBMITTED'] = 1
    df.to_csv(log_file)
