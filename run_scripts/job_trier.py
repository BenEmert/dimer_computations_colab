import os, sys
import time
import numpy as np
import subprocess

log_file = 'my.log'

sleep_secs = 30*60 # length of time (secs) to wait before trying to submit more jobs. Using 30min.

job_list = ['inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset313000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset312000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset311000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset310000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset309000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset308000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset307000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset306000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset305000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset304000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset303000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset302000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset301000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset300000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset299000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset298000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset297000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset296000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset295000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset294000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset293000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset292000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset291000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset290000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset289000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset288000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset287000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset286000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset285000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset284000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset283000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset282000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset281000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset280000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset279000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset278000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset277000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset276000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset275000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset274000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset273000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset272000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset271000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset270000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset269000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset268000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset267000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset266000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset265000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset264000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset263000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset262000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset261000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset260000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset259000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset258000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset257000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset256000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset255000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset254000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset253000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset252000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset251000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset250000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset249000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset248000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset247000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset246000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset245000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset244000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset243000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset242000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset241000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset240000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset239000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset238000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset237000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset236000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset235000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset234000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset233000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset232000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset231000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset230000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset229000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset228000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset227000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset226000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset225000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset224000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset223000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset222000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset221000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset220000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset219000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset218000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset217000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset216000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset215000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset214000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset213000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset212000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset211000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset210000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset209000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset208000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset207000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset206000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset205000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset204000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset203000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset202000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset201000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset200000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset199000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset198000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset197000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset196000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset195000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset194000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset193000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset192000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset191000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset190000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset189000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset188000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset187000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset186000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset185000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset184000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset183000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset182000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset181000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset180000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset179000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset178000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset177000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset176000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset175000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset174000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset173000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset172000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset171000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset170000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset169000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset168000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset167000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset166000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset165000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset164000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset163000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset162000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset161000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset160000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset159000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset158000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset157000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset156000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset155000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset154000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset153000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset152000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset151000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset150000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset149000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset148000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset147000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset146000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset145000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset144000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset143000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset142000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset141000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset140000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset139000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset138000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset137000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset136000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset135000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset134000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset133000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset132000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset131000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset130000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset129000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset128000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset127000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset126000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset125000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset124000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset123000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset122000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset121000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset120000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset119000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset118000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset117000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset116000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset115000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset114000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset113000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset112000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset111000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset110000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset109000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset108000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset107000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset106000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset105000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset104000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset103000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset102000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset101000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset100000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset99000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset98000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset97000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset96000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset95000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset94000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset93000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset92000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset91000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset90000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset89000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset88000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset87000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset86000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset85000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset84000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset83000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset82000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset81000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset80000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset79000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset78000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset77000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset76000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset75000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset74000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset73000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset72000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset71000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset70000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset69000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset68000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset67000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset66000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset65000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset64000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset63000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset62000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset61000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset60000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset59000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset58000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset57000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset56000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset55000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset54000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset53000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset52000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset51000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset50000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset49000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset48000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset47000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset46000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset45000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset44000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset43000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset42000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset41000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset40000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset39000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset38000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset37000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset36000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset35000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset34000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset33000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset32000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset31000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset30000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset29000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset28000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset27000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset26000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset25000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset24000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset23000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset22000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset21000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset20000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset19000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset18000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset17000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset16000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset15000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset14000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset13000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset12000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset11000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset10000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset9000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset8000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset7000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset6000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset5000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset4000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset3000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset2000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset1000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m10_offset0.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset80000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset79000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset78000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset77000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset76000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset75000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset74000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset73000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset72000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset71000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset70000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset69000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset68000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset67000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset66000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset65000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset64000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset63000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset62000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset61000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset60000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset59000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset58000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset57000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset56000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset55000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset54000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset53000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset52000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset51000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset50000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset49000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset48000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset47000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset46000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset45000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset44000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset43000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset42000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset41000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset40000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset39000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset38000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset37000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset36000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset35000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset34000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset33000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset32000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset31000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset30000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset29000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset28000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset27000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset26000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset25000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset24000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset23000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset22000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset21000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset20000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset19000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset18000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset17000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset16000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset15000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset14000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset13000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset12000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset11000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset10000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset9000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset8000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset7000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset6000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset5000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset4000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset3000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset2000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset1000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m5_offset0.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset46000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset45000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset44000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset43000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset42000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset41000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset40000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset39000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset38000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset37000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset36000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset35000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset34000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset33000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset32000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset31000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset30000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset29000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset28000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset27000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset26000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset25000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset24000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset23000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset22000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset21000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset20000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset19000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset18000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset17000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset16000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset15000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset14000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset13000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset12000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset11000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset10000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset9000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset8000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset7000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset6000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset5000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset4000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset3000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset2000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset1000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m4_offset0.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m3_offset23000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m3_offset22000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m3_offset21000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m3_offset20000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m3_offset19000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m3_offset18000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m3_offset17000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m3_offset16000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m3_offset15000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m3_offset14000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m3_offset13000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m3_offset12000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m3_offset11000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m3_offset10000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m3_offset9000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m3_offset8000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m3_offset7000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m3_offset6000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m3_offset5000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m3_offset4000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m3_offset3000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m3_offset2000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m3_offset1000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen/m3_offset0.job']

with open(log_file, 'w') as fh:
    for job_file in job_list:
            cmd = ['sbatch', job_file]
            status = 1
            while status!=0:
                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True)
                    # check for successful run and print the error
                    status = proc.returncode
                    out = proc.stdout
                except:
                    status = 1
                    out = 'EXCEPTION'
                if status!=0:
                    my_str = 'Job submission FAILED: {} {}'.format(out, cmd)
                    my_str += '\n Will try again in {} mins'.format(sleep_secs/60)
                    print(my_str)
                    fh.writelines(my_str)
                    time.sleep(sleep_secs)
            new_str = 'Job submitted: {}'.format(' '.join(cmd))
            print(new_str)
            fh.writelines(my_str)
