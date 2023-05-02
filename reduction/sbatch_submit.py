#this file is used to submit many parallel jobs for data reduction
import sys
import os
import glob
import numpy as np 


path_to_dataset = "/p/lustre1/angelico/hv-test-chamber/Run4/cosmics-overnight/"
date_of_dataset = "4-19-23"
path_to_config = path_to_dataset+"run4_config.yaml"
jobname = "cos-ov"
files_per_chunk = 5000 #must be integer
filelist = glob.glob(path_to_dataset+"*.csv")
n_files = len(filelist)

#estimate time needed
t_min = files_per_chunk*5/1000 #minutes
t_min_reduced = t_min % 60
t_hours = t_min/60



filecount = 0 
jobcount = 0
while True:
    jobcount += 1
    if(filecount > n_files):
        break
    chunk = [filecount, filecount + files_per_chunk]
    activate_venv = 'source $HOME/my_personal_env/bin/activate'
    cmd_options = '--export=ALL -p pbatch -t {:02d}:{:02d}:00 -n 1 -J {} -o {}{}.out'.format(int(t_hours), int(t_min_reduced), str(jobcount)+jobname, path_to_dataset, str(jobcount)+jobname)
    exe = 'python $HOME/ad2-data-processing/reduction/reduce_data_LLNL_4-28-2023.py {} {} {:d} {:d} {}'.format(path_to_dataset, path_to_config, chunk[0], chunk[1], date_of_dataset)
    cmd_full = '{} && sbatch {} --wrap=\'{}\''.format(activate_venv,cmd_options,exe)

    print(cmd_full)
    #os.system(cmd_full)
    print('job {} sumbitted'.format(str(jobcount)+jobname))

    filecount += files_per_chunk
    
