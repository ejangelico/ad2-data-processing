#this file is used to submit many parallel jobs for data reduction
import os


#make a list of datasets to submit
topdir = "/p/lustre2/nexouser/data/StanfordData/angelico/hv-test-chamber/Run8/"
datasets = ["ds01/","ds02/","ds03/", "ds04/","ds05/","ds06/","ds07/",\
        "ds08/","ds09/","ds10/","ds11/","ds12/"]

path_to_config = "/g/g15/angelico/ad2-data-processing/configs/run8_config.yaml"
jobname = "-run8"


activate_venv = 'source $HOME/my_personal_env/bin/activate'


jobcount = 0
for ds in datasets:
    path_to_dataset = topdir+ds
    cmd_options = '--export=ALL -p pbatch -t 0:30:00 -n 1 -J {} -o {}{}.out'.format(ds[:-1]+jobname, topdir, ds[:-1]+jobname)
    exe = 'python $HOME/ad2-data-processing/driver_scripts/preprocess_datadir.py {} {}'.format(path_to_dataset, path_to_config)
    cmd_full = '{} && sbatch {} --wrap=\'{}\''.format(activate_venv,cmd_options,exe)

    print(cmd_full)
    os.system(cmd_full)
    print('job {} sumbitted'.format(str(jobcount)+jobname))

    jobcount += 1
    
