#This script takes all "reduced.p" files in DS folders of a Run and combines them
#for a run-level global dataset. 

import numpy as np 
import pandas as pd 
import os
import glob
import pickle
import sys

output_file = "combined_reduced.p"
use_all = False #if you want to find all, recursively, reduced.p files. 
#otherwise, put the ds names here

if(len(sys.argv) != 2):
    print("Usage: python combine_reduced.py <# correspond to run number>")
    sys.exit()

runnum = int(sys.argv[1])

run9_ds = ["ds01/","ds02/","ds03/", "ds04/","ds05/","ds06/","ds07/",\
        "ds08/","ds09/","ds10/","ds11/","ds12/","ds13/","ds14/"]

run8_ds = ["ds01/","ds02/","ds03/", "ds04/","ds05/","ds06/","ds07/",\
        "ds08/","ds09/","ds10/","ds11/", "ds12/"]

run7_ds = ["ds01/","ds02/","ds03/", "ds04/","ds05/","ds06/","ds07/",\
        "ds08/","ds09/","ds10/","ds11/"]

run6_ds = ["ds01/", "ds02/","ds03/","ds04/","ds05/"]

run5_ds = ["ds03/", "ds04/", "ds05/", "ds06/", \
            "ds07/","ds08/","ds09/","ds10/","ds11/","ds12/","ds13/","ds15/","ds16/",\
            "ds17/","ds18/","ds19/","ds20/","ds21/","ds22/","ds23/","ds24/","ds25/"]

run10_ds = ["ds01/","ds02/","ds03/","ds04/","ds05/","ds06/","ds07/","ds08/","ds09/","ds10/","ds11/","ds12/","ds13/","ds14/"]


runs = {5:run5_ds, 6:run6_ds, 7:run7_ds, 8:run8_ds, 9:run9_ds, 10:run10_ds}

#topdir = "../../data/Run8/"
topdir = "/p/lustre2/nexouser/data/StanfordData/angelico/hv-test-chamber/Run{:d}/".format(runnum)
datasets = runs[runnum]

if(use_all):
    red_files = glob.glob(topdir+"**/reduced.p", recursive=True)
else:
    red_files = []
    for _ in datasets:
        temp_files = glob.glob(topdir+_+"reduced.p")
        if(len(temp_files) > 0):
            red_files.append(temp_files[0])

dfs = [] #all loaded into ram, then concatenated at the end
for f in red_files:
    print("On {}".format(f))
    dfs.append(pickle.load(open(f, "rb"))[0])

#concat all
output_df = pd.concat(dfs, ignore_index=True)
pickle.dump([output_df], open(topdir+output_file, "wb"))




