import sys
import os
sys.path.append("../Dataset/")
sys.path.append("../ParseStruck/")
sys.path.append("../ParseAD2/")
import NGMBinaryFile
import ParseAD2
from StruckPreReduction import prereduce
import glob
import pickle
import pandas as pd
import yaml


def load_config(infile):
    #safe read this yaml file
    with open(infile, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except:
            print("Had an issue reading the config file, make sure it is a .yaml or .yml file")
            return None
        
    return config

#This submission script is meant to preprocess all 
#data, both struck and AD2 data, in a single data directory. 
#This is the step before reducing the data, which analyzes both DAQ systems together and
#performs more reduction related tasks. 

if __name__ == "__main__":

    if(len(sys.argv) != 3):
        print("python preprocess_datadir_<>.py <path to data> <path to config file>")
        sys.exit()


    topdir = sys.argv[1]
    struckdir = sys.argv[1]+"/struck/"
    config = sys.argv[2]
    
    infiles = glob.glob(struckdir+"*.bin")
    infiles = sorted(infiles)

    print("Pre-processing struck data")
    for i, f in enumerate(infiles):
        print("Loading data from {}, file {:d} of {:d}".format(f, i, len(infiles)))
        ngmb = NGMBinaryFile.NGMBinaryFile(input_filename=f, output_directory=struckdir, config_file = config)
        ngmb.GroupEventsAndWriteToPickle(save=False)
        print("Pre-reducing the data from {}".format(f))
        df, date = prereduce(ngmb, config)

        #save at this stage the pre-reduced struck data
        pickle.dump([df, date], open(struckdir+"prereduced_"+str(i)+".p", "wb"))

    print("Pre-processing AD2 data")

    print("Loading data from {}".format(topdir))
    ad2 = ParseAD2.ParseAD2(topdir, config)
    ad2.load_filenames()
    print("Pre-reducing the data from {}".format(topdir))
    df, date = ad2.prereduce_data()

    #save at this stage the pre-reduced ad2 data
    pickle.dump([df, date], open(topdir+"prereduced_ad2.p", "wb"))







    
