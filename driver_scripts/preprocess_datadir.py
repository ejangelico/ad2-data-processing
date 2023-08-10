import sys
import os
sys.path.append("../Dataset/")
sys.path.append("../ParseStruck/")
sys.path.append("../ParseAD2/")
import NGMBinaryFile
import ParseAD2
from StruckPreReduction import prereduce, get_times_from_readthread
import glob
import pickle
import pandas as pd


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
    readthread = "sisreadthread.log" #assumed in the same folder as /struck/ files. 
    
    infiles = glob.glob(struckdir+"*.bin")
    infiles = sorted(infiles)

    print("Pre-processing struck data")
    #process the readthread logfile to get nanosecond timestamps
    #of when clocks were reset for file-runs. 
    readthread_stamps = get_times_from_readthread(struckdir+readthread)

    for i, f in enumerate(infiles):
        print("Loading data from {}, file {:d} of {:d}".format(f, i, len(infiles)))
        ngmb = NGMBinaryFile.NGMBinaryFile(input_filename=f, output_directory=struckdir, config_file = config)
        ngmb.GroupEventsAndWriteToPickle(save=False)
        print("Pre-reducing the data from {}".format(f))
        df, date = prereduce(ngmb, config, readthread_stamps)
        if(len(df.index) == 0):
            print("No events found in the struck prereduction of file {}".format(f))
            print("Not going to pickel it.")
            continue

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







    
