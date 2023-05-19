import sys
import os
sys.path.append("../Dataset/")
sys.path.append("../ParseStruck/")
import NGMBinaryFile
from StruckPreReduction import prereduce
import glob
import pickle
import pandas as pd




if __name__ == "__main__":

    if(len(sys.argv) != 3):
        print("python preprocess_struck_<>.py <path to data> <path to config file>")
        sys.exit()


    topdir = sys.argv[1]
    config = sys.argv[2]
    
    infiles = glob.glob(topdir+"*.bin")
    infiles = sorted(infiles)

    for i, f in enumerate(infiles):
        print("Loading data from {}, file {:d} of {:d}".format(f, i, len(infiles)))
        ngmb = NGMBinaryFile.NGMBinaryFile(input_filename=f, output_directory=topdir, config_file = config)
        ngmb.GroupEventsAndWriteToPickle(save=True)
        print("Pre-reducing the data from {}".format(f))
        df, date = prereduce(ngmb)

        #save at this stage the pre-reduced struck data
        pickle.dump([df, date], open(topdir+"prereduced_"+str(i)+".p", "wb"))
        break


    
