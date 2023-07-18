import sys
import os
sys.path.append("../Dataset/")
sys.path.append("../ParseAD2/")
import ParseAD2
import glob
import pickle
import pandas as pd




if __name__ == "__main__":

    if(len(sys.argv) != 3):
        print("python preprocess_ad2_<>.py <path to data> <path to config file>")
        sys.exit()


    topdir = sys.argv[1]
    config = sys.argv[2]
    
    print("Loading data from {}".format(topdir))
    ad2 = ParseAD2.ParseAD2(topdir, config)
    ad2.load_filenames()
    print("Pre-reducing the data from {}".format(topdir))
    df, date = ad2.prereduce_data()

    #save at this stage the pre-reduced struck data
    pickle.dump([df, date], open(topdir+"prereduced_ad2.p", "wb"))


    
