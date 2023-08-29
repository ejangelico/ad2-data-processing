import sys
import os
sys.path.append("../Dataset/")
import Dataset
import glob
import pickle
import pandas as pd
import time 


if __name__ == "__main__":

    if(len(sys.argv) != 3):
        print("python reduce_<>.py <path to data> <path to config file>")
        sys.exit()


    topdir = sys.argv[1]
    struckdir = sys.argv[1]+"/struck/"
    config = sys.argv[2]

    #create dataset object from these filepaths
    print("Starting reduction")
    t0 = time.time()
    ds = Dataset.Dataset(topdir, config)
    ds.reduce_data()
    pickle.dump([ds.reduced_df], open(topdir+"reduced.p", "wb"))
    t1 = time.time()
    print("Reduction took {:.2f} sec".format(t1 - t0))

