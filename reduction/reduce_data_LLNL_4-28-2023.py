import sys
import os
sys.path.append("../analysis/")
import analysis.Dataset as Dataset
import pickle






if __name__ == "__main__":

    if(len(sys.argv) != 6):
        print("python reduce_data_<>.py <path to data> <path to config> <first file> <last file> <date of dataset>")
        print("where the first file and last file are the chunk of files in chronological order that are processed in this job")
        print("date of dataset is like 4-19-23 for april 19th, 2023")
        sys.exit()

    prefixes = ["pmts", "glitch"]

    topdir = sys.argv[1]
    config = sys.argv[2]
    event_limit = [int(sys.argv[3]), int(sys.argv[4])]
    date_of_dataset = sys.argv[5]
    d = Dataset.Dataset(topdir)
    d.load_raw(prefixes, date_of_dataset, event_limit=event_limit)
    d.create_wave_df()
    #because of the event limiting argument,
    #there are times when there may be no events in the dataframe. 
    #then don't save any output files. 
    if(len(d.wave_df.index) == 0):
        sys.exit()
    
    #otherwise, pickle the unreduced dataset
    pickle_filetag = str(event_limit[0]) + "-wave_df.p"
    pickle.dump([d.wave_df], open(pickle_filetag, "wb"))

    #reduce the data
    d.create_reduced_df(config_file=config)

    pickle_filetag = str(event_limit[0]) + "-reduced_df.p"
    pickle.dump([d.reduced_df], open(pickle_filetag, "wb"))

    pickle_filetag = str(event_limit[0]) + "-dataset.p"
    pickle.dump([d], open(pickle_filetag, "wb"))




    