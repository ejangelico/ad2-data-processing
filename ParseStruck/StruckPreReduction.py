import numpy as np 
import os 
import pandas as pd
import NGMBinaryFile
import datetime
import sys
import yaml

#this class takes in the hdf5 pandas dataframes from NGMBinaryFile output processing and 
#pre processing the events for integrating with the AD2 data. The operations include
#1. reconstructing timestamps in UTC time and microseconds relative to the clock full timestamp counter
#2. formats the output dataframe to be consistent with the Dataset.py object
#3. Deletes empty events, i.e. bugginess if both channels have no amplitude relative to baseline. 

#takes an NGMBinaryFile object, checks that the dataframe has been formed. 
def prereduce(ngmb, config_file):

    #consistency checks before starting. 
    #has the data at least been grouped into a dataframe from binary?
    if(len(ngmb.output_df.index) == 0):
        print("**Can't pre-reduce the data from {} because it hasn't yet been loaded with NGMBinaryFile::GroupEventsAndWriteToPickle".format(ngmb.input_filename))
        print("Doing so here anyway, but make sure you approve")
        ngmb.GroupEventsAndWriteToPickle(save=False)
        print("Done, moving on to prereduction")


    config = load_config(config_file)
    if(config == None):
        print("Had an error loading the config file, not going to prereduce")
        sys.exit()
    
    #this is the dictionary of lists that will form an output
    #pandas dataframe that can be merged with the AD2 data prior to
    #reduction. The columns will match that class (Dataset.py), but are
    #explained here as well for clarity. 
    #"Seconds": number of seconds after the date in the filename when trigger occurred, with nanosecond precision (9 decimals)
    #On a 64 bit computer, we can store a number such as 86399.123456789 which is in the last second of the day.  
    #"Data" all channels are always digitized, "Data": [[v0, v1, v2, ...], [v0, v1, v2, ...]], one voltage stream for each channel. 
    #In merging stage, two additional columns will appear. Channels which will be [0,1,2,3] where the channel numbers are unique across DAQ systems,
    #and Type: ["pmts", "pmts", "glitch", "anode"] associated with the channel numbers and prefixes. 
    pref = config["prefix"]
    output_dict = {"Seconds":[], "Data": []}

    #the date, 2023-5-10 17:00:41 for example, will be passed in the pickle file 
    #so that it can be stored during merging of the two digitization dataframes. 
    #However, I don't want to put Nevents x Date amount of bytes that is just a copy
    #of the date over and over again in the output_dict, so it just appears once. 
    date = get_date_from_filename(ngmb, config["timestamp_offset"])

    indf = ngmb.output_df

    #we are now going to loop through the dataframe and do the following:
    #1. Baseline subtract the data and convert to mV
    #2. Continue past the event if either channel has no voltage sample above config["no_pulse_threshold"]
    print("Prereducing data...")
    bw = config["baseline_window"] #format is [sample_i, sample_f]
    pol = np.sign(int(config["polarity"])) #will use this so that every data stream is positive. 
    thr = float(config["no_pulse_threshold"])
    mv = float(config["mv_per_adc"])
    dT = 1.0/float(config["clock"]) #seconds
    for i, row in indf.iterrows():
        vs = []
        above_thresh = []
        for ch in range(len(row["Data"])):
            v = np.array(row["Data"][ch])
            v_mv = v*mv
            base = np.mean(v_mv[int(min(bw)):int(max(bw))])
            v_mv = v_mv - base
            v_mv = v_mv*pol 
            #check if it passes the threshold
            if(max(v_mv) < thr):
                above_thresh.append(0)
            else:
                above_thresh.append(1)

            vs.append(v_mv)
        #skip past events that don't pass thresh
        if(0 in above_thresh):
            continue 
            
        output_dict["Data"].append(vs)

        #the nanosecond timestamp is the same for both channels in this code. 
        output_dict["Seconds"].append(row["Timestamp"][0]*dT)
    

    output_df = pd.DataFrame(output_dict)

    print("Pre reduction of {} removed {:.2f}% of events due to the no_pulse_threshold. ".format(ngmb.input_filename, (1 - float(len(output_df.index))/len(indf.index))))
    return output_df, date




def load_config(config_file):
    if(os.path.isfile(config_file)):
        #safe read this yaml file

        with open(config_file, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except:
                print("Had an issue reading the config file, make sure it is a .yaml or .yml file")
                return None
            
        return config["struck_reduction"]
        
    else:
        print("Couldnt load configuration file... {} doesn't exist".format(config_file))
        return None
    

#if the struck takedata script or C++ code changes as far as how the
#date is formatted, then this needs to change. It's pretty hard coded. 
def get_date_from_filename(ngmb, offset):
    fn = ngmb.input_filename
    f_spl = fn.split('/')[-1].split('_')[1]
    year4 = f_spl[:4]
    month = f_spl[4:6]
    day = f_spl[6:8]

    #unfortunately, in Run 5 and Run 6, the sisreadthread wasn't kept organized
    #enough to extract to better than 1s precision the time of the acquisition start.
    #it is only stored in the filename which has 1s precision, whereas we have nanosecond
    #precision written in the sisreadthread files. FIX THIS next time. For now, we 
    #synchronize DAQs to 1s precision, and the struck relative sync is nanosecond precision 
    #within the same files. 
    hour = f_spl[8:10]
    minute = f_spl[10:12]
    sec = f_spl[12:14]

    d = datetime.datetime.strptime(year4+month+day+hour+minute+sec, "%Y%m%d%H%M%S")

    #then the Struck code for some reason has a 7 hours offset. This is set in the config file. 
    dt = datetime.timedelta(seconds=offset)
    d = d + dt
    return d



