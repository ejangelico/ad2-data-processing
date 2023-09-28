import numpy as np 
import os 
import pandas as pd
import NGMBinaryFile
import datetime
import sys
import yaml

#this class takes in the hdf5 pandas dataframes from NGMBinaryFile output processing and 
#pre processing the events for integrating with the AD2 data. The operations include
#1. reconstructing timestamps in UTC time and nanoseconds relative to the clock full timestamp counter
#2. formats the output dataframe to be consistent with the Dataset.py object
#3. Deletes empty events, i.e. bugginess if both channels have no amplitude relative to baseline. 

#takes an NGMBinaryFile object, checks that the dataframe has been formed. 

def prereduce(ngmb, config_file, readthread_stamps):

    #consistency checks before starting. 
    #has the data at least been grouped into a dataframe from binary?
    if(len(ngmb.output_df.index) == 0):
        print("**Can't pre-reduce the data from {} because it hasn't yet been loaded with NGMBinaryFile::GroupEventsAndWriteToPickle".format(ngmb.input_filename))
        print("Doing so here anyway, but make sure you approve")
        ngmb.GroupEventsAndWriteToPickle(save=False)
        print("Done, moving on to prereduction")

    #check again, if it still is, then we have a bigger issue and need to return nothing
    if(len(ngmb.output_df.index) == 0):
        print("Still no data in the NGMB object, so this file maybe didn't have any events. Continuing on without it. ")
        return pd.DataFrame(), None #the prereduce script will catch an empty dataframe. 


    config = load_config(config_file)
    if(config == None):
        print("Had an error loading the config file, not going to prereduce")
        sys.exit()
    
    #this is the dictionary of lists that will form an output
    #pandas dataframe that can be merged with the AD2 data prior to
    #reduction. The columns will match that class (Dataset.py), but are
    #explained here as well for clarity. 
    #"Seconds": number of seconds since UNIX EPOCH 00:00 UTC Jan 1, 1970. 
    #when trigger occurred. "Nanoseconds": is nanoseconds after that second marker. 
    #On a 64 bit computer, we can store a number such as 86399.123456789 which is in the last second of the day.  
    #"Data" all channels are always digitized, "Data": [[v0, v1, v2, ...], [v0, v1, v2, ...]], one voltage stream for each channel. 
    #In merging stage, two additional columns will appear. Channels which will be [0,1,2,3] where the channel numbers are unique across DAQ systems,
    #and Type: ["pmts", "pmts", "glitch", "anode"] associated with the channel numbers and prefixes. 
    pref = config["prefix"]
    output_dict = {"Seconds":[], "Nanoseconds": [], "Data": []}

    #get the date of the filename, with NO offset from GMT to PST timezones. 
    date = get_date_from_filename(ngmb, 0)
    #find the closest date in the readthreadlog timestamps list. 
    closest_index = min(range(len(readthread_stamps)), key=lambda i: abs(readthread_stamps[i]['Seconds'] - date.timestamp()))
    #catch a GROSS error where the readthreadlog is super stale, greater than 100 seconds
    tcatch = 100 #seconds
    if(np.abs(readthread_stamps[closest_index]["Seconds"] - date.timestamp()) > tcatch):
        print("ERROR!: sisreadthread.log is over 100 seconds stale, and you may not have kept it organized in data handling!")

    file_seconds = readthread_stamps[closest_index]["Seconds"]
    file_seconds = file_seconds + config["timestamp_offset"] #correct for GMT to PST timezone
    file_nsec = readthread_stamps[closest_index]["Nanoseconds"]

    indf = ngmb.output_df

    #we are now going to loop through the dataframe and do the following:
    #1. convert to mV
    #2. Continue past the event if either channel has no voltage sample above config["no_pulse_threshold"]
    print("Prereducing data...")
    pol = np.sign(int(config["polarity"])) #will use this so that every data stream is positive. 
    mv = float(config["mv_per_adc"])
    dT = 1.0/float(config["clock"]) #seconds
    for i, row in indf.iterrows():
        vs = []
        for ch in range(len(row["Data"])):
            v = np.array(row["Data"][ch])
            v_mv = v*mv
            v_mv = v_mv*pol 
            vs.append(v_mv)

            
        output_dict["Data"].append(vs)

        #the nanosecond timestamp is the same for both channels in this code, so take 0th chan.  

        #this try except catches a strange case where there are no events?
        try:
            clock_sec_after_file_date = row["Timestamp"][0]*dT #seconds, with nanosecond precision. 
        except:
            continue
        clock_nsec_after_file_date = (clock_sec_after_file_date - np.floor(clock_sec_after_file_date))*1e9 #just part after decimal. 
        clock_sec_after_file_date = np.floor(clock_sec_after_file_date) #just isolate 1 second precision part. 

        total_nsec = clock_nsec_after_file_date + file_nsec

        #if the nsec after file date from clock cycles and nsec of file timestamp adds to greater 
        #than 1, then the seconds counter needs to increment and our nsec timestmap component is changed. 
        #essentially this is modulus, pulling out only the decimal part of this long float. 
        if(total_nsec >= 1e9):
            clock_sec_after_file_date += 1
            total_nsec = total_nsec - 1e9

        output_dict["Seconds"].append(file_seconds + clock_sec_after_file_date)
        output_dict["Nanoseconds"].append(total_nsec)
    

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

#to get nanosecond (and reliable second) precision on the clock
#reset time of the Strucks, we need to look at the sisreadthread.log file
#which contains the actual GMT timestamp of when the clocks were set to 0, 
#with nanosecond precision. 
#
#We then will compare the timestamps to find the closest filename to that
#timestamp which we then can associate filenames with nanosecond level precision.
#
#This function returns a list of timestamps since epoch in seconds
#as well as the nanoseconds associated with that time. DOES NOT APPLY
#the GMT to PST timezone conversion offset. 
def get_times_from_readthread(readthread):
    f = open(readthread, 'r')
    lines = f.readlines()
    output = []
    for l in lines:
        if("After" in l[:10]):
            try:
                nsec = float(l.split('+')[-1].split(' ')[0])
            except:
                #wierd thing I've only seen once in one line on one dataset, there is a space
                #between the + and the nanosecond int. 
                nsec = float(l.split('+')[-1].split(' ')[1])
            date_byte = l.split(' ')[3:8]
            date_str = ' '.join(date_byte)
            date_format = "%a, %d %b %Y %H:%M:%S"
            ts_datetime = datetime.datetime.strptime(date_str, date_format)
            output.append({"Seconds":ts_datetime.timestamp(), "Nanoseconds":nsec})
    return output



