import numpy as np
import pandas as pd
import datetime
import pickle
import yaml
import glob
import os
import sys


#This class mirrors the NGMBinaryFile class for Strucks, in that
#it loads the data, forms some basic information about the data,
#and pre-conditions it into a dataframe that is ready to be merged
#with the pre-reduced struck dataframe. 


class ParseAD2:
    def __init__(self, data_dir, config_file = None):

        #load configuration
        self.config = None
        self.config_file = None
        self.month = None
        self.year = None
        if config_file is not None:
            self.config_file = config_file
            self.load_config()
        else:
            print("no config file provided, using default (PROBABLY WRONG!!!)")
            self.config_file = "../configs/default_config.yaml"
            self.load_config()

        self.topdir = data_dir
        if(os.path.exists(self.topdir) == False):
            print("The directory {} does not exist! Can't process AD2 files. Stopping")
            sys.exit()
       


        #data structures for holding info about raw data
        #separated_file_lists["glitch"] = [file0, file1, file2, ...]
        self.separated_file_lists = {} #indexed by string prefix
        self.separated_timestamps = {} #datetime objects from filename
        self.file_prefixes = list(self.config.keys()) #keys in ad2_reduction are file prefixes 
        
        self.ouptut_df = pd.DataFrame()

    #load files in the data directory and parse waveforms into a dataframe that is
    #mergable and matches with Struck dataframe. 
    def prereduce_data(self):
        #check to see if file names have been loaded
        if(len(list(self.separated_file_lists.keys())) == 0):
            print("No file names were loaded yet using load_filenames, so will do that now.")
            self.load_filenames()
        
        #this is the dictionary of lists that will form an output
        #pandas dataframe that can be merged with the Struck data prior to
        #reduction. 
        #"Seconds": number of seconds since UNIX EPOCH 00:00 UTC Jan 1, 1970. 
        #when trigger occurred. "Nanoseconds": is nanoseconds after that second marker. 
        #On a 64 bit computer, we can store a number such as 86399.123456789 which is in the last second of the day.  
        #"Data" all channels are always digitized, "Data": [[v0, v1, v2, ...], [v0, v1, v2, ...]], one voltage stream for each channel. 
        #"Channels" is the software ID channel number in that event. This matches prefix with some
        # software ID, which will help distinguish the types of detectors when merging.  
        #"dT" is the sampling time in that event, which may be dynamic on an event by event basis. 
        output_dict = {"Seconds": [], "Nanoseconds": [], "Data": [], "Channels": [], "dT": []}

        #the date, 2023-5-10 17:00:41 for example, will be passed in the pickle file 
        #so that it can be stored during merging of the two digitization dataframes. 
        #However, I don't want to put Nevents x Date amount of bytes that is just a copy
        #of the date over and over again in the output_dict, so it just appears once. 
        #Do this for the first event in the file list, save the date, and then
        #the seconds after 00:00:00 that day are in output_dict for each event. 
        date = self.separated_timestamps[self.file_prefixes[0]][0] #first timestamp of first prefix. 

        #for each prefix, process the data into the output_dict, and associate a channel ID. 
        for pref in self.file_prefixes:
            pol = np.sign(int(self.config[pref]["polarity"])) #will use this so that every data stream is positive. 
            chs = self.config[pref]["channel_map"]["ad2_channel"] #indexing the raw AD2 data
            sw_chs = self.config[pref]["channel_map"]["software_channels"] #new identifiers for these channels
            #if there's a mistake in the "channel map" of the config file, end. 
            if(len(sw_chs) != len(chs)):
                print("ISSUE! Software channels and active channels don't match in config file.")
                sys.exit()

            for i, f in enumerate(self.separated_file_lists[pref]):
                timestamp = self.separated_timestamps[pref][i] #datetime object containing up to ms precision. 
                seconds_since_epoch = np.floor(timestamp.timestamp()) #floor to ignore microsecond precision. 
                nanoseconds = (timestamp.microsecond)*1e3 #nanosecond
                output_dict["Seconds"].append(seconds_since_epoch)
                output_dict["Nanoseconds"].append(nanoseconds)
                output_dict["dT"].append(self.get_sampling_period_from_file(f)) #sampling period in seconds

                #load the csv data 
                csv_data = np.genfromtxt(self.topdir + f, delimiter=',',  skip_header=20, dtype=float)
                
                #now correct polarity, and only for channels
                #listed as active channels in the config file. 
                temp_data = []
                for ch in chs:
                    v = np.array(csv_data[:,ch+1]) #nth column, 0th is time. 
                    v_mv = v*1000 #these files write in units of volts
                    v_mv = v_mv*pol 
                    temp_data.append(v_mv)

                output_dict["Data"].append(temp_data)
                output_dict["Channels"].append(sw_chs)

        self.output_df = pd.DataFrame(output_dict)
        return self.output_df, date
                







    #just loads filenames and separates by prefix, with option to limit the number of events. 
    #for example, file_prefixes = ["pmt", "anode"]
    #Date of dataset is used because the file timestamps dont contain 
    #the month or year. If absolute times matter, include the date.
    #event_limit = [min event number, max event number] (in order please, can use -1)
    def load_filenames(self, event_limit=None):

        print("Looking through files in directory " + self.topdir + " and grouping based on prefix")
        #full list of .csv files
        file_list = glob.glob(self.topdir+"*.csv")
        if(len(file_list) == 0):
            print("No data files found in directory: " + self.topdir)
            return
       
        #turn filenames into only filenames, not with whole path
        file_list = [_.split('/')[-1] for _ in file_list]

        #add prefixes to separated file lists self attribute
        for pref in self.file_prefixes:
            self.separated_file_lists[pref] = []
            self.separated_timestamps[pref] = []
            print("Selecting files with prefix " + pref)
            #selects filenames by prefix. so separate_file_lists['pmt'] = ['pmt14.53.24.449', 'pmt10.34....', ...]
            self.separated_file_lists[pref] = list(filter(lambda x: x[:len(pref)] == pref, file_list))  

            #sort the list by timestamp
            self.separated_timestamps[pref] = [self.get_timestamp_from_filename(_) for _ in self.separated_file_lists[pref]]
            if(len(self.separated_timestamps[pref]) == 0):
                print("Found no files with the prefix: " + pref)
                continue
            #this line sorts both lists simultaneously 
            #based on the datetime values in the date_times list
            self.separated_timestamps[pref], self.separated_file_lists[pref] = \
            (list(t) for t in zip(*sorted(zip(self.separated_timestamps[pref], self.separated_file_lists[pref]))))

            print("Done: found " + str(len(self.separated_file_lists[pref])) + "\n\n")

        if(event_limit is not None):
            print("Limiting the number of events to the chronological range:", end=' ')
            print(event_limit)
            for pref in self.separated_timestamps:
                if(event_limit[0] < 0):
                    event_limit[0] = 0 
                if(event_limit[1] >= len(self.separated_timestamps[pref])):
                    event_limit[1] = len(self.separated_timestamps[pref]) - 1
                if(event_limit[0] >= len(self.separated_timestamps[pref])):
                    self.separated_file_lists[pref] = []
                    self.separated_timestamps[pref] = []
                    continue

                self.separated_timestamps[pref] = self.separated_timestamps[pref][event_limit[0]:event_limit[1]]
                self.separated_file_lists[pref] = self.separated_file_lists[pref][event_limit[0]:event_limit[1]]





    def load_config(self):
        if(os.path.isfile(self.config_file)):
            #safe read this yaml file
            with open(self.config_file, 'r') as stream:
                try:
                    config = yaml.safe_load(stream)
                except:
                    print("Had an issue reading the config file, make sure it is a .yaml or .yml file")
                    return None
            self.month = config["month"]
            self.year = config["year"]
            self.config = config["ad2_reduction"]
            
        else:
            print("Couldnt load configuration file... {} doesn't exist".format(self.config_file))

    #------parsing utilities-----#
    def get_str_timestamp_from_filename(self, filename):
        #filename may have leading terms with "/". remove them
        fn = filename.split('/')[-1]
        #independent of prefix length, the length of timestamp
        # from the end of filename is always fixed. 
        #now we have, for example, pmt18.11.14.00.852.csv dd,hh,mm,ss,mmm
        timedotted = fn[-19:-4]
        return timedotted 

    #returns datetime object instead of string
    def get_timestamp_from_filename(self, filename):
        timedotted = self.get_str_timestamp_from_filename(filename)
        #cant use the "%d" flag twice, remove from dataset_date
        monthyear = self.month + "-" + self.year
        date_format = "%m-%Y %d.%H.%M.%S.%f"
        try:
            date = datetime.datetime.strptime(monthyear + " " + timedotted, date_format)
        except:
            #in some cases, the filename may not have a timestamp. in this case, return current time
            date = datetime.datetime.now()
        return date

    #takes a datetime as input and creates a filename
    def get_filename_from_timestamp(self, stamp, file_prefix):
        date_format = "%d.%H.%M.%S.%f"
        fn = datetime.datetime.strftime(stamp, date_format)
        fn = file_prefix+fn[:-3]+".csv" #:-3 is because it returns %f in milliseconds
        return fn

    #takes a timedelta and returns milliseconds since epoch
    def get_milliseconds_from_timedelta(self,td):
        return td.total_seconds()*1000

    #gets the sampling period from file header in seconds
    def get_sampling_period_from_file(self, filename):
        #parse header for the timestep
        f = open(self.topdir+filename, 'r', errors='ignore')
        ls = f.readlines()
        raw_sample_rate = ls[4]
        raw_sample_rate = raw_sample_rate.split(' ')[-1]
        raw_sample_rate = float(raw_sample_rate.split('H')[0])
        return (1.0/raw_sample_rate) #seconds
