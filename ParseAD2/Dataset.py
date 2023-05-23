import os
import numpy as np
import time
import matplotlib.pyplot as plt 
import random
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from tqdm.notebook import tqdm, trange
from datetime import datetime
import pickle
import yaml
import glob

class Dataset:
    #top_data_directory is the directory containing all of the
    #CSV files for a particular dataset. Within this directory are
    #also folders containing reduced data, and pandas hdf5 files
    #of waveforms. The structure is the same for each dataset, so
    #whether one is going to be using raw, reduced, or waveform files,
    #they are all referencing the top_data_directory as the point of reference. 
    #top data directory, and all others, need "/" at the end of string. 
    def __init__(self, top_data_directory):

        #check if directory exists
        if(os.path.isdir(top_data_directory) == 0):
            print("Directory does not exist: " + top_data_directory)
            print("Please try again")

        self.topdir = top_data_directory

        #both of these dataframes indexible by event number
        self.reduced_df = pd.DataFrame() #populated with a pd.DataFrame, tabulating reduced data info
        self.wave_df = pd.DataFrame() #populated with a pd.DataFrame that contains waveforms

        #data structures for holding info about raw data
        #separated_file_lists["pmt"] = [file0, file1, file2, ...]
        self.separated_file_lists = {} #indexed by string prefix of files: "pmt" or "anode" for example
        self.separated_timestamps = {} #datetime objects
        self.file_prefixes = []
        #glassman events are timestamps and HV information from the glassman control AD2.
        #this allows you to determine what the high voltage info was at a time of an event. 
        self.g_events_file = None
        self.g_events = {} #["time"] = datetime, ["v_cont"] = kV applied by controller, ["v_meas"] = kV measured by analog monitor

        #data structures for time paired events
        #[[pmtfile, anodefile], [pmtfile, anodefile], ...] 
        #where the order of files follows the order of file_prefixes
        self.time_paired_files = [] 
        self.date_of_dataset = None

        #saving the configuration used for data reduction
        self.config_dict = None

        #save average waveforms, if computed
        self.avg_wave = None


    def clear(self):
        #both of these dataframes indexible by event number
        # populated with a pd.DataFrame, tabulating reduced data info
        self.reduced_df = pd.DataFrame()
        self.wave_df = pd.DataFrame() #populated with a pd.DataFrame that contains waveforms

        #data structures for holding info about raw data
        #separated_file_lists["pmt"] = [file0, file1, file2, ...]
        self.separated_file_lists = {} #indexed by string prefix of files: "pmt" or "anode" for example
        self.separated_timestamps = {} #datetime objects
        self.file_prefixes = []

        #data structures for time paired events
        #[[pmtfile, anodefile], [pmtfile, anodefile], ...] 
        #where the order of files follows the order of file_prefixes
        self.time_paired_files = []
        self.date_of_dataset = None
    
    def test(self):
        pass

    #this function will add two dataset objects together
    #preserving as much information as possible. One thing this
    #does not quite preserve in the present version is the date of dataset,
    #where the only relevant piece of info there is the month (as filenames have the day). 
    #This should mainly be used when re-chaining datasets that were processed independently
    #for LLNL batch reduction. I.e. it is assumed that the analysis config and most things are
    #the same. 
    def __add__(self, other):
        newd = Dataset(self.topdir)
        newd.reduced_df = pd.concat([self.reduced_df, other.reduced_df], ignore_index=True)
        newd.wave_df = pd.concat([self.wave_df, other.wave_df], ignore_index=True)

        newd.file_prefixes = self.file_prefixes 
        for pref in self.file_prefixes:
            if(pref in other.file_prefixes):
                newd.separated_file_lists[pref] = np.concatenate([self.separated_file_lists[pref], other.separated_file_lists[pref]])
                newd.separated_timestamps[pref] = np.concatenate([self.separated_timestamps[pref], other.separated_timestamps[pref]])
            else:
                print("Cannot add two datasets together because they don't have the same file prefixes!")
                return None
            
        #g_events
        for key in self.g_events:
            newd.g_events[key] = np.concatenate([self.g_events[key], other.g_events[key]])
        
        #time pairing
        newd.time_paired_files = np.concatenate([self.time_paired_files, other.time_paired_files])
        
        #other elements
        newd.config_dict = self.config_dict
        newd.date_of_dataset = self.date_of_dataset

        #I don't think that time sorting any of these lists is relevant at this stage,
        #so if you have an algorithm that depends on it, sort within that algorithm. 
        return newd

    #----------Loading and saving functions----------------#

    #does an "ls" of the raw data directory
    #and separates files into lists based on their prefix.
    #for example, file_prefixes = ["pmt", "anode"]
    #Date of dataset is used because the file timestamps dont contain 
    #the month or year. If absolute times matter, include the date.
    #event_limit = [min event number, max event number] (in order please, can use -1)
    def load_raw(self, file_prefixes, date_of_dataset="01-01-21", event_limit=None):

        #clear existing data
        self.clear()

        #book keeping
        self.date_of_dataset = date_of_dataset

        print("Looking through files in directory " + self.topdir + " and grouping based on prefix")
        #full list of .csv files
        file_list = glob.glob(self.topdir+"*.csv")
        if(len(file_list) == 0):
            print("No data files found in directory: " + self.topdir)
            return
       
        #turn filenames into only filenames, not with whole path
        file_list = [_.split('/')[-1] for _ in file_list]

        #add prefixes to separated file lists self attribute
        self.file_prefixes = file_prefixes
        for pref in file_prefixes:
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


        

    def load_glassman_events(self, g_file):
        self.g_events_file = g_file 
        dd = np.genfromtxt(g_file, delimiter=',', dtype=float)
        ts = dd[:,0] #timestamp in second since epoch
        ts = [datetime.fromtimestamp(_) for _ in ts] 
        v_meas = dd[:,1] #the high voltage monitor voltage measured at output of glassman supply
        v_cont = dd[:,2] #the output of AD2 DAC at time of event, driving the glassman supply HV. 

        #there is a linear relationship between the two recorded voltages, and it is measured
        #in a calibration procedure with stored "calibrations" data where we took a 1000x attenuation
        #DC HV probe and measured the at the end of the delivery cable as it ramped. 
        v_cont = [_*7.556 + 0.022 for _ in v_cont]
        v_meas = [_*7.58 - 0.046 for _ in v_meas]

        self.g_events = {"time":ts, "v_cont":v_cont, "v_meas":v_meas}


    #the only reason to save any loaded raw data
    #is to skip the step of separating by file prefix. 
    #so this function saves a pickle file of the separated file list.
    #output_pickle = "myfile.p" , and will be saved in the top data directory
    def save_raw(self, output_pickle):
        pickle.dump([self.separated_file_lists], open(self.topdir+output_pickle, "wb"))

    def load_raw_from_pickle(self, input_pickle):
        self.separated_file_lists = pickle.load(open(self.topdir+input_pickle, "rb"))


    #implement def load_reduced and load_wave and save here. 

    # load yaml configuration file
    def load_config(self,config_file):
        print("Loading config file " + config_file)
        with open(config_file) as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.config_dict = config 
        return config
    
    #-----end loading and saving functions-------#

    #create a dataframe containing just raw waveform info,
    #not processed waveform info. If time_paired is true, 
    #this checks if time-pairing has been performed. If not,
    #it runs the time pairing. If time-pairing is not desired,
    #it will just group events by how they appear in the file lists
    def create_wave_df(self, time_paired=False, allowed_dt = 20):
        #if no raw files have been parsed, throw an error
        #to have them load in. 
        if(len(self.separated_file_lists.keys()) == 0):
            print("No raw data files have been loaded into buffer.")
            print("Please call load_raw() with file prefixes")
            return

        #empty the present dataframe, if it is not already
        df_cols = []
        for pref in self.file_prefixes:
            df_cols.append(pref+"Timestamp")
            df_cols.append(pref+"SamplingPeriod")
            df_cols.append(pref+"0-data")
            df_cols.append(pref+"1-data")
        self.wave_df = pd.DataFrame(columns=df_cols)

        #because we want the option to use time paired or non time paired,
        #this list is temporary to allow for either option with the same data struct
        grouped_file_list = [] #same structure as self.time_paired_files
        
        if(time_paired):
            print("Time pairing the files.")
            if(len(self.time_paired_files) == 0):
                self.pair_by_timestamp(allowed_dt)
            else:
                print("Files were already time paired.")
            grouped_file_list = self.time_paired_files

        else:
            print("Pairing by their acquisition order. Not time synchronized")
            #pair files by their order in the file lists
            #the number of events in the filetype with largest number of events
            max_evts = max([len(self.separated_file_lists[pref]) for pref in self.file_prefixes])
            looper = tqdm(range(max_evts))
            for i in looper:
                pair = []
                for pref in self.file_prefixes:
                    if(i >= len(self.separated_file_lists[pref])):
                        pair.append(None)
                    else:
                        pair.append(self.separated_file_lists[pref][i])
                grouped_file_list.append(pair)

                
        #each pair of files is an event
        looper = tqdm(grouped_file_list, desc='Parsing waveform data from files')
        for pair in looper:
            wave_series = pd.Series(dtype='object') #a series associated with the event
            #loop through each prefix, and create column names based on the prefix
            for j, pref in enumerate(self.file_prefixes):
                #if this "file" in the pair is "None", don't file the pd.Series()
                if(pair[j] is None):
                    wave_series[pref+"Timestamp"] = None
                    wave_series[pref+"SamplingPeriod"] = None
                    wave_series[pref+'0-data'] = []
                    wave_series[pref+'1-data'] = []
                    continue

                #temporary filename variable
                fn = self.topdir+pair[j]
                #datetime timestamp
                wave_series[pref+"Timestamp"] = self.get_timestamp_from_filename(fn)
                wave_series[pref+"Timestamp"] = pd.to_datetime(wave_series[pref+"Timestamp"])
                #sampling period from header in microseconds
                wave_series[pref+"SamplingPeriod"] = self.get_sampling_period_from_file(fn)

                #get the data series from file, indexed by "ts" times, "0" channel 0, "1" channel 1
                d = pd.read_csv(fn, header=None, skiprows=20, names=['ts','0','1'], encoding='iso-8859-1')
                
                #if there is an empty column, or channel that is not activated
                #pd.read_csv fills with nans. 
                if(np.isnan(d['0']).any()):
                    wave_series[pref+'0-data'] = []
                else:
                    wave_series[pref+'0-data'] = d['0'].to_numpy() #numpy array

                if(np.isnan(d['1']).any()):
                    wave_series[pref+'1-data'] = []
                else:
                    wave_series[pref+'1-data'] = d['1'].to_numpy() #numpy array
                
            
            #append the event to the df
            self.wave_df = pd.concat([self.wave_df, wave_series.to_frame().transpose()], ignore_index=True)


        #at the end, we have the full wave df
        print("Finished filling a dataframe with " + str(len(self.wave_df.index)) + " waveform events")


    #attempts to pair events from one oscope to another
    #based on timestamp. The complexity of this has to do with 
    #the following downfalls of the AD2:
    #the scopes do not have a 1:1 event count or rate. Often times
    #one scope will miss the trigger. Often times, one scope will
    #take anomolously long to digitize. Often times, the number of
    #triggers per second on one scope will be much larger than the 
    #number of triggers on the other scope. 

    #at the moment, I'm only attempting to pair for two oscilloscopes,
    #i.e. assuming file_indices is length 2. Will get complicated for more. 

    #allowed_dt = window around one file to which another file
    #is considered a pair candidate (milliseconds).

    #Algorithm layout, an attempt to be efficient.
    #For prefix 1 (p1), loop through each timestamp. 
    #For each, grab a sublist of indexes +- 100 or so events in the
    #vicinity of the other scope, prefix 2 (p2). Find all events
    #in the other scope that have a timestamp within the allowed_dt. 
    #also calculate the minimum dT in this set of candidate pairs, and
    #remember which timestamp has that minimum dT. 
    #Save these for later after all events have been looped through. 

    #For each event and set of candidates, compare with a sublist
    #of p1 events +-25 events in the vicinity of the event in question. 
    #Multiple events will say that the same event from p2 is the closest event. 
    #Pair the events from this set of 50 that has the minimum dT. Remove that
    #paired event from the list of all other 50 and re-calculate their closest event. 
    #Eventually, some events will have no more within allowed_dt, and those events are lost. 
    def pair_by_timestamp(self, allowed_dt):
        #if no raw files have been parsed, throw an error
        #to have them load in. 
        if(len(self.separated_file_lists.keys()) == 0):
            print("No raw data files have been loaded into buffer.")
            print("Please call load_raw() with file prefixes")
            return

        #during raw data parsing, a dictionary of datetimes was created.
        #lets create a dictionary with a shorter name for ease of readability
        ts = self.separated_timestamps #indexed by file prefix

        #search subset half width (sshw):
        #the number of indices of the list of 
        #scope number 2 to search for nearby events. 
        #The larger this is, the longer the alg takes to run. 
        sshw = 50 
        #distillation half width (dhw):
        #the number of indices in scope 1 to compare
        #minimum dT pair candidates, and remove candidates
        #that are chosen pairs with another event. 
        dhw = int(sshw/2)

        if(len(self.file_prefixes) != 2):
            print("Presently, the time pairing algorithm is built for \
                two AD2 scopes only. Please take a look and adjust if you need.")
            return

        scope_0_times = ts[self.file_prefixes[0]]
        scope_1_times = ts[self.file_prefixes[1]]
        print(scope_0_times)

        
        

        #this is an intense data structure. this list
        #holds elements which are dictionaries, holding info
        #on the timestamp in scope 0, and a list of candidate 
        #timestamps in scope 1, the minimum timestamp, its
        #index in the list, and the minimum time difference. 
        scope_0_candidates = [] 

        #save counters that assess efficiency, losses, due to various reasons
        lost_from_allowed_dt = 0 
        lost_from_multiple_candidates = 0

        looper = tqdm(enumerate(scope_0_times), desc="Finding candidates within allowed_dt")
        #loop through all scope_0 times and find candidates in scope_1 
        #that are within sshw = +-50 list indices away
        for i0, stamp0 in looper:
            idx_hi = i0 + sshw
            idx_lo = i0 - sshw
            if(idx_hi >= len(scope_1_times)):
                idx_hi = len(scope_1_times) - 1
            if(idx_lo < 0):
                idx_lo = 0

            #initialize a stamp_dict for this scope_0 timestamp
            stamp_candidates = {"stamp": stamp0, "cands": []}
            #+1 in the upper range to include idx_hi
            #loop through and find all possible candidates,
            #defined as timestamps within the allowed_dt
            for i1 in range(idx_lo, idx_hi+1):
                stamp1 = scope_1_times[i1]
                dt = abs(stamp1 - stamp0)
                dt = self.get_milliseconds_from_timedelta(dt)
                if(dt <= allowed_dt):
                    stamp_candidates["cands"].append(stamp1)
            
            #If any candidates are found, calculate the best candidate
            #(closest in time) and record some metadata about that candidate.
            #safe this candidate in a list outside of this loop's scope
            if(len(stamp_candidates["cands"]) != 0):
                stamp_candidates = self.calculate_min_timestamp(stamp_candidates)
                scope_0_candidates.append(stamp_candidates)
            else:
                lost_from_allowed_dt += 1 

        #fig, ax = plt.subplots(figsize=(12,8))
        #dts = [_["mindt"] for _ in scope_0_candidates]
        #ax.hist(dts)
        #plt.show()

        #characterization: see what the time difference is
        #between a file and its next to closest event in other scope
        

        


        print("Lost " + str(lost_from_allowed_dt) + " events of " + str(len(scope_0_times)) + " / " + str(len(scope_1_times)) + " due to windowing")
        print("Dividing, : " + '{0:.2f}'.format(lost_from_allowed_dt/float(len(scope_0_times))) + " / " + '{0:.2f}'.format(lost_from_allowed_dt/float(len(scope_1_times))))
        #debugging, count how many stamps have multiple candidates
        multiplicities = {}
        for _ in scope_0_candidates:
            n = len(_["cands"])
            if(n not in multiplicities):
                multiplicities[n] = 0 
            multiplicities[n] += 1 

        print("Multiplicities with no distillation:", end=' ')
        print(multiplicities)

        scope_0_candidates = self.distill_timestamps(scope_0_candidates, dhw)
        #debugging, count how many stamps have multiple candidates
        multiplicities = {}
        for _ in scope_0_candidates:
            n = len(_["cands"])
            if(n not in multiplicities):
                multiplicities[n] = 0 
            multiplicities[n] += 1 
        print("Multiplicities with distillation:", end=' ')
        print(multiplicities)

        
        #fig, ax = plt.subplots(figsize=(12,8))
        #dts = [_["mindt"] for _ in scope_0_candidates if _["mindt"] is not None]
        #ax.hist(dts, label=str(np.std(dts)))
        #ax.legend()
        #plt.show()

        self.time_paired_files = []
        #finally, pair up events with their minimum dt candidate
        looper = tqdm(scope_0_candidates, desc="Determining final pairing and saving pairs...")
        for stamp in looper:
            if(len(stamp["cands"]) == 0):
                lost_from_multiple_candidates += 1
                continue
            file0 = self.get_filename_from_timestamp(stamp["stamp"], self.file_prefixes[0])
            file1 = self.get_filename_from_timestamp(stamp["minstamp"], self.file_prefixes[1]) 
            self.time_paired_files.append([file0, file1])

        #sort to be chronological
        self.time_paired_files = sorted(self.time_paired_files, key=lambda x: self.get_timestamp_from_filename(x[0]))
        
        #return losses
        starting_lengths = [len(self.separated_file_lists[_]) for _ in self.file_prefixes]
        return lost_from_allowed_dt, lost_from_multiple_candidates, starting_lengths


    def plot_timestamp_differences(self, ax=None):
        if(len(self.time_paired_files) == 0):
            print("You have not paired timestamps yet")
            return

        if(ax is None):
            fig, ax = plt.subplots(figsize=(12,8))

        time_diffs = []
        for pair in self.time_paired_files:
            delta = self.get_timestamp_from_filename(pair[0]) - self.get_timestamp_from_filename(pair[1])
            time_diffs.append(self.get_milliseconds_from_timedelta(delta))

        mean = str(round(np.mean(time_diffs),2))
        std = str(round(np.std(time_diffs), 2))
        binwidth = 1 #ms
        bins = np.arange(min(time_diffs), max(time_diffs), binwidth)
        ax.hist(time_diffs, bins, label="std: " + str(std) + ", mean: " + str(mean))
        ax.legend()
        ax.set_xlabel("differences in timestamp (ms)")
        ax.set_ylabel("events per" + str(binwidth) + " ms binwidth")
        return ax

    def get_timestamp_differences(self):
        if(len(self.time_paired_files) == 0):
            print("You have not paired timestamps yet")
            return

        

        time_diffs = []
        for pair in self.time_paired_files:
            delta = self.get_timestamp_from_filename(pair[0]) - self.get_timestamp_from_filename(pair[1])
            time_diffs.append(self.get_milliseconds_from_timedelta(delta))

        return time_diffs #ms








    #----functions related to the pairing of timestamps, on timestamp_dict structs-----#
    #this subfunction will recalculate all of the 
    #attributes of that dictionary, given the stamp_dict. 
    def calculate_min_timestamp(self, stamp_dict):
        l = stamp_dict["cands"] #(candidates)
        pivot = stamp_dict["stamp"] #the file in scope 0
        if(len(l) == 0):
            return {"stamp":pivot, "cands":l, "minidx": None, "mindt": None, "minstamp": None}
        #find the minimum time difference between the timestamp
        #of scope 0 and the close candidates in scope 1
        minidx, min_time_diff = min(enumerate(l), key=lambda x: abs(x[1] - pivot))
        #return a stamp_dict with updated values
        return {"stamp":pivot, "cands":l, "minidx": minidx, "mindt": self.get_milliseconds_from_timedelta(abs(pivot - min_time_diff)), "minstamp":l[minidx]}

    #loop through the candidates list and compare it to itself.
    #find and remove candidates that have other events with the same
    #candidate, but closer time difference. called "Distillation"
    def distill_timestamps(self, stamp_dict_list, dhw):
        stamp_dict_list_copy = [_ for _ in stamp_dict_list] #soft copy for removing elements
        looper = tqdm(enumerate(stamp_dict_list_copy), desc="Distilling repeated candidates ")
        for i0, stamp0 in looper:
            #look within a window of indices for possible repeats
            idx_hi = i0 + dhw
            idx_lo = i0 - dhw
            if(idx_hi >= len(stamp_dict_list)):
                idx_hi = len(stamp_dict_list) - 1
            if(idx_lo < 0):
                idx_lo = 0
            for j in range(idx_lo, idx_hi + 1):
                stamp1 = stamp_dict_list_copy[j]
                #check if either of these stamps have
                #the same candidate. If so, remove based on which
                #one is closer in time. 

                #if stamp1 == stamp0, skip... 
                if(stamp0 == stamp1):
                    continue

                #get the intersection of the list of candidates
                #(i.e. candidates that appear in both files)
                common_timestamps = list(set(stamp0["cands"]) & set(stamp1["cands"]))
                #if no common timestamps, continue
                if(len(common_timestamps) == 0):
                    continue

                #otherwise, remove and prefer candidates that are closer
                #note, we modify the stamp_dict_list not the stamp_dict_list_copy
                for ct in common_timestamps:
                    dt0 = abs(ct - stamp0["stamp"])
                    dt1 = abs(ct - stamp1["stamp"])
                    if(dt0 < dt1):
                        stamp_dict_list[j]["cands"].remove(ct)
                        stamp_dict_list[j] = self.calculate_min_timestamp(stamp_dict_list[j])
                    elif(dt0 > dt1):
                        stamp_dict_list[i0]["cands"].remove(ct)
                        stamp_dict_list[i0] = self.calculate_min_timestamp(stamp_dict_list[i0])
                    else:
                        #if they are equal? do nothing... 
                        continue

        return stamp_dict_list 

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
        dataset_date_truncated = self.date_of_dataset.split('-') 
        dataset_date_truncated = dataset_date_truncated[0] + "-" + dataset_date_truncated[-1]
        date_format = "%m-%y %d.%H.%M.%S.%f"
        try:
            date = datetime.strptime(dataset_date_truncated + " " + timedotted, date_format)
        except:
            #in some cases, the filename may not have a timestamp. in this case, return current time
            date = datetime.now()
        return date

    #takes a datetime as input and creates a filename
    def get_filename_from_timestamp(self, stamp, file_prefix):
        date_format = "%d.%H.%M.%S.%f"
        fn = datetime.strftime(stamp, date_format)
        fn = file_prefix+fn[:-3]+".csv" #:-3 is because it returns %f in microseconds
        return fn

    #takes a timedelta and returns milliseconds since epoch
    def get_milliseconds_from_timedelta(self,td):
        return td.total_seconds()*1000

    #gets the sampling period from file header in microseconds
    def get_sampling_period_from_file(self, filename):
        #parse header for the timestep
        f = open(filename, 'r', errors='ignore')
        ls = f.readlines()
        raw_sample_rate = ls[4]
        raw_sample_rate = raw_sample_rate.split(' ')[-1]
        raw_sample_rate = float(raw_sample_rate.split('H')[0])
        return (1.0/raw_sample_rate)*1e6 #microseconds


    #-------printing utilities-----#
    def print_timestamps_sidebyside(self, n = 30):
        for i in range(n):
            for pref in self.file_prefixes:
                if(i >= len(self.separated_timestamps[pref])):
                    continue
                print(pref+str(self.get_timestamp_from_filename(self.separated_file_lists[pref][i])) + ", ", end='')
            print("\n")


    #-------utilities---------#
    #this function returns n random events from the
    #wave_df; this requires wave_df to be loaded
    def get_random_waveforms(self, nevents):
        nevents_loaded = len(self.wave_df.index)
        if(nevents_loaded == 0):
            print("No events loaded into the waveform DataFrame, cant plot")
            return


        #if there are more events requested than exist, print that you are plotting all
        if(nevents >= nevents_loaded):
            print("Requested " + str(nevents) + " but only " + str(nevents_loaded) + " have been loaded.")
            print("Plotting all events")
            event_list = range(nevents_loaded)

        #otherwise, sample the range of event indices without repeating
        else:
            event_list = random.sample(range(nevents_loaded), nevents)

        return self.wave_df.loc[event_list]
    

    #get events from each prefix that occur within a specified
    #time period. The timem period is [datetime, datetime] objects. 
    def get_events_within_time_period(self, time_period):
        nevents_loaded = len(self.wave_df.index)
        if(nevents_loaded == 0):
            print("No events loaded into the waveform DataFrame, please create_wave_df")
            return
        
        outdf = pd.DataFrame()
        for pref in self.file_prefixes:
            mask = (time_period[0] < self.wave_df[pref+"Timestamp"]) & (self.wave_df[pref+"Timestamp"] < time_period[1])
            outdf = pd.concat([outdf, self.wave_df[mask]])
        
        return outdf






    def get_time_paired_files(self):
        return self.time_paired_files
    def get_separated_timestamps(self):
        return self.separated_timestamps
    def get_separated_filelists(self):
        return self.separated_file_lists
    def get_wavedf(self):
        return self.wave_df 
    def get_rawdf(self):
        return self.raw_df 
    def get_nevents_loaded(self):
        return len(self.wave_df.index)

    #----------------------reduction functions-----------------------#

    def create_reduced_df(self, config_file='../configs/default_config.yaml'):
        # check whether the waveform dataframe has been populated yet
        # if not, throw an error
        if len(self.wave_df.index)==0:
            print('Waveform data has not been extracted from raw files.')
            print('Please call create_waveform_df() before creating reduced dataframe.')
            return

        t0 = time.time()

        #analysis/processing parameters
        config_dict = self.load_config(config_file)
        #store for later, if pickled
        self.config_dict = config_dict 

        filter_tau = config_dict['filter_tau'] #ns #filtering anode signal gaussian
        zero_amplitude_threshold = config_dict['zero_amplitude_threshold'] #if max-sample is less than this, don't integrate
        fit_amplitude_threshold = config_dict['fit_amplitude_threshold'] #if max-sample is above this, fit the exponential to get tau and amplitude
        anode_integration_window = config_dict['anode_integration_window'] #us, skip integrating the baseline samples, just integrate this window about peak time.
        pmt_integration_window = config_dict['pmt_integration_window'] #us, encompasing the entire event
        rogowski_voltage = config_dict['rogowski_voltage']

        #empty the reduced dataframe, if it is not already
        reduced_df = pd.DataFrame()

        #because we want the option to use time paired or non time paired,
        #this list is temporary to allow for either option with the same data struct
        #grouped_file_list = self.grouped_file_list
        
        #each pair of files is an event
        #looper = tqdm(grouped_file_list, desc='Creating reduced data from raw files')
        #for pair in looper:

        # loop through all events identified in waveform dataframe creation
        #each pair of files is an event
        looper = tqdm(self.wave_df.iterrows(),desc='Reducing waveform data',total=len(self.wave_df.index))
        for i, event_series in looper:
            
            # create pandas series for theevent
            reduced_series = pd.Series(dtype='object')
            reduced_series['RogowskiVoltage'] = rogowski_voltage


            # loop through each prefix and create column names based on the prefix
            for j, pref in enumerate(self.file_prefixes):
                #include the timestamp elements, which are used to reference back to the waveforms themselves
                reduced_series[pref+"Timestamp"] = event_series[pref+"Timestamp"] #will turn out None for empty prefix entries


                # background subtraction depending on whether scope is for anode or pmts
                if("glitch" in pref or "anode" in pref):
                    # if there's no sampling period, the event doesn't have data for this scope
                    if(event_series[pref+'SamplingPeriod'] is None):

                        reduced_series['GlitchAmplitude'] = None
                        reduced_series['AnodeAmplitude'] = None
                        reduced_series['GlitchTau'] = None
                        reduced_series['AnodeTau'] = None
                        reduced_series['GlitchPeakidx'] = None
                        reduced_series['AnodePeakidx'] = None
                        reduced_series['GlitchIntegral'] = None
                        reduced_series['AnodeIntegral'] = None
                        continue
                    
                    
                    self.baseline_subtract_window(event_series,pref)
                    #calculate amplitude and integral
                    #put processed quantities like amplitude and integral in a reduced series
                    amplitudes, taus, peakidx, integrals = self.get_basic_waveform_properties(event_series, fit_amplitude_threshold, anode_integration_window, zero_amplitude_threshold,pref) #finds tau's if relevant

                    reduced_series['GlitchAmplitude'] = amplitudes[0] #in mV, negative or positive
                    reduced_series['AnodeAmplitude'] = amplitudes[1] #in mV, negative or positive
                    reduced_series['GlitchTau'] = taus[0] #time constant of exponential fit
                    reduced_series['AnodeTau'] = taus[1] #time constant of exponential fit
                    reduced_series['GlitchPeakidx'] = peakidx[0] #index referencing event_series['Data']
                    reduced_series['AnodePeakidx'] = peakidx[1] #index referencing event_series['Data']
                    reduced_series['GlitchIntegral'] = integrals[0] #mV*us
                    reduced_series['AnodeIntegral'] = integrals[1] #mV*us

                elif("pmt" in pref):

                    # if there's no timestamp, the event doesn't have data for this scope
                    if(event_series[pref+'SamplingPeriod'] is None):
                        reduced_series['PMT1Amplitude'] = None
                        reduced_series['PMT2Amplitude'] = None
                        reduced_series['PMT1Peakidx'] = None
                        reduced_series['PMT2Peakidx'] = None
                        reduced_series['PMT1Integral'] = None
                        reduced_series['PMT2Integral'] = None
                        reduced_series['PMT1std'] = None
                        reduced_series['PMT2std'] = None
                        continue

                    

                    self.baseline_subtract_window(event_series,pref)
                    #calculate amplitude and integral
                    #put processed quantities like amplitude and integral in a reduced series
                    amplitudes, taus, peakidx, integrals = self.get_basic_waveform_properties(event_series, fit_amplitude_threshold, pmt_integration_window, zero_amplitude_threshold,pref) #finds tau's if relevant
            
                    reduced_series['PMT1Amplitude'] = amplitudes[0] #in mV, negative or positive
                    reduced_series['PMT2Amplitude'] = amplitudes[1] #in mV, negative or positive
                    reduced_series['PMT1Peakidx'] = peakidx[0] #index referencing event_series['Data']
                    reduced_series['PMT2Peakidx'] = peakidx[1] #index referencing event_series['Data']
                    reduced_series['PMT1Integral'] = integrals[0] #million electrons
                    reduced_series['PMT2Integral'] = integrals[1] #million electrons
                    reduced_series['PMT1std'] = np.std(event_series[pref+'0-data'])
                    reduced_series['PMT2std'] = np.std(event_series[pref+'1-data'])

            reduced_df = pd.concat([reduced_df, reduced_series.to_frame().transpose()], ignore_index=True)

        self.reduced_df = reduced_df
        print("Done")
        #reinitialize
        reduced_df = pd.DataFrame()

    
    #----------------------analysis functions------------------------#

    #mean of the samples within a hard configured time window.
    #uses the config yaml file, without it, it assumes first 100 us 
    #One step up would be to peak find and go relative to a peak.
    #Discriminates between PMTs and Anode based on the prefix
    def baseline_subtract_window(self,event_series,prefix):
        dt = event_series[prefix+'SamplingPeriod']# in us
        
        if("anode" in prefix or "glitch" in prefix):
            window = self.config_dict["anode_baseline_window"]
        elif("pmt" in prefix):
            window = self.config_dict["pmt_baseline_window"]
        else:
            window = self.config_dict["baseline_window"]

        window_indices = [int(min(window)/dt), int(max(window)/dt)] #indexes defining the window

        for chan in range(2):
            raw_data = event_series[prefix+str(chan)+'-data']
            #if this prefix/event doesnt exist, it will be nans
            if(raw_data == []):
                continue
            baseline_buffer = raw_data[min(window_indices):max(window_indices)]
            baseline = np.mean(baseline_buffer)
            event_series[prefix+str(chan)+'-data'] = raw_data - baseline



        
    def get_basic_waveform_properties(self, event_series, fit_amplitude_threshold, window, zero_amplitude_threshold, prefix):
        amps = [] #amplitudes
        taus = [] #exp time constants, none usually
        pidx = [] #peak times
        integrals = []

        for chan in range(2):
            rawdata = event_series[prefix+str(chan)+'-data']

            if(rawdata == []):
                amps.append(None)
                taus.append(None)
                pidx.append(None)
                continue

            #for i, rawdata in enumerate(event_series['Data']):
            #returns highest absolute value, negative or positive polar
            #i.e. max(-5, 3, key=abs) will return -5 (as opposed to 5)
            maxval = max(rawdata.min(), rawdata.max(), key=abs) 
            maxidx = np.where(rawdata == maxval)[0][0]

            if("anode" in prefix or "glitch" in prefix):
                #if the signal is larger than the threshold, do an exponential fit. 
                if((abs(maxval) > fit_amplitude_threshold) and False):
                    #do fit later
                    dt = event_series[prefix+'SamplingPeriod'] #us
                    times = np.array(np.arange(0, len(rawdata)*dt, dt))
                    p0 = [maxval, times[maxidx], dt*10, 150e3] #guess for fitter, amp and time of peak. 
                    #somehow anode_fit_function got lost. reimplement this later.
                    popt, pcov = curve_fit(anode_fit_function, times, rawdata, p0=p0)
                    fity = anode_fit_function(times, *popt)
                    """
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot([_/1e6 for _ in times], fity, label=popt)
                    ax.plot([_/1e6 for _ in times], rawdata)
                   #ax.set_xlim([times[maxidx]/1e6 - 0.1, times[maxidx]/1e6 + 0.1])
                    ax.legend()
                    plt.show()
                    """
                    pidx.append(maxidx)
                    amps.append(maxval)
                    taus.append(None)


                #otherwise, just return peak time, no tau, and amplitude
                else:
                    amps.append(maxval)
                    taus.append(None)
                    pidx.append(maxidx) 

            elif("pmt" in prefix):
                amps.append(maxval)
                taus.append(None)
                pidx.append(maxidx)


        #integration functions
        for chan in range(2):
            rawdata = event_series[prefix+str(chan)+'-data']

            if(rawdata == []):
                integrals.append(None)
                continue
            
            if("anode" in prefix or "glitch" in prefix):
                anode_peak = pidx[chan]
                dt = event_series[prefix+'SamplingPeriod'] #us
                lowidx = int(anode_peak + min(window)/dt)
                hiidx = int(anode_peak + max(window)/dt) 
                integ_data = rawdata[lowidx:hiidx]*1000 #mV
                integrals.append(np.trapz(integ_data, dx=dt)) #in mV*us


            elif("pmt" in prefix):
                dt = event_series[prefix+'SamplingPeriod'] #us
                lowidx = int(pidx[chan] + min(window)/dt)
                hiidx = int(pidx[chan] + max(window)/dt) #1e3 because window in us
                integ_data = rawdata[lowidx:hiidx] #V
                integ_vs = np.trapz(integ_data, dx=dt)/1e6 #V*s
                integ_coul = integ_vs/50.0 #50 ohms
                integ_mega_elec = (integ_coul/1.6e-19)/1e9 #billion electrons
                integrals.append(integ_mega_elec) #billion electrons





        return amps, taus, pidx, integrals


    #for data that is triggered in a time-consistent way,
    #it can be easy to obtain an average waveform from the dataset. 
    def get_average_wave(self):
        if(self.avg_wave is not None):
            return self.avg_wave

        avg_wave = {}
        for ch in range(2):
            avg_wave[ch] = []

            for i, row in self.wave_df.iterrows():
                self.baseline_subtract_window(row, "acq") #baseline subtracts in place
                if(len(avg_wave[ch]) == 0):
                    avg_wave[ch] = np.array(row["acq"+str(ch)+"-data"])
                else:
                    avg_wave[ch] = avg_wave[ch] + np.array(row["acq"+str(ch)+"-data"])

            #normalize
            avg_wave[ch] /= len(self.wave_df.index)

        self.avg_wave = avg_wave
        return avg_wave

    #utility to quickly get a timebase from first event in dataset
    def get_timebase(self, prefix):
        event = self.wave_df.iloc[0]
        print(len(event[prefix+'0-data']))
        ts = np.array(range(len(event[prefix+'0-data'])))*event[prefix+'SamplingPeriod']
        return ts
    
    #look at event times and calculate the time difference between
    #adjacent triggers
    def get_event_time_differences(self, prefix):
        ts = self.separated_timestamps[prefix]
        if(len(ts) == 0):
            print("No files loaded into raw yet")
            return
        
    
        dts = [(ts[i] - ts[i-1]).total_seconds() for i in range(1, len(ts))] #seconds
        return dts









