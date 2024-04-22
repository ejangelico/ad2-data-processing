#This class is meant to distinguish tools geared toward waveform analysis
#and pre-reduction with tools geared towards interacting with the
#reduced data dataframe, which contains only quantities after waveform reduction. 

#Furthermore, it is really geared towards Run level analysis, where
#multiple reduced dataframes are combined. (but not always the case)


import numpy as np
import matplotlib.pyplot as plt 
import pickle
import os
import yaml
import pandas as pd
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d


class AnalysisTools:
    #reduced_df_pickle is the filename of the pickle file containing reduced dataframe
    #ramp_topdir is a directory for which the class looks for all "ramp.txt" files
    #recursively in order to identify ramp timing with timestamps in the reduced DF. 
    def __init__(self, reduced_df_pickle, config_file, title='', ramp_topdir=None):
        self.fn = reduced_df_pickle
        self.df = None
        self.title = title

        self.config_file = config_file
        self.config = {}
        self.load_config()

        self.ad2_chmap = {}
        self.struck_chmap = {}
        self.load_chmaps()

        self.all_sw_chs = list(self.ad2_chmap.keys()) + list(self.struck_chmap.keys())

        #to reproduce timing of ramp data, we need to collect
        #all ramp files (and gevents) from the Runs
        self.ramp_topdir = ramp_topdir
        self.ramp_data = pd.DataFrame() #ramp data flat and linear, not separated into chunks associated with ramps and flat tops
        self.g_event_data = pd.DataFrame()
        self.time_duration_map = {"t":[], "dur":[], "v":[]} # a 1:1 mapping between timestamps (unix epoch) and duration above 0V to cut out down-time from time plots. 
        if(self.ramp_topdir != None):
            self.load_hv_textfiles() #parses the files into ramp_data and g_event_data
            self.correct_hv_data() #analyzes the ramp data to separate into a list of ramps and a list of flat tops
            self.create_time_duration_map() #creates the time_duration_map

    #reloads the df object from the pickle file
    def load_dataframe(self):
        self.df = None
        self.df = pickle.load(open(self.fn, "rb"))[0]


    def load_config(self):
        if(os.path.isfile(self.config_file)):
            #safe read this yaml file
            with open(self.config_file, 'r') as stream:
                try:
                    self.config = yaml.safe_load(stream)
                except:
                    print("Had an issue reading the config file, make sure it is a .yaml or .yml file")
                    return None
        else:
            print("Couldnt load configuration file... {} doesn't exist".format(self.config_file))

    def load_chmaps(self):
        for ad2 in self.config["ad2_reduction"]:
            for i, sw_ch in enumerate(self.config["ad2_reduction"][ad2]["channel_map"]["software_channel"]):
                #for saving these software channels for easier access
                self.ad2_chmap[sw_ch] = self.config["ad2_reduction"][ad2]["channel_map"]["prereduced_index"][i]

        for i, sw_ch in enumerate(self.config["struck_reduction"]["channel_map"]["software_channel"]):
            self.struck_chmap[sw_ch] = self.config["struck_reduction"]["channel_map"]["prereduced_index"][i]

    def clear_hv_data(self):
        self.ramp_data = pd.DataFrame()
        self.g_event_data = pd.DataFrame()
        self.ramps = []
        self.flat_tops = []
        self.time_duration_map = {"t":[], "dur":[], "v":[]}

    def load_hv_textfiles(self):
        #one distinction of this function in this class is that
        #it is flexible compared to the Dataset class to the point where
        #it looks for all "ramp.txt" files recursively starting from
        #a top directory in the heirarchy. This means the ramp.txt could
        #be from one dataset within a run (if topdir is "ds01/") or could
        #be from multiple datasets in a run (if topdir is "Run8/")
        #we have a few different HV supplies used for different voltage ranges.
        #This conversion text file just has a single text floating point in it
        #that represents the DAC to kV conversion factor.
        
        self.clear_hv_data()

        for root, dirs, files in os.walk(self.ramp_topdir):
            #only process rampdata if you're in a directory with a ramp.txt file. 
            if(self.config["ramp_name"] in files):

                if("dac_conversion.txt" in files):
                    temp = open(os.path.join(root, "dac_conversion.txt"), "r")
                    l = temp.readlines()[0]
                    dac_conv = float(l)
                else:
                    dac_conv = 4 #use the 40 kV glassman value. 
                
                
                #load the rampfile data
                d = np.genfromtxt(os.path.join(root, self.config["ramp_name"]), delimiter=',', dtype=float)
                ts = d[:,0] #seconds since that epoch above
                v_dac = np.array(d[:,1]) #voltage in volts applied to the control input of the HV supply. needs to be converted for actualy HV applied. 
                v_mon = np.array(d[:,2]) #monitored, if plugged into the external monitor of the supply
                c_mon = np.array(d[:,3]) #monitoring of current, if plugged in. 
                v_app = v_dac*dac_conv
                v_mon = v_mon*dac_conv
                c_mon = c_mon*dac_conv

                temp_dict = {}
                temp_dict["t"] = ts
                temp_dict["v_app"] = v_app
                temp_dict["v_mon"] = v_mon #THIS is the more accurate voltage being applied, not v_app. See calibration folder of 40 kV glassman supply. 

                #add to the ramps dataframe
                self.ramp_data = pd.concat([self.ramp_data, pd.DataFrame(temp_dict)], axis=0, ignore_index=True)


            #load the g_events data, if it exists
            if(self.config["g_events_name"] in files):
                d = np.genfromtxt(os.path.join(root, self.config["g_events_name"]), delimiter=',', dtype=float)
                #there is a silly thing with genfromtxt where if its a 1 line file, it makes a 1D array instead of the usual
                #2D array. This line forces it into a 2D array so the other lines don't need some other if statement. 
                if(len(d.shape) == 1): 
                    d = np.array([d])
                #if it is an empty file, continue
                if(d.shape[1] > 0):
                    ts = d[:,0] #seconds since that epoch above
                    v_mon = np.array(d[:,1])*dac_conv
                    v_app = np.array(d[:,2])*dac_conv

                    temp_dict = {}
                    temp_dict["t"] = ts
                    temp_dict["v_app"] = v_app
                    temp_dict["v_mon"] = v_mon

                    self.g_event_data = pd.concat([self.g_event_data, pd.DataFrame(temp_dict)], axis=0, ignore_index=True)

        #sort both dataframes by time
        if(len(self.g_event_data.index) != 0):
            self.g_event_data = self.g_event_data.sort_values("t")
        if(len(self.ramp_data.index) != 0):
            self.ramp_data = self.ramp_data.sort_values("t")


    #This function attemps to correct the ramp data 
    #to act as a reference for high voltage interpolation
    #and exponential smoothing. It also will eventually
    #be a good place to put a better calibratin of HV applied. 
    def correct_hv_data(self):
        if(len(self.ramp_data.index) == 0):
            print("No ramp data in this dataset")
            return
        
        #timestamps in seconds
        ts_a = np.array(self.ramp_data["t"])
        ts_g = np.array(self.g_event_data["t"])

        #the algorithm heavily uses the v_applied data
        #because it is smooth and has no noise. Later,
        #once the time series is corrected, the v_mon
        #stream will be used to calibrate the HV applied.
        vs_a = np.array(self.ramp_data["v_app"])
        vs_m = np.array(self.ramp_data["v_mon"])
        vs_ga = np.array(self.g_event_data["v_app"])
        vs_gm = np.array(self.g_event_data["v_mon"])
        
        #the times of monitored voltages in the ramp.txt file
        #are not always the same timestep, but they are always
        #less than 3 seconds UNLESS we had a reset event or
        #are turning off the system. Here are a few thresholds
        #for the algorithm to use. 
        reset_thresh = 3.0 #seconds
        #eventually the time series of voltages will be
        #divided evenly by some fine time scale, that later
        #gets exponential smoothing based on known timeconstants
        #in the circuit. The location before 100M resistor to chamber 
        #is 50 ms time constant, the rogowski potential is 4 ms timeconstant. 
        fine_dt = 0.01 #seconds this is shorter than 50 ms and greater than 4 ms

        #first process the g_events, which always cause a reset to occur. 
        #The operating principle of this algorithm is to add data points to the 
        #time stream based on what the next starting voltage is if there is a large
        #time gap or a gevent. These two lists are just those additional points,
        #which will then be appended to the original list and re-sorted accordingly. 
        new_ts = []
        new_vs_a = [] 
        new_vs_m = []
        for i, gt in enumerate(ts_g):
            #find the closest time in the ramp data
            idx = (np.abs(ts_a - gt)).argmin()
            #make sure that the next data point in time is greater
            #than the time threshold. If not, find the next one that is.
            end_flag = False #if it reaches the end of the full time stream
            while True:
                if(idx < len(ts_a) - 1):
                    if(ts_a[idx+1] - ts_a[idx] < reset_thresh):
                        idx += 1
                        continue
                    else:
                        break
                else:
                    end_flag = True
                    break
            
            if(end_flag):
                break

            #add a data point to the ramp data that is equivalent to the
            #g-event measurement. As this is the final voltage measured before reset. 
            new_ts.append(gt)
            new_vs_a.append(vs_ga[i])
            new_vs_m.append(vs_gm[i])
            #add a point that is fin_dt away from the g-event reset time
            #and has a voltage equal to the next starting voltage
            #(far in the future)
            new_ts.append(gt + fine_dt)
            new_vs_a.append(vs_a[idx+1])
            new_vs_m.append(vs_m[idx+1])
        


        #add the lists back to the time series and resort
        ts_a = np.append(ts_a, new_ts)
        vs_a = np.append(vs_a, new_vs_a)
        vs_m = np.append(vs_m, new_vs_m) #artificially adding a perfect value to the monitor voltage. 
        #sort the lists by ts_a simultaneous
        idx = np.argsort(ts_a)
        ts_a = ts_a[idx]
        vs_a = vs_a[idx]
        vs_m = vs_m[idx]

        #next, with the g-events accounted for, we will re-parse the
        #voltage-time stream looking for any gaps in time. Correct
        #those gaps with an fine_dt step with the next voltage value. 
        new_ts = []
        new_vs_a = []
        new_vs_m = []
        for i in range(len(ts_a) - 1):
            if(ts_a[i+1] - ts_a[i] > reset_thresh):
                #add a point that is fine_dt away from the reset time
                #and has a voltage equal to the next starting voltage
                #(far in the future)
                new_ts.append(ts_a[i] + fine_dt)
                new_vs_a.append(vs_a[i+1])
                new_vs_m.append(vs_m[i+1])

        #repeat the sorting process

        #add the lists back to the time series and resort
        ts_a = np.append(ts_a, new_ts)
        vs_a = np.append(vs_a, new_vs_a)
        vs_m = np.append(vs_m, new_vs_m) #artificially adding a perfect value to the monitor voltage. 
        #sort the lists by ts_a simultaneous
        idx = np.argsort(ts_a)
        ts_a = ts_a[idx]
        vs_a = vs_a[idx]
        vs_m = vs_m[idx]


        self.ramp_data = pd.DataFrame()
        self.ramp_data["t"] = ts_a
        self.ramp_data["v_app"] = vs_a
        self.ramp_data["v_mon"] = vs_m


    #this function will make a 1-to-1 mapping between
    #timestamp and how long the system has been above 
    #a voltage threshold (close to 0V) during the run. 
    #Effectively, this collapses large breaks like overnight
    #sleep and lunch into a time axis that is easier to interpret. 
    #the voltage threshold not being "time above 0V" is because
    #the monitor and applied voltages are corrected in such a way that
    #sometimes has a small non-physical non-zero offset. Default is 200V. 
    def create_time_duration_map(self, v_thresh=0.2):
        self.time_duration_map = {"t":[], "dur":[], "v":[]}
        vs_r = np.array(self.ramp_data["v_app"])
        ts_r = np.array(self.ramp_data["t"])
        for i in range(1, len(ts_r)):
            if(vs_r[i] > v_thresh and vs_r[i-1] > v_thresh):
                #time difference between this data point and last one
                dt = ts_r[i] - ts_r[i-1]

                #condition for the first datapoint
                if(self.time_duration_map["t"] == []):
                    self.time_duration_map["t"].append(ts_r[i-1])
                    self.time_duration_map["t"].append(ts_r[i])
                    self.time_duration_map["dur"].append(0)
                    self.time_duration_map["dur"].append(dt)
                    self.time_duration_map["v"].append(vs_r[i-1])
                    self.time_duration_map["v"].append(vs_r[i])
                else:
                    self.time_duration_map["t"].append(ts_r[i])
                    self.time_duration_map["dur"].append(self.time_duration_map["dur"][-1] + dt)
                    self.time_duration_map["v"].append(vs_r[i])

    #This is the function that sets the high voltage value
    #for each event based on the timestamp of that event. 
    #It should be applied only to corrected ramp data. 
    #It does an interpolation as well as an exponential time
    #smoothing based on the 50 ms time constant of the applied
    #voltage to the 100M resistor at the base of the chamber. 
    def get_hv_at_time(self, t):    
        if(len(self.ramp_data.index) == 0):
            return None

        tb = 5 #seconds, either end around the timestamp in question to look at. 
        tb_b = [t - tb, t + tb]
        #find the closest times in the ramp data to those time bounds. 
        idx_l = (np.abs(np.array(self.ramp_data["t"]) - tb_b[0])).argmin()
        idx_r = (np.abs(np.array(self.ramp_data["t"]) - tb_b[1])).argmin()
        
        #then, because large gaps can sometimes be > 5 seconds, we will add 2
        #data points on either end of this window to make sure the interpolation
        #covers the range. 
        idx_l -= 2
        idx_r += 2
        if(idx_l < 0): idx_l = 0
        if(idx_r > len(self.ramp_data.index)): idx_r = len(self.ramp_data.index)
        
        ts = np.array(self.ramp_data["t"])[idx_l:idx_r]
        vs = np.array(self.ramp_data["v_app"])[idx_l:idx_r]
        if(np.min(ts) > t or np.max(ts) < t):
            #if this time is somehow still not in the range of the data
            return None

        #at this point, it is possible that the time range is days long,
        #making the algorithm completely rediculously long. So we linearly
        #interpolate the raw data to get a TRUE window of 1 second on either
        #side to then feed to the exponential filter. 
        s_raw = interp1d(ts, vs) #interpolate raw data over a 10 to many second window
        tb = 1 #seconds
        tb_b = [t - tb, t + tb]
        fine_dt = 0.01 #seconds this is shorter than 50 ms and greater than 4 ms
        #linearly interpolate to even the time domain
        ts_fine = np.arange(tb_b[0], tb_b[1], fine_dt) #only execute that interpolation in a small region
        vs_fine = s_raw(np.array(ts_fine))

        #exponential filter with a time constant of 50 ms
        exp_tau = 0.05 #seconds
        vs_exp, ts_exp = self.exponential_filter(ts_fine, vs_fine, exp_tau)

        #return the value that is the linear interpolation of the filtered wave
        s = interp1d(ts_exp, vs_exp)
        v_temp = s(t)

        #apply a calibration factor here based on a calibration file. 
        #I do not have yet a calibration for the 75 kV glassman. I do for the
        #40 kV glassman. When that gets done, implement it here. 
        v_temp = 1*v_temp

        return v_temp



    def exponential_filter(self, ts, vs, tau):
        filt_vs = [vs[0]]
        a = np.exp(-1*np.abs(ts[0] - ts[1])/tau)
        for i in range(1, len(vs)):
            filt_vs.append(a*filt_vs[i-1] + (1 - a)*vs[i])
        return np.array(filt_vs), ts

    def get_gevents_in_window(self, t0, t1):
        if(len(self.g_event_data.index) == 0):
            return None

        mask = (self.g_event_data["t"] >= t0) & (self.g_event_data["t"] <= t1)
        return self.g_event_data[mask]

    #small tool for finding duration given a 
    #timestamp using the time_duration_map
    def get_duration_from_timestamp(self, t):
        if(self.time_duration_map["t"] == []):
            print("Time duration map is not constructed. Please load hv textfiles and identify ramps.")
            return None

        idx = (np.abs(np.array(self.time_duration_map["t"]) - t)).argmin()
        return self.time_duration_map["dur"][idx]

    

    #this will get waveforms, from their waveform
    #level pre-reduced files, that pass a mask
    #on the df. Pass a list of sw_chs to grab. 
    def get_waveforms_from_cuts(self, mask, sw_chs):
        dd = self.df[mask] #masked df
        output_events = {}
        for sw_ch in sw_chs:
            #get the index of the prereduced "Data" element that this corresponds to
            if(sw_ch in self.struck_chmap):
                prered_idx = self.struck_chmap[sw_ch]
            else:
                prered_idx = self.ad2_chmap[sw_ch]


            output_events[sw_ch] = []
            filenames = list(dd["ch{:d} filename".format(sw_ch)])
            evidx = list(dd["ch{:d} evidx".format(sw_ch)]) 

            filenames_set = list(set(filenames)) #unique filenames only. 
            for f in filenames_set:
                if(f == None or isinstance(f, float)): continue
                df, date = pickle.load(open(f, "rb"))
                for i in range(len(evidx)):
                    if(filenames[i] == f):
                        event = df.iloc[evidx[i]]
                        #this event is now a row that has a "Data" element with
                        #all channels from that DAQ system included. reduce it only
                        #to the sw_ch data stream, as that is what was requested.  
                        output_events[sw_ch].append({"Seconds":event["Seconds"], "Nanoseconds":event["Nanoseconds"], "Data":event["Data"][prered_idx]})

        return output_events


    #this will get waveforms, from the prereduced
    #files, based on an input pandas dataframe
    #as opposed to a mask like in the above function. 
    def get_waveforms_from_df(self, dd, sw_chs):
        output_events = {}
        for sw_ch in sw_chs:
            #get the index of the prereduced "Data" element that this corresponds to
            if(sw_ch in self.struck_chmap):
                prered_idx = self.struck_chmap[sw_ch]
            else:
                prered_idx = self.ad2_chmap[sw_ch]



            output_events[sw_ch] = []
            filenames = list(dd["ch{:d} filename".format(sw_ch)])
            evidx = list(dd["ch{:d} evidx".format(sw_ch)]) 

            filenames_set = list(set(filenames)) #unique filenames only. 
            for f in filenames_set:
                #reject None's and NaNs
                if(f == None or isinstance(f, float)): continue
                df, date = pickle.load(open(f, "rb"))
                for i in range(len(evidx)):
                    if(filenames[i] == f):
                        event = df.iloc[evidx[i]]
                        #this event is now a row that has a "Data" element with
                        #all channels from that DAQ system included. reduce it only
                        #to the sw_ch data stream, as that is what was requested.  
                        output_events[sw_ch].append({"Seconds":event["Seconds"], "Nanoseconds":event["Nanoseconds"], "Data":event["Data"][prered_idx]})

        return output_events

    
    #takes a dataframe that is input_events, probably
    #formed by a mask such as charge amplitude > 5 mV. 
    #sw_ch is the channel to get time info from input_events
    #Finds events with timestamps (sec and nanosec) within 
    #a provided half-coincidence window. 

    #A key element that makes this function somewhat long is that we do NOT
    #want to return copies of events. This happens, for example, when charge channel
    #3 is used as the sw_ch, and the coinc window is large enough to include
    #multiple charge depositions in the window, each charge deposition will
    #have its own time through the loop for coincidence and duplicate in the 
    #output list of dataframes. 
    def get_coincidence(self, input_events, sw_ch, coinc, coinc_ns):
        event_dfs = [] #list of dataframes that match coincidence cuts for each event
        for i, ev in input_events.iterrows():
            t0 = ev["ch{:d} seconds".format(sw_ch)]
            t0_ns = ev["ch{:d} nanoseconds".format(sw_ch)]
            
            #this loop includes all channels that have
            #events within the provided time window, even the 
            #input sw_ch
            temp_df = pd.DataFrame()
            for sch in self.all_sw_chs:
                mask = ((self.df["ch{:d} seconds".format(sch)] - (t0 - coinc) + ((self.df["ch{:d} nanoseconds".format(sch)] - (t0_ns - coinc_ns))/1e9)) >= 0) &\
                    ((self.df["ch{:d} seconds".format(sch)] - (t0 + coinc) + ((self.df["ch{:d} nanoseconds".format(sch)] - (t0_ns + coinc_ns))/1e9)) <= 0)
                selected = self.df[mask]
                #one issue with the reduced dataframe in this context is that
                #particular rows of the dataframe may contain events in different channels
                #that happen at completely different times. So in this loop we've found
                #events for channel "sch" that are in coincidence, but those same rows
                #may contain events in another channel at a different time.
                #The following line selects a sub-df of this df containing ch{:d} in the column.
                ch_subdf = selected[selected.columns[selected.columns.str.contains('ch{:d}'.format(sch))]]
                temp_df = pd.concat([temp_df, ch_subdf], ignore_index=True)
                
            event_dfs.append(temp_df)

        #this is the start of the code to remove duplicates. It uses the evidx attribute
        #to only keep events that have the largest set of evidxs for each unique combination
        #of evidxs in a single event window. 
        filtered_event_dfs = []
        #first get all evidxs using the sw_ch of interest
        evidxs = []
        for ev in event_dfs:
            mask = ~(np.abs(ev["ch{:d} evidx".format(sw_ch)]).isna())
            evidxs.append(tuple(ev[mask]["ch{:d} evidx".format(sw_ch)]))

        #find only unique evidxs
        evidxs_set = list(set(evidxs))
        #for each event, remove any event that is a subset of a different
        #event. This will keep only the largest sets that contain other subsets. 
        kept_evidxs = []
        for i in range(len(evidxs_set)):
            test_set = set(evidxs_set[i])
            keep = True #switches if we dont want to keep
            for j in range(len(evidxs_set)):
                if(i == j): continue
                if(test_set.issubset(evidxs_set[j])):
                    keep = False
                    break
            if(keep == True):
                kept_evidxs.append(evidxs_set[i])

        #propagate that back to the event_dfs
        added_evidxs = [] #to make sure not to add duplicates again
        for ev in event_dfs:
            mask = ~(np.abs(ev["ch{:d} evidx".format(sw_ch)]).isna())
            if(tuple(ev[mask]["ch{:d} evidx".format(sw_ch)]) in kept_evidxs and (tuple(ev[mask]["ch{:d} evidx".format(sw_ch)]) not in added_evidxs)):
                added_evidxs.append(tuple(ev[mask]["ch{:d} evidx".format(sw_ch)]))
                filtered_event_dfs.append(ev)

        event_dfs = filtered_event_dfs
            

        return event_dfs
    

    #use this function to get light triggers
    #where the trigger is NOT due to the charge
    #trigger output alone 
    def get_light_triggers(self):
        pmt_thresh = 4 #times the noise std
        #It is perfectly reasonable for the charge trigger output to come in
        #during a light pulse. So instead of making this function about ch4 being
        #above or below threshold, I will make it about the PMT channels having something
        #other than noise. 

        #later, for this mask, use min-max sample instead of a baseline check, as the baseline is often affected
        #by the HV phenomena in this light channel. 
        mask = (~self.df["ch0 amp"].isna()) & (((self.df["ch0 amp"] - self.df["ch0 baseline"]) > pmt_thresh*self.df["ch0 noise"]) \
                                               | ((self.df["ch1 amp"] - self.df["ch1 baseline"]) > pmt_thresh*self.df["ch1 noise"]))
        return self.df[mask]
    

    #this function is intended to find all light events
    #that are far away in time from any charge related events. Requires
    #some time period "T" away from charge triggers with seconds precision only. 
    #N being None or being an integer triggers a more efficient version of this function.
    #if you just want 1000 random light triggers, you can randomly select and check if has
    #any proximity to a charge trigger. If it is None, it tries to get ALL light triggers. 
    def get_cosmic_triggers(self, T, N=None):
        #getting all light triggers
        light_df = self.get_light_triggers()
        #select only those triggers that are a time T at least away
        #from any charge triggers. Use the amplitude threshold of the
        #charge channels to determine if a charge trigger is noise or not, 
        #and accept events where there is just noise waveform on charge scope. 
        amp_thr = self.config["ad2_reduction"]["glitch"]["fit_amplitude_threshold"] #sigma
        ch_df = self.df[(~self.df["ch3 amp"].isna())]
        ch_df = ch_df[np.abs(ch_df["ch3 amp"]) > amp_thr*ch_df["ch3 noise"]]

        output_light_df = pd.DataFrame()

        if(N == None):
            #ch_df is usually much smaller, so I will loop through
            #each charge event and remove all light events that fall within
            #the timeframe. 
            N_good_events = 0
            for i, row in light_df.iterrows():
                if(N_good_events % 100 == 0): print("On event {:d} of {:d}".format(N_good_events, len(light_df.index)))
                t0 = row["ch0 seconds"]
                charge_prox_mask = (np.abs(ch_df["ch3 seconds"] - t0) < T)
                if(len(ch_df[charge_prox_mask].index) == 0):
                    output_light_df = pd.concat([output_light_df, row], axis=1)
                    N_good_events += 1

        else:
            N = int(N)
            chosen_indices = []
            N_good_events = 0
            while True:
                if(N_good_events >= N): break
                if(N_good_events % 100 == 0): print("On event {:d} of {:d}".format(N_good_events, N))
                index = np.random.randint(0, len(light_df.index))
                if(index in chosen_indices): continue
                chosen_indices.append(index)
                t0 = light_df.iloc[index]["ch0 seconds"]
                charge_prox_mask = (np.abs(ch_df["ch3 seconds"] - t0) < T)
                if(len(ch_df[charge_prox_mask].index) == 0):
                    output_light_df = pd.concat([output_light_df, light_df.iloc[index]], axis=1)
                    N_good_events += 1

        output_light_df = output_light_df.transpose()
        output_light_df = output_light_df.reset_index(drop=True)

        return output_light_df







    


    

