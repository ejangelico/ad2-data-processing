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
        self.ramps = [] #a list of individually separated ramps
        self.flat_tops = [] # a list of individually separated flat tops in voltage applied
        self.time_duration_map = {"t":[], "dur":[], "v":[]} # a 1:1 mapping between timestamps (unix epoch) and duration above 0V to cut out down-time from time plots. 
        if(self.ramp_topdir != None):
            self.load_hv_textfiles() #parses the files into ramp_data and g_event_data
            self.identify_ramps() #analyzes the ramp data to separate into a list of ramps and a list of flat tops

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
        
        self.ramp_data = pd.DataFrame() #ramp data flat and linear, not separated into chunks associated with ramps and flat tops
        self.g_event_data = pd.DataFrame()

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
                temp_dict["e_app"] = v_mon/self.config["rog_gap"] #converting to assumed electric field in kV/cm
                temp_dict["c_mon"] = c_mon

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


    #this function takes the flat, 1D data of HV ramp info and separates
    #it into indexable ramps and flat tops. It additionally attempts to 
    #make a mapping between a timestamp and the "total duration above 0V"
    #to remove many hour/day spaces between ramps and datasets. It does so 
    #by having an additional list of "durations" that is 1:1 indexible
    #with the time list.
    def identify_ramps(self, ref=None):
        if(len(self.ramp_data.index) == 0):
            print("No ramp data in this dataset")
            return
        ts = np.array(self.ramp_data["t"])
        if(ref == None):
            ref = "v_mon"
        if('mon' in ref):
            ref = "v_mon"
            vs = np.array(self.ramp_data[ref])
            #gaussian smooth the noisy voltage monitor data at 2 sample interval
            vs = gaussian_filter(vs, 2)
        else:
            ref = "v_app"
            vs = np.array(self.ramp_data[ref])


        #this is not measured data, rather is applied data, so everything is digitally generated and smooth. 
        #instead of using peak finding, its going to call a ramp a period where derivative is positive. 
        #it will call a flat top a period where the value doesn't change. 
        ramps = [{"t":[], "v":[]}]
        flat_tops = [{"t":[], "v":[]}]
        ramping = True #always starts with a ramp as opposed to flat top. use this to trigger a transition in state
        last_vdiff = None
        #time to duration mapping
        td_map = {"t":[ts[0]], "dur":[0], "v":[vs[0]]} #both in seconds
        #threshold for time to duration mapping, duration about "thresh" kV
        td_thresh = 0.0

        state_change_thresh = 0.000 #kilovolts, threshold for whether the derivative has changed. 
        for i in range(1,len(ts)):
            td_map["t"].append(ts[i])
            td_map["v"].append(vs[i])
            if(vs[i] <= td_thresh):
                td_map["dur"].append(td_map["dur"][-1])
            else:
                td_map["dur"].append(td_map["dur"][-1] + self.config["ramp_sampling_time"])

            vdiff = vs[i] - vs[i-1]
            #first iteration only
            if(last_vdiff is None): last_vdiff = vdiff

            state_change = (np.sign(vdiff) != np.sign(last_vdiff)) and (np.abs(vdiff) > state_change_thresh) #if state changes, then this will be true
            if(ramping and (state_change == False)):
                ramps[-1]["t"].append(ts[i-1])
                ramps[-1]["v"].append(vs[i-1])
            elif((ramping == False) and (state_change == False) and (vdiff == 0)):
                flat_tops[-1]["t"].append(ts[i-1])
                flat_tops[-1]["v"].append(vs[i-1])
            elif(ramping and state_change):
                #what kind of state change is it? Are we going from ramp to a new ramp some time later?
                #or are we going from a ramp to a flat top? 
                if(vdiff < 0):
                    #we are starting a new ramp
                    #add the last value to the last ramp and append a fresh element to the list
                    ramps[-1]["t"].append(ts[i-1])
                    ramps[-1]["v"].append(vs[i-1])
                    ramps.append({"t":[], "v":[]})
                elif(vdiff == 0):
                    #we are starting a flat top
                    #add this value to the most recent flat top element and change the state flag
                    ramping = False
                    flat_tops[-1]["t"].append(ts[i-1])
                    flat_tops[-1]["v"].append(vs[i-1])
            elif((ramping == False) and state_change):
                #we are going from flat top to a new ramp.
                #add the last datapoint to the flat top list and make a fresh
                #flat top element and ramp element, then change the ramping state flag
                ramping = True
                flat_tops[-1]["t"].append(ts[i-1])
                flat_tops[-1]["v"].append(vs[i-1])
                flat_tops.append({"t":[], "v":[]})
                ramps.append({"t":[], "v":[]})
            else:
                print("There is a case in the ramp separation analysis that wasnt considered:")
                print("Ramping: " + str(ramping))
                print("State change: " + str(state_change))
                print("vdiff: " + str(vdiff))
                print("last vdiff: " + str(last_vdiff))
            
            last_vdiff = vdiff
        
        self.ramps = ramps
        self.flat_tops = flat_tops #saved for later.
        self.time_duration_map = td_map


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

        if(N == None):
            #ch_df is usually much smaller, so I will loop through
            #each charge event and remove all light events that fall within
            #the timeframe. 
            for i, row in ch_df.iterrows():
                print(i)
                t0 = row["ch3 seconds"]
                keep_mask = (np.abs(light_df["ch0 seconds"] - t0) > T) 
                light_df = light_df[keep_mask]
            output_light_df = light_df
        else:
            N = int(N)
            chosen_indices = []
            N_good_events = 0
            output_light_df = pd.DataFrame()
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







    


    

