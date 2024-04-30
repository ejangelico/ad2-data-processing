
import os
import yaml
import datetime 
import numpy as np 
import matplotlib.pyplot as plt 
import glob 
import pickle
import pandas as pd
from scipy.ndimage import gaussian_filter
import scipy.integrate 
import utilities
import scipy.signal
import scipy.odr as odr
import time
import sys
from scipy.interpolate import interp1d
plt.style.use("~/evanstyle.mplstyle") #replace with your own style sheet. 



#This is a dataset class that acts on a combined Struck and AD2 dataset. It is a bit more detailed,
#in that it has all the information necessary to do data reduction, parses pre-reduced data from
#both DAQ system, has the ability to do some time synchronization, looks at ramp files, looks at 
#g_event files, etc. 
class Dataset:
    #config file is a yaml file that has most information. for example see
    #<path>/ad2-data-processing/configs/run5_config.yaml. 
    #topdir is the directory of the dataset, so "<path>/<to>/ds02/"
    def __init__(self, topdir, config):
        self.config_file = config
        self.topdir = topdir
        self.config = None 
        self.load_config() #now config is a dict form of that yaml file. 

        #load files associated with applied high voltage
        self.ramp_data = pd.DataFrame() #ramp data flat and linear, not separated into chunks associated with ramps and flat tops
        self.g_event_data = pd.DataFrame()
        self.time_duration_map = {"t":[], "dur":[], "v":[]} # a 1:1 mapping between timestamps (unix epoch) and duration above 0V to cut out down-time from time plots. 
        self.load_hv_textfiles() #parses the files into ramp_data and g_event_data
        self.correct_hv_data()
        self.create_time_duration_map()

        #don't load, but prep file lists corresponding to each DAQ system.
        self.ad2_files = glob.glob(self.topdir+"prereduced*.p")
        self.struck_files = sorted(glob.glob(self.topdir+self.config["struck_folder"]+"prereduced*.p")) #sorted because there are many, with an index at the end of filename. 
        self.struck_timebounds = {} #for each struck pickle file, store the first event timestamp and last event timestamp for quick searching through files. 
        if(len(self.ad2_files) == 0):
            print("Problem! Found no AD2 files with prefix 'prereduced*.p' in {}".format(self.topdir))
        if(len(self.struck_files) == 0):
            print("Problem! Found no Struck data files with prefix 'prereduced*.p' in {}".format(self.topdir+self.config["struck_folder"]))

        #finally, we will store all reduced data from waveform analysis into one DF
        self.columns = []
        self.ad2_chmap = {} #indexed by software channel, gives the index of this channel within the "Data" list in prereduced DF row.  
        for ad2 in self.config["ad2_reduction"]:
            for i, sw_ch in enumerate(self.config["ad2_reduction"][ad2]["channel_map"]["software_channel"]):
                #for saving these software channels for easier access
                self.ad2_chmap[sw_ch] = self.config["ad2_reduction"][ad2]["channel_map"]["prereduced_index"][i]
                self.columns.append("ch{:d} amp".format(sw_ch))
                self.columns.append("ch{:d} full integral".format(sw_ch))
                self.columns.append("ch{:d} baseline".format(sw_ch))
                self.columns.append("ch{:d} postbaseline".format(sw_ch)) #baseline calculated from the back of the waveform
                self.columns.append("ch{:d} noise".format(sw_ch))
                self.columns.append("ch{:d} min".format(sw_ch))
                self.columns.append("ch{:d} max".format(sw_ch))
                self.columns.append("ch{:d} absmax".format(sw_ch))
                self.columns.append("ch{:d} min time".format(sw_ch))
                self.columns.append("ch{:d} max time".format(sw_ch))
                self.columns.append("ch{:d} absmax time".format(sw_ch))
                self.columns.append("ch{:d} seconds".format(sw_ch))
                self.columns.append("ch{:d} nanoseconds".format(sw_ch))
                self.columns.append("ch{:d} n negpeaks".format(sw_ch)) #integer number of TRULY negative polar pulses in the raw waveform
                self.columns.append("ch{:d} n pospeaks".format(sw_ch)) #positive polar (i.e. charge going from cathode to gnd (not anode))
                self.columns.append("ch{:d} multipulse times".format(sw_ch)) #times of the multipulse events, for looking at discharge frequency
                self.columns.append("ch{:d} hv".format(sw_ch)) #HV in kV
                self.columns.append("ch{:d} field".format(sw_ch)) #field in kV/cm
                #more specialized
                self.columns.append("ch{:d} charge".format(sw_ch)) #converting amplitude or otherwise into pC 
                self.columns.append("ch{:d} energy".format(sw_ch)) #nJ, assuming charge at a particular voltage, energy fraction of capacitor

                #for identifying events with waveforms if you want to re-reference waveform files
                self.columns.append("ch{:d} filename".format(sw_ch))
                self.columns.append("ch{:d} evidx".format(sw_ch)) #index in the dataframe stored in that file

        #TODO: please add a baseline subtracted version in addition to
        #non-baseline subtracted version of the integrals, because I'm 
        #presently hard coding integration times in order to baseline subtract
        #integrals... 
        self.struck_chmap = {} #indexed by software channel, gives the index of this channel within the "Data" list in prereduced DF row. 
        for i, sw_ch in enumerate(self.config["struck_reduction"]["channel_map"]["software_channel"]):
            self.struck_chmap[sw_ch] = self.config["struck_reduction"]["channel_map"]["prereduced_index"][i] #index of the channel in the list of prereduced data
            self.columns.append("ch{:d} amp".format(sw_ch))
            self.columns.append("ch{:d} afterpulse integral".format(sw_ch))
            self.columns.append("ch{:d} trigger integral".format(sw_ch))
            self.columns.append("ch{:d} baseline".format(sw_ch))
            self.columns.append("ch{:d} postbaseline".format(sw_ch)) #baseline calculated from the back of the waveform
            self.columns.append("ch{:d} noise".format(sw_ch))
            self.columns.append("ch{:d} min".format(sw_ch))
            self.columns.append("ch{:d} max".format(sw_ch))
            self.columns.append("ch{:d} min time".format(sw_ch))
            self.columns.append("ch{:d} max time".format(sw_ch))
            self.columns.append("ch{:d} seconds".format(sw_ch))
            self.columns.append("ch{:d} nanoseconds".format(sw_ch))
            self.columns.append("ch{:d} hv".format(sw_ch)) #HV in kV
            self.columns.append("ch{:d} field".format(sw_ch)) #field in kV/cm

            #for identifying events with waveforms if you want to re-reference waveform files
            self.columns.append("ch{:d} filename".format(sw_ch))
            self.columns.append("ch{:d} evidx".format(sw_ch)) #index in the dataframe stored in that file


             

        self.reduced_df = None #is created and set in self.reduce_data()
        

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

        if(os.path.isfile(self.topdir+"dac_conversion.txt")):
            temp = open(self.topdir+"dac_conversion.txt", "r")
            l = temp.readlines()[0]
            dac_conv = float(l)
        else:
            dac_conv = 4 #use the 40 kV glassman value. 
            
        if(os.path.isfile(self.topdir+self.config["ramp_name"])):
            #load the rampfile data
            d = np.genfromtxt(self.topdir+self.config["ramp_name"], delimiter=',', dtype=float)
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
            temp_dict["v_mon"] = v_mon 
            temp_dict["dac_conv"] = dac_conv #used to identify which supply is being used, for calibrating the actual voltage. 

            #add to the ramps dataframe
            self.ramp_data = pd.concat([self.ramp_data, pd.DataFrame(temp_dict)], axis=0, ignore_index=True)


        #load the g_events data, if it exists
        if(os.path.isfile(self.topdir+self.config["g_events_name"])):
            d = np.genfromtxt(self.topdir+self.config["g_events_name"], delimiter=',', dtype=float)
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
                temp_dict["dac_conv"] = dac_conv #used to identify which supply is being used, for calibrating the actual voltage. 

                self.g_event_data = pd.concat([self.g_event_data, pd.DataFrame(temp_dict)], axis=0, ignore_index=True)

        #sort both dataframes by time
        if(len(self.g_event_data.index) != 0):
            self.g_event_data = self.g_event_data.sort_values("t")
        if(len(self.ramp_data.index) != 0):
            self.ramp_data = self.ramp_data.sort_values("t")


    #for a given applied voltage and supply, indicated by
    #the dac_conversion factor, what is the actual voltage applied
    #to the cathode? This is not totally complete yet, as we are still 
    #calibrating! But whatever information we do have, is input in this function. 
    #By default returns uncalibrated data. 
    def supply_calibration(self, dac_conv, v_app):
        pth = self.config["calib_path"] #name of the calibrations folder
        cpth = self.topdir + "../../" + pth #calibration path
        cv = v_app #corrected voltage

        if(dac_conv == 4.0):
            #40 kV glassman supply
            if(os.path.isfile(cpth + "40kV_glassman_calibration.p")):
                #monitor correction, a number close to 1.0, normalized by v_app
                s = pickle.load(open(cpth + "40kV_glassman_calibration.p", "rb"))[0]
                
                #at a certain point, the DAC cant output large enough voltages due to current limit
                if(v_app > 18.98):
                    cv = s(18.98/dac_conv)*18.98
                else:
                    cv = s(v_app/dac_conv)*v_app
                    
        #leakage current solely from the bleeder resistors and resistances. 
        #no capacitance leakage current included nor xenon leakage current. 
        leakage_factor = 0.9967 
        cv = leakage_factor*cv

        return cv

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

        if(len(self.g_event_data.index) == 0):
            self.g_event_data = pd.DataFrame(columns=["v_app", "v_mon", "dac_conv", "t"])

        ts_g = np.array(self.g_event_data["t"])

        #the algorithm heavily uses the v_applied data
        #because it is smooth and has no noise. Later,
        #once the time series is corrected, the v_mon
        #stream will be used to calibrate the HV applied.
        vs_a = np.array(self.ramp_data["v_app"])
        vs_m = np.array(self.ramp_data["v_mon"])
        vs_ga = np.array(self.g_event_data["v_app"])
        vs_gm = np.array(self.g_event_data["v_mon"])
        dac = np.array(self.ramp_data["dac_conv"])
        dac_g = np.array(self.g_event_data["dac_conv"])

        #calibrate the applied voltages
        vs_cal = [self.supply_calibration(dac[i], vs_a[i]) for i in range(len(dac))]
        
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
        new_vs_cal = [] #calibrated voltages
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
            new_vs_cal.append(self.supply_calibration(dac_g[i], vs_ga[i]))
            #add a point that is fin_dt away from the g-event reset time
            #and has a voltage equal to the next starting voltage
            #(far in the future)
            new_ts.append(gt + fine_dt)
            new_vs_a.append(vs_a[idx+1])
            new_vs_m.append(vs_m[idx+1])
            new_vs_cal.append(self.supply_calibration(dac[idx+1], vs_a[idx+1]))
        


        #add the lists back to the time series and resort
        ts_a = np.append(ts_a, new_ts)
        vs_a = np.append(vs_a, new_vs_a)
        vs_m = np.append(vs_m, new_vs_m) #artificially adding a perfect value to the monitor voltage. 
        vs_cal = np.append(vs_cal, new_vs_cal)
        #sort the lists by ts_a simultaneous
        idx = np.argsort(ts_a)
        ts_a = ts_a[idx]
        vs_a = vs_a[idx]
        vs_m = vs_m[idx]
        vs_cal = vs_cal[idx]

        #next, with the g-events accounted for, we will re-parse the
        #voltage-time stream looking for any gaps in time. Correct
        #those gaps with an fine_dt step with the next voltage value. 
        new_ts = []
        new_vs_a = []
        new_vs_m = []
        new_vs_cal = []
        for i in range(len(ts_a) - 1):
            if(ts_a[i+1] - ts_a[i] > reset_thresh):
                #add a point that is fine_dt away from the reset time
                #and has a voltage equal to the next starting voltage
                #(far in the future)
                new_ts.append(ts_a[i] + fine_dt)
                new_vs_a.append(vs_a[i+1])
                new_vs_m.append(vs_m[i+1])
                new_vs_cal.append(vs_cal[i+1])

        #repeat the sorting process

        #add the lists back to the time series and resort
        ts_a = np.append(ts_a, new_ts)
        vs_a = np.append(vs_a, new_vs_a)
        vs_m = np.append(vs_m, new_vs_m) #artificially adding a perfect value to the monitor voltage. 
        vs_cal = np.append(vs_cal, new_vs_cal)
        #sort the lists by ts_a simultaneous
        idx = np.argsort(ts_a)
        ts_a = ts_a[idx]
        vs_a = vs_a[idx]
        vs_m = vs_m[idx]
        vs_cal = vs_cal[idx]


        self.ramp_data = pd.DataFrame()
        self.ramp_data["t"] = ts_a
        self.ramp_data["v_app"] = vs_a
        self.ramp_data["v_mon"] = vs_m
        self.ramp_data["v_cal"] = vs_cal

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
        vs_r = np.array(self.ramp_data["v_cal"])
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
        vs = np.array(self.ramp_data["v_cal"])[idx_l:idx_r]
        if(np.min(ts) > t or np.max(ts) < t):
            #if this time is somehow still not in the range of the data
            return None

        #at this point, it is possible that the time range is days long,
        #making the algorithm completely rediculously long. So we linearly
        #interpolate the raw data to get a TRUE window of 1 second on either
        #side to then feed to the exponential filter. 
        s_raw = interp1d(ts, vs) #interpolate raw data over a 10 to many second window
        tb = 1 #seconds
        #reduce the range of the fine-scale interpolation function
        if(t - tb > np.min(ts)):
            tb_b[0] = t - tb
        else:
            tb_b[0] = np.min(ts)
        if(t + tb < np.max(ts)):
            tb_b[1] = t + tb
        else:
            tb_b[1] = np.max(ts)

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
    


    def get_field_for_hv(self, kv):
        if(kv == None):
            return None
        return kv/self.config["rog_gap"] #kV/cm

    #finding a sublist
    #between two time brounds (chronologically ordered) with list 
    #chronological as well. 
    def find_sublist_within_bounds(self, lst, t0, t1):
        lst = np.array(lst)
        mask = np.where((lst >= t0) & (lst <= t1))
        return mask, lst[mask]
    
    #loads each struck pickle file and save their
    #timebounds for quick searching later. 
    def reload_struck_timebounds(self):
        self.struck_timebounds = {} #reset the stored dict
        for f in self.struck_files:
            df, date = pickle.load(open(f, "rb"))
            #just in case, make sure sorted by timestamp
            df = df.sort_values(by="Seconds")
            tb = [df["Seconds"].iloc[0], df["Seconds"].iloc[-1]]
            self.struck_timebounds[f] = tb
        
        return self.struck_timebounds




    #function to find PMT events within the vacinity
    #of a particular timestamp, with time vacinity halfT
    #(-halfT, t0, halfT). Timestamp t0 is seconds since epoch. 
    #if more precision is needed, t0_nsec is optional. 
    def find_light_in_vacinity(self, t0, t0_ns, halfT, halfT_ns):
        if(self.struck_timebounds == {}):
            self.reload_struck_timebounds() #reload the timebounds of struck files for quick searching. 

        #if t0_nsec is not none, then we need to treat halfT with 
        #extra precision. So we opt to have all nsec precision in the
        #search algorithm, but set those numbers to 0 if these are none


        #We have three times of interest, t0, t0 - halfT, and t0 + halfT. 
        #These could bridge files. There is a rule for this function that
        #the halfT can't be larger than a typical file time bound, 100 seconds. 
        #Please write a different function for that. So I opt for an algorithm that finds
        #the files that contain all three times, which could be 1 or 2 filenames given that
        #rule above.
        tcutoff = 100 #seconds
        if(halfT > tcutoff/2):
            print("Your halfT is {:.2f} seconds, and that is too long based on typical struck acquisition times of 100 seconds. Please use a different function.")
            return None
        
        three_times = [t0 - halfT, t0, t0 + halfT] #second resolution on file bounds
        files_containing_times = []
        for f in self.struck_timebounds:
            tb = self.struck_timebounds[f]
            for t in three_times:
                if(tb[0] < t < tb[1]):
                    files_containing_times.append(f)

        #reduce any repeated filenames
        files_containing_times = list(set(files_containing_times))
        desired_df = pd.DataFrame()
        for f in files_containing_times:
            #at this point, we can just make a dataframe mask for each file
            #to select events within the bounds. 
            df, date = pickle.load(open(f, "rb"))

            #this mask logic is tricky because (1) we want nanosecond precision and (2) to get that,
            #we need to subtract the epoch off of seconds (in order to not overflow the bit-memory of floats)
            #and then also turn it into a decimal float with nanosecond precision in units of seconds. 
            mask = ((df["Seconds"] - (t0 - halfT) + ((df["Nanoseconds"] - (t0_ns - halfT_ns))/1e9)) >= 0) &\
                    ((df["Seconds"] - (t0 + halfT) + ((df["Nanoseconds"] - (t0_ns + halfT_ns))/1e9)) <= 0)
            evs = df[mask] 
            desired_df = pd.concat([desired_df, evs], ignore_index=True)

        desired_df = desired_df.sort_values(by="Seconds")
        return desired_df
    
    #event is assumed to be a row of the struck dataframe
    def plot_light(self, event, ax = None, ax_in = None):
        dT = 1e6/float(self.config["struck_reduction"]["clock"])
        times = np.arange(0, len(event["Data"][0])*dT, dT) #times in microseconds

        if(ax is None):
            fig, ax = plt.subplots()

        for ch in range(len(event["Data"])):
            ax.plot(times, event["Data"][ch], linewidth=0.8)

            if(ax_in != None):
                ax_in.plot(times, event["Data"][ch], linewidth=0.8)
        
        return ax
    
    def plot_charge(self, event, ax = None):
        dT = event["dT"]
        if(ax == None):
            fig, ax = plt.subplots()
        for sw_ch in self.ad2_chmap:
            prered_ind = self.ad2_chmap[sw_ch]
            v = event["Data"][prered_ind]
            times = np.array(np.linspace(0, len(v)*dT*1e6, len(v))) #times in microseconds
            ax.plot(times, v, linewidth=0.8, label="ch{:d}".format(sw_ch))
        
        ax.set_xlabel("time [us]")
        ax.set_ylabel("amplitude [mV]")

        return ax
    


    #for an amplitude or voltage in the charge detection
    #circuit, there is a conversion to charge deposited 
    #(it's an integrating circuit). This is where you can
    #change the calibration code if its more complicated than 
    #linear. In this case, it is, because a diode is in parallel
    #for protection, and it starts to conduct and affect the charge
    #reconstructed at high amplitudes. Returns in pC
    def get_charge_from_mv(self, mv):
        #starts to deviate if greater than 140 mV. 
        if(mv < 140):
            return mv/2.6 #linear, 2.6 mV/pC
        if(mv >= 140):
            #for the moment, simply letting it stay linear. no events have been observed this large
            return mv/2.6
    



    

    ###########################################################################
    #                   Data reduction related code                            #
    ###########################################################################


    #Admittedly, this is a complicated set of functions that could be simplified. 
    #This does the following:
    #1) Gaussian smooths the raw waveform 
    #2) Finds the peak time and amplitude of the smoothed waveform
    #3) Corrects the time and amplitude by looking at the raw waveform
    #in a window near smoothed waveform local maxima
    #4) Interprets each peak as being a charge deposition by looking at the
    #baseline near where each peak occurs, corrects based on any previous pulse
    #where there may still be a falling tail from the 400us integrating circuit
    #5) Reports remaining peaks and checks that they are above threshold
    #all of this is done in a polarity agnostic way 
    def analyze_for_multipeaks_charge(self, row, sw_ch=None, debug=False):
        output = {} #keys are sw channels, but elements are dictionaries with extracted information

        dT = row["dT"]
        #mV, threshold for decided one analysis vs another based on low amplitude. 
        amp_thr_sig = self.config["ad2_reduction"]["glitch"]["fit_amplitude_threshold"] #sigma threshold
        #see long comment in the analyze_charge_waveforms function for more info
        bl_wind_frac = np.array(self.config["ad2_reduction"]["glitch"]["baseline_window"]) #fraction of buffer about 0
        #exponential decay of integrator circuit
        exp_tau = self.config["ad2_reduction"]["glitch"]["exponential_tau"] #microseconds
        #minimum spacing between peaks for rejecting noise
        t_space = 50 #us, this is driven by what has been observed in waveform data as the space between multiple charge depositions
        #gaussian smoothing time for finding peaks
        t_smooth = self.config["ad2_reduction"]["glitch"]["filter_tau"] #microseconds
        #when looking backwards at raw data to find a corrected peak time
        #relative to the gaussian smoothed shifted time, this window is used
        peak_window = int(100/(dT*1e6)) #us converted to samples. 

        #a switch in case you only want to process one channel or all
        if(sw_ch is None):
            chs = self.ad2_chmap
        else:
            chs = [sw_ch]

        for sw_ch in chs:
            prered_ind = self.ad2_chmap[sw_ch]
            #get waveform np arrays and time np arrays. 
            v = np.array(row["Data"][prered_ind])
            N_samples = len(v)
            bl_wind = N_samples*0.5*bl_wind_frac + N_samples/2
            bl_wind = bl_wind.astype(int)

            v = v - np.mean(v[bl_wind[0]:bl_wind[1]]) #baseline subtract
            ts = np.array(np.linspace(0, len(v)*dT*1e6, len(v))) #times in microseconds
            if(t_smooth != 0):
                v_sm = gaussian_filter(v, t_smooth/(dT*1e6))
            else: 
                v_sm = v

            raw_noise = np.std(v[bl_wind[0]:bl_wind[1]])
            amp_thr = amp_thr_sig*raw_noise
            width_thresh = 0.8*int(np.log10(2)*exp_tau/(dT*1e6)) #samples, width of peak to be considered a peak. a bit less than the half-height width of exponential

            #finding positive and negative polarity peaks separately. 
            peak_dist = np.max([int(t_space/(dT*1e6)), 1]) #minimum distance between peaks
            pospeaks, _ = scipy.signal.find_peaks(v_sm, distance=peak_dist, height=amp_thr, width=width_thresh)
            negpeaks, _ = scipy.signal.find_peaks(-1*v_sm, distance=peak_dist, height=amp_thr, width=width_thresh)

            output["ch{:d} n pospeaks".format(sw_ch)] = len(pospeaks)
            output["ch{:d} n negpeaks".format(sw_ch)] = len(negpeaks) 
            npeaks = len(negpeaks) + len(pospeaks)
            #If there are any peaks found do the following:
            #1) find the peak time and amplitude for each peak, adjusting for the gaussian filter's time smearing
            #2) correct the amplitude for the falling tail of the exponential
            #3) remove any peaks that are below the amplitude threshold
            if(npeaks >= 1):
                #peak times in the gaussian filtered wave are always
                #going to be a bit after the true peak. Define a window of X
                #microseconds prior to the gaussian filtered peak time that looks
                #for the max sample. 
                
                adjusted_peak_times = []
                peak_amplitudes = []
                polarities = [] #only needed when doing peak_amplitude correction based on falling edge of exponentials
                #loop through each peak and find the amplitude within a window
                for pidx in pospeaks:
                    #make sure the window doesn't have an indexing problem on the waveform list
                    if(pidx - peak_window < 0):
                        tmp_start = 0
                    else:
                        tmp_start = pidx - peak_window
                    
                    #find the maximum value in the window leading from tmp_start to pidx and its
                    #time in microseconds, append that to the lists
                    tmp_max = np.max(v[tmp_start:pidx])
                    tmp_max_idx = np.argmax(v[tmp_start:pidx])
                    tmp_max_time = ts[tmp_start + tmp_max_idx]
                    adjusted_peak_times.append(tmp_max_time)
                    peak_amplitudes.append(tmp_max)
                    polarities.append(1)

                for pidx in negpeaks:
                    #make sure the window doesn't have an indexing problem on the waveform list
                    if(pidx - peak_window < 0):
                        tmp_start = 0
                    else:
                        tmp_start = pidx - peak_window
                    
                    #find the minimum value in the window leading from tmp_start to pidx and its
                    #time in microseconds, append that to the lists
                    tmp_min = np.min(v[tmp_start:pidx])
                    tmp_min_idx = np.argmin(v[tmp_start:pidx])
                    tmp_min_time = ts[tmp_start + tmp_min_idx]
                    adjusted_peak_times.append(tmp_min_time)
                    peak_amplitudes.append(tmp_min)
                    polarities.append(-1)
                
                #there is configuration bit in the run9_config that triggers
                #a computationally expensive fitting mode or a less computationally
                #expensive amplitude correction mode. This is needed when there
                #are multiple pulses, or issues with the peak finder where it finds
                #small peaks on the baseline. BECAUSE if one just adds the peak
                #amplitudes, and there are a few mistakenly found peaks along the
                #falling tail of a large exponential, one adds charge equal to the
                #voltage on that tail at that moment. One wants to correct for the fact
                #that you're adding a bit of charge to an integrator that is discharging,
                #so one needs to correct for the pseudo-baseline at the moment of the additional
                #pulse. 
                if(self.config["ad2_reduction"]["glitch"]["fit_mode"] == True):
                    pass #write a fit that is a sum of exponentials using the peak times as the guess
                else:
                    #this is the correction mode that just looks n microseconds prior to the peak
                    #to find the baseline, and subtracts that from the peak amplitude.
                    for i in range(len(adjusted_peak_times)):
                        peak_idx = int(adjusted_peak_times[i]/(dT*1e6))
                        if(peak_idx - peak_window < 0):
                            tmp_start = 0  
                        else:
                            tmp_start = peak_idx - peak_window
                        sub_baseline = v[tmp_start] #single value for the baseline before this peak. 
                        peak_amplitudes[i] = peak_amplitudes[i] - sub_baseline
                
                #now that the peak amplitudes are corrected for a falling tail, 
                #remove any counts for negative or positive peaks that are below
                #the amplitude threshold, as the peak finder in scipy is susceptible
                #to noise below threshold if its on the falling tail of a big pulse. 
                readjusted_peak_times = []
                readjusted_peak_amplitudes = []
                for i, pa in enumerate(peak_amplitudes):
                    if(polarities[i]*pa >= amp_thr):
                        readjusted_peak_times.append(adjusted_peak_times[i])
                        readjusted_peak_amplitudes.append(pa)
                    else:
                        if(polarities[i] > 0 and output["ch{:d} n pospeaks".format(sw_ch)] > 0):
                            output["ch{:d} n pospeaks".format(sw_ch)] -= 1
                        if(polarities[i] < 0 and output["ch{:d} n negpeaks".format(sw_ch)] > 0):
                            output["ch{:d} n negpeaks".format(sw_ch)] -= 1

                output["ch{:d} peaktimes".format(sw_ch)] = readjusted_peak_times
                output["ch{:d} peakamps".format(sw_ch)] = readjusted_peak_amplitudes

            #otherwise, there were no peaks found and the waveform is just barely
            #at the threshold such that a gaussian smoothing makes any sample passing
            #threshold go away. 
            else:
                output["ch{:d} peaktimes".format(sw_ch)] = []
                output["ch{:d} peakamps".format(sw_ch)] = []
                readjusted_peak_times = []
                readjusted_peak_amplitudes = [] #just for debugging plot. 


            if(debug):
                fig, ax = plt.subplots()
                ax.plot(ts, v)
                ax.plot(ts, v_sm)
                ax.plot(ts[pospeaks], v_sm[pospeaks], 'bo')
                ax.plot(ts[negpeaks], v_sm[negpeaks], 'go')
                ax.plot(readjusted_peak_times, readjusted_peak_amplitudes, 'ko')
                ax.set_title("negpeaks: {:d}, pospeaks: {:d}".format(output["ch{:d} n negpeaks".format(sw_ch)], output["ch{:d} n pospeaks".format(sw_ch)]))
                plt.show()

        return output



    #perform fit based analysis on charge channels.
    #There are a few conditions where the function decides
    #that fitting is not appropriate, so it returns strings
    #that flag why, so that the reduction function can triage
    #and send it through a different method.   
    def analyze_charge_waveforms(self, row, debug=False):
        output = {} #keys are sw channels, but elements are dictionaries with extracted information

        dT = row["dT"]
        #mV, threshold for decided one analysis vs another based on low amplitude. 
        amp_thr_sig = self.config["ad2_reduction"]["glitch"]["fit_amplitude_threshold"] #n*sigma of noise

        #I've used a number of units for programmable baseline window and integration window. 
        #Because there are some runs or datasets where the charge buffer is not constant, I've normalized
        #to the total buffer length. So baseline window is the fraction of the buffer about 0. So it baseline
        #being from [0, 400] in samples in a 1000 sample acquisition will be [-1, -0.4]. Note that at the moment
        #the integration window is really only useful for very small glitches, as the amplitude of a general
        #pulse above threshold is calculated by an exponential fit. 
        bl_wind_frac = np.array(self.config["ad2_reduction"]["glitch"]["baseline_window"]) #fraction of buffer about 0
        int_wind_frac = np.array(self.config["ad2_reduction"]["glitch"]["integration_window"]) 

        
        rog_cap = float(self.config["rog_cap"]) #pF
        for sw_ch in self.ad2_chmap:
            prered_ind = self.ad2_chmap[sw_ch]
            #get waveform np arrays and time np arrays. 
            v = np.array(row["Data"][prered_ind])
            ts = np.array(np.linspace(0, len(v)*dT*1e6, len(v))) #times in microseconds
            N_samples = len(v)
            bl_wind = N_samples*0.5*bl_wind_frac + N_samples/2 #centered about 0
            int_wind = N_samples*0.5*int_wind_frac + N_samples/2
            bl_wind = bl_wind.astype(int)
            int_wind = int_wind.astype(int)
            #noise
            output["ch{:d} noise".format(sw_ch)] = np.std(v[bl_wind[0]:bl_wind[1]]) #mV
            amp_thr = amp_thr_sig*output["ch{:d} noise".format(sw_ch)] #mV, threshold for deciding one analysis vs another based on low amplitude.

            #calculate baselines for use
            output["ch{:d} baseline".format(sw_ch)] = np.mean(v[bl_wind[0]:bl_wind[1]])
            output["ch{:d} postbaseline".format(sw_ch)] = np.mean(v[-1*bl_wind[1]:])

            #change the local waveform variable
            v = v - output["ch{:d} baseline".format(sw_ch)]

            #put max and min samples of the waveform into the output
            #and the time at which they occurred
            output["ch{:d} min".format(sw_ch)] = np.min(v)
            output["ch{:d} max".format(sw_ch)] = np.max(v)
            output["ch{:d} min time".format(sw_ch)] = dT*np.argmin(v)*1e6 #microseconds
            output["ch{:d} max time".format(sw_ch)] = dT*np.argmax(v)*1e6

            #find the abs-max sample and the sign of that sample. 
            absmax_index = np.argmax(np.abs(v))
            output["ch{:d} absmax time".format(sw_ch)] = dT*absmax_index*1e6
            absmax_t = dT*absmax_index*1e6 #local stored variable, a little quicker to type

            #this next trig_idx variable represents what the analysis algorithm is going to treat a
            trig_idx = absmax_index
            absmax_val = v[absmax_index] #independent of polarity
            
            #count the number of samples in the entire waveform that are above the amplitude threshold.
            #If one just looks at events with absmax above or equal to thresh, you get
            #tons of events that are clearly noise but with 1 sample above.
            n_samples_above_thr = len(np.where(np.abs(v) >= amp_thr)[0])
            n_thresh = 3 #number of samples above threshold to trigger
            v_polysub = [] #delete if not using debug feature

            if(n_samples_above_thr >= n_thresh):
                #how many pulses are there in this waveform?
                n_pulses_dict = self.analyze_for_multipeaks_charge(row, sw_ch=sw_ch, debug=debug)
                output["ch{:d} n pospeaks".format(sw_ch)] = n_pulses_dict["ch{:d} n pospeaks".format(sw_ch)]
                output["ch{:d} n negpeaks".format(sw_ch)] = n_pulses_dict["ch{:d} n negpeaks".format(sw_ch)]
                #if greater than 1, do a multi-pulse analysis 
                if(n_pulses_dict["ch{:d} n pospeaks".format(sw_ch)] >= 1 or n_pulses_dict["ch{:d} n negpeaks".format(sw_ch)] >= 1):
                    #then this dict also contains the amplitude reconstruction, sum of
                    #amplitudes of all pulses in the waveform
                    output["ch{:d} amp".format(sw_ch)] = np.sum(n_pulses_dict["ch{:d} peakamps".format(sw_ch)])
                    output["ch{:d} multipulse times".format(sw_ch)] = n_pulses_dict["ch{:d} peaktimes".format(sw_ch)]

                #Otherwise, the peak finder found nothing, which can happen for a few reasons:
                #(1) There is a peak, but it is pretty small and gaussian smoothing washes it out. 
                #(2) there is some low frequency baseline wander so that a number of samples are
                #above threshold (because baseline is calculated from beginning of waveform) but
                #there is no pulse, so no local maxima is found by peak finder. This is the worst case,
                #as it can sometimes be called "2 mV" amplitude by using the absmax_val. 
                else:
                    #a distinguishing feature of case 2 that is NOT in case 1 is that the early time baseline
                    #and post baseline will differ by an amount greater than the threshold. Check for this. 
                    bl_diff = np.abs(output["ch{:d} baseline".format(sw_ch)] - output["ch{:d} postbaseline".format(sw_ch)])
                    if(bl_diff > amp_thr):
                        #then this is case 2. Fit a second order polynomial and subtract it
                        p1 = np.polyfit(ts, v, 2)
                        v_polysub = v - np.polyval(p1, ts)
                        #now that the waveform is polynomial subtracted, find the absmax again
                        output["ch{:d} amp".format(sw_ch)] = np.max(np.abs(v_polysub))
                        output["ch{:d} n pospeaks".format(sw_ch)] = 0
                        output["ch{:d} n negpeaks".format(sw_ch)] = 0

                    #otherwise, this is case 1. Just use the absmax_val
                    else:
                        output["ch{:d} amp".format(sw_ch)] = absmax_val
                        if(absmax_val > 0):
                            output["ch{:d} n pospeaks".format(sw_ch)] = 1
                            output["ch{:d} n negpeaks".format(sw_ch)] = 0
                        else:  
                            output["ch{:d} n pospeaks".format(sw_ch)] = 0
                            output["ch{:d} n negpeaks".format(sw_ch)] = 1

                    """
                    #get a chunk of data after the trigger time so as 
                    #to fit an exponential with a start time t0. 
                    fit_time_buffer = int(10/(dT*1e6)) #samples, time after the trigger to which to include the fit
                    ts_trim = ts[trig_idx+fit_time_buffer:]
                    v_trim = v[trig_idx+fit_time_buffer:]
                    initial_guess = [absmax_val, 400] #400 is the estimated tau from previous explorative fits. 

                    #ODR based objects for performing fit
                    odr_model = odr.Model(utilities.exp_wrapper_fixedt0(absmax_t))
                    odr_data = odr.Data(ts_trim, v_trim)
                    odr_inst = odr.ODR(odr_data, odr_model, beta0=initial_guess)
                    odr_result = odr_inst.run()
                    amp = odr_result.beta[0]
                    tau = odr_result.beta[1]
                    output["ch{:d} amp".format(sw_ch)] = amp 
                    """
                    

            #if the event has no samples above the sample threshold for fit based analysis
            else:
                #just find amplitude as max sample. 
                output["ch{:d} amp".format(sw_ch)] = absmax_val
                output["ch{:d} n pospeaks".format(sw_ch)] = 0
                output["ch{:d} n negpeaks".format(sw_ch)] = 0

            #to confirm with plots    
            if(debug and (output["ch{:d} n negpeaks".format(sw_ch)] > 0 or output["ch{:d} n pospeaks".format(sw_ch)] > 0 or len(v_polysub) != 0)):
                fig, ax = plt.subplots()
                ax.plot(ts, v)
                ax.axhline(amp_thr, color='r', linestyle='--', alpha=0.5)
                ax.axhline(-amp_thr, color='r', linestyle='--', alpha=0.5)
                ax.axhline(output["ch{:d} amp".format(sw_ch)], color='b', linewidth=3)
                ax.axvline(ts[bl_wind[0]], color='k', alpha=0.5)
                ax.axvline(ts[bl_wind[1]], color='k', alpha=0.5)
                if(len(v_polysub) != 0):
                    print("Happened")
                    ax.plot(ts, v_polysub, color='m', linewidth=0.8)
                ax.set_title("pospeaks: {:d}, negpeaks: {:d}, baseline: {:.1f}".format(output["ch{:d} n pospeaks".format(sw_ch)], output["ch{:d} n negpeaks".format(sw_ch)], output["ch{:d} baseline".format(sw_ch)]))
                #ax.plot(ts_trim, utilities.fitfunc_exp(odr_result.beta, ts_trim, absmax_t), label="{:.2f} e^(-t/{:.2f})".format(*odr_result.beta))
                ax.set_xlabel("[us]")
                ax.set_ylabel('[mV]')
                ax.legend()
                plt.show()

            #analysis variables agnostic to multipulse structure, amplitude threshold, and otherwise.

            #integral
            output["ch{:d} full integral".format(sw_ch)] = scipy.integrate.trapezoid(v[int_wind[0]:int_wind[1]], dx=dT*1e6) #mV*us
            #what is kV applied at this time, and save time variables
            output["ch{:d} seconds".format(sw_ch)] = row["Seconds"]
            output["ch{:d} nanoseconds".format(sw_ch)] = row["Nanoseconds"]
            kv = self.get_hv_at_time(row["Seconds"] + row["Nanoseconds"]/1e9) # will auto round to 1e-6 sec
            output["ch{:d} hv".format(sw_ch)] = kv 
            output["ch{:d} field".format(sw_ch)] = self.get_field_for_hv(kv)
            #the charge can be reconstructed from the amplitude (or integral, if you calibrate that way)
            reco_charge = self.get_charge_from_mv(output["ch{:d} amp".format(sw_ch)])
            output["ch{:d} charge".format(sw_ch)] = reco_charge
            #if you assume C = 20 pF for the rogowski, then this charge has deposited
            #some quantity of energy, equal to dU = Q(dQ)/C in Joules
            if(kv != None):
                fullQ = -rog_cap*kv*1000 #V*pF = pC
                dU = reco_charge*fullQ/rog_cap/1000 #pC*pC/pF = pJ, most are on order nJ hence factor of 1000
                output["ch{:d} energy".format(sw_ch)] = dU
            else:
                output["ch{:d} energy".format(sw_ch)] = None

            output["ch{:d} evidx".format(sw_ch)] = row.name

            
                
        return output

    #TODO: please add a baseline subtracted version in addition to
    #non-baseline subtracted version of the integrals, because I'm 
    #presently hard coding integration times in order to baseline subtract
    #integrals... 
    #Note, there are a few unique aspects of the light readout that
    #are governed by physics. One is that I do NOT subtract baseline from
    #the waveform, but rather choose to do that in analysis stage because
    #the high voltage phenomena will drastically affect baseline because
    #lots of light will occur in the assumed baseline window. 
    def analyze_light_waveforms(self, row):
        output = {}
        ph_window = self.config["struck_reduction"]["pulseheight_window"] #samples
        ap_window = self.config["struck_reduction"]["afterpulse_window"] #samples 
        bl_window = self.config["struck_reduction"]["baseline_window"] #samples
        dT = 1e6/float(self.config["struck_reduction"]["clock"]) #us

        for i, sw_ch in enumerate(self.struck_chmap):
            prered_ind = self.struck_chmap[sw_ch]
            v = row["Data"][prered_ind]
            #we are intentionally not baseline subtracting the light
            #system because high voltage events will pretty significantly
            #shift the baseline. Please do it in post analysis. 

            #pulse height and basic integrals
            output["ch{:d} amp".format(sw_ch)] = np.max(v[ph_window[0]:ph_window[1]])
            output["ch{:d} afterpulse integral".format(sw_ch)] = scipy.integrate.trapezoid(v[ap_window[0]:ap_window[1]], dx=dT) #mV*us
            output["ch{:d} trigger integral".format(sw_ch)] = scipy.integrate.trapezoid(v[ph_window[0]:ph_window[1]], dx=dT) #mV*us

            #put max and min samples of the waveform into the output
            #and the time at which they occurred
            output["ch{:d} min".format(sw_ch)] = np.min(v)
            output["ch{:d} max".format(sw_ch)] = np.max(v)
            output["ch{:d} min time".format(sw_ch)] = dT*np.argmin(v) #microseconds
            output["ch{:d} max time".format(sw_ch)] = dT*np.argmax(v)

            #noise and baseline
            output["ch{:d} noise".format(sw_ch)] = np.std(v[bl_window[0]:bl_window[1]])
            output["ch{:d} baseline".format(sw_ch)] = np.mean(v[bl_window[0]:bl_window[1]])
            #baseline calculated from the back of the waveform
            output["ch{:d} postbaseline".format(sw_ch)] = np.mean(v[-1*bl_window[1]:]) 

            #hv and timing
            kv = self.get_hv_at_time(row["Seconds"] + row["Nanoseconds"]/1e9) #returns None if no ramp data 
            output["ch{:d} hv".format(sw_ch)] = kv 
            output["ch{:d} field".format(sw_ch)] = self.get_field_for_hv(kv)
            output["ch{:d} seconds".format(sw_ch)] = row["Seconds"]
            output["ch{:d} nanoseconds".format(sw_ch)] = row["Nanoseconds"]
            #for file waveform referencing
            output["ch{:d} evidx".format(sw_ch)] = row.name
    

        return output

    def reduce_data(self, charge=True, light=True):
        redict = {} #form the reduced dataframe out of a dictionary that we append things to. 
        for c in self.columns:
            redict[c] = [] #each an empty list we will append to for each event. 
        t0 = time.time()
        #struck / light component
        if(light):
            print("Reducing struck data")
            
            for count, f in enumerate(self.struck_files):
                print("On file {:d} of {:d}".format(count, len(self.struck_files) - 1), end='\r')
                df, date = pickle.load(open(f, "rb"))
                for i, row in df.iterrows():
                    output = self.analyze_light_waveforms(row)
                    #add filename to track this event
                    for sw_ch in self.struck_chmap:
                        output["ch{:d} filename".format(sw_ch)] = f

                    #loop through all of the column names in this output dic,
                    #which should match the columns in self.columns. 
                    for c in output:
                        redict[c].append(output[c])
        
        #ad2 / charge component
        if(charge):
            print("Reducing charge data")
            for count, f in enumerate(self.ad2_files):
                print("On file {:d} of {:d}".format(count, len(self.ad2_files) - 1), end='\r')
                df, date = pickle.load(open(f, "rb"))
                for i, row in df.iterrows():
                    print("On event {:d} of {:d}".format(i, len(df.index)), end='\r')
                    output = self.analyze_charge_waveforms(row, debug=False)
                    #add filename to track this event
                    for sw_ch in self.ad2_chmap:
                        output["ch{:d} filename".format(sw_ch)] = f

                    #loop through all of the column names in this output dic,
                    #which should match the columns in self.columns. 
                    for c in output:
                        redict[c].append(output[c])


        #finally append to the output dataframe
        #this orient='index' fills empty columns with NaNs
        #but sets the "keys" of the dict to rows. Transpose at the end
        #puts keys of the dict as columns
        self.reduced_df = pd.DataFrame.from_dict(redict, orient='index').T
        tf = time.time()
        print("Reduction took {:f} seconds".format(tf - t0))
