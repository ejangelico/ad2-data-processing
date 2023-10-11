
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
        self.ramp_data = {} #ramp data flat and linear, not separated into chunks associated with ramps and flat tops
        self.g_event_data = {}
        self.ramps = [] #a list of individually separated ramps
        self.flat_tops = [] # a list of individually separated flat tops in voltage applied
        self.load_hv_textfiles() #parses the files into ramp_data and g_event_data
        self.identify_ramps() #analyzes the ramp data to separate into a list of ramps and a list of flat tops

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
                self.columns.append("ch{:d} seconds".format(sw_ch))
                self.columns.append("ch{:d} nanoseconds".format(sw_ch))
                self.columns.append("ch{:d} n peaks".format(sw_ch)) #integer number of peaks found in this waveform
                self.columns.append("ch{:d} hv".format(sw_ch)) #HV in kV
                self.columns.append("ch{:d} field".format(sw_ch)) #field in kV/cm
                #more specialized
                self.columns.append("ch{:d} charge".format(sw_ch)) #converting amplitude or otherwise into pC 
                self.columns.append("ch{:d} energy".format(sw_ch)) #nJ, assuming charge at a particular voltage, energy fraction of capacitor

                #for identifying events with waveforms if you want to re-reference waveform files
                self.columns.append("ch{:d} filename".format(sw_ch))
                self.columns.append("ch{:d} evidx".format(sw_ch)) #index in the dataframe stored in that file

        self.struck_chmap = {} #indexed by software channel, gives the index of this channel within the "Data" list in prereduced DF row. 
        for i, sw_ch in enumerate(self.config["struck_reduction"]["channel_map"]["software_channel"]):
            self.struck_chmap[sw_ch] = self.config["struck_reduction"]["channel_map"]["prereduced_index"][i] #index of the channel in the list of prereduced data
            self.columns.append("ch{:d} amp".format(sw_ch))
            self.columns.append("ch{:d} afterpulse integral".format(sw_ch))
            self.columns.append("ch{:d} trigger integral".format(sw_ch))
            self.columns.append("ch{:d} baseline".format(sw_ch))
            self.columns.append("ch{:d} postbaseline".format(sw_ch)) #baseline calculated from the back of the waveform
            self.columns.append("ch{:d} noise".format(sw_ch))
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



    #parses and loads the g_events.txt and ramp.txt file with info on
    #what HV is applied at what time, and what time/HV trip signals are received. 
    def load_hv_textfiles(self):

        #we have a few different HV supplies used for different voltage ranges.
        #This conversion text file just has a single text floating point in it
        #that represents the DAC to kV conversion factor. 
        if(os.path.isfile(self.topdir+"dac_conversion.txt")):
            temp = open(self.topdir+"dac_conversion.txt", "r")
            l = temp.readlines()[0]
            dac_conv = float(l)
        else:
            dac_conv = 4 #use the 40 kV glassman value. 


        ad2_epoch = datetime.datetime(1969, 12, 31, 17,0,0)
        #ad2_epoch = datetime.datetime(1970, 1, 1, 0,0,0)
        
        if(os.path.isfile(self.topdir+self.config["ramp_name"])):
            #load the rampfile data
            d = np.genfromtxt(self.topdir+self.config["ramp_name"], delimiter=',', dtype=float)
            ts = d[:,0] #seconds since that epoch above
            ts = [datetime.timedelta(seconds=_) + ad2_epoch for _ in ts] #datetime objects
            v_dac = np.array(d[:,1]) #voltage in volts applied to the control input of the HV supply. needs to be converted for actualy HV applied. 
            v_mon = np.array(d[:,2]) #monitored, if plugged into the external monitor of the supply
            c_mon = np.array(d[:,3]) #monitoring of current, if plugged in. 

            

            v_app = v_dac*dac_conv
            self.ramp_data["t"] = ts
            self.ramp_data["v_app"] = v_app
            self.ramp_data["v_mon"] = v_mon*dac_conv #THIS is the more accurate voltage being applied, not v_app. See calibration folder of 40 kV glassman supply. 
            self.ramp_data["e_app"] = self.ramp_data["v_mon"]/self.config["rog_gap"] #converting to assumed electric field in kV/cm
            self.ramp_data["c_mon"] = c_mon

        else:
            print("no ramp file present at {}, leaving it empty".format(self.topdir+self.config["ramp_name"]))

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
                ts = [datetime.timedelta(seconds=_) + ad2_epoch for _ in ts] #datetime objects
                v_mon = np.array(d[:,1])*dac_conv
                v_app = np.array(d[:,2])*dac_conv
                self.g_event_data["t"] = ts 
                self.g_event_data["v_app"] = v_app
                self.g_event_data["v_mon"] = v_mon
        else:
            print("no g-events-file present at {}, leaving it empty".format(self.topdir+self.config["g_events_name"]))


    #this function takes the flat, 1D data of HV ramp info and separates
    #it into indexable ramps and flat tops. 
    def identify_ramps(self):
        if(self.ramp_data == {}):
            print("No ramp data in this dataset")
            return
        ts = self.ramp_data["t"]
        vs = self.ramp_data["v_mon"]

        #gaussian smooth the noisy voltage monitor data at 2 sample interval
        vs = gaussian_filter(vs, 2)

        #this is not measured data, rather is applied data, so everything is digitally generated and smooth. 
        #instead of using peak finding, its going to call a ramp a period where derivative is positive. 
        #it will call a flat top a period where the value doesn't change. 
        ramps = [{"t":[], "v_mon":[]}]
        flat_tops = [{"t":[], "v_mon":[]}]
        ramping = True #always starts with a ramp as opposed to flat top. use this to trigger a transition in state
        last_vdiff = None

        state_change_thresh = 0.000 #kilovolts, threshold for whether the derivative has changed. 
        for i in range(1,len(ts)):
            vdiff = vs[i] - vs[i-1]
            #first iteration only
            if(last_vdiff is None): last_vdiff = vdiff

            state_change = (np.sign(vdiff) != np.sign(last_vdiff)) and (np.abs(vdiff) > state_change_thresh) #if state changes, then this will be true
            if(ramping and (state_change == False)):
                ramps[-1]["t"].append(ts[i-1])
                ramps[-1]["v_mon"].append(vs[i-1])
            elif((ramping == False) and (state_change == False) and (vdiff == 0)):
                flat_tops[-1]["t"].append(ts[i-1])
                flat_tops[-1]["v_mon"].append(vs[i-1])
            elif(ramping and state_change):
                #what kind of state change is it? Are we going from ramp to a new ramp some time later?
                #or are we going from a ramp to a flat top? 
                if(vdiff < 0):
                    #we are starting a new ramp
                    #add the last value to the last ramp and append a fresh element to the list
                    ramps[-1]["t"].append(ts[i-1])
                    ramps[-1]["v_mon"].append(vs[i-1])
                    ramps.append({"t":[], "v_mon":[]})
                elif(vdiff == 0):
                    #we are starting a flat top
                    #add this value to the most recent flat top element and change the state flag
                    ramping = False
                    flat_tops[-1]["t"].append(ts[i-1])
                    flat_tops[-1]["v_mon"].append(vs[i-1])
            elif((ramping == False) and state_change):
                #we are going from flat top to a new ramp.
                #add the last datapoint to the flat top list and make a fresh
                #flat top element and ramp element, then change the ramping state flag
                ramping = True
                flat_tops[-1]["t"].append(ts[i-1])
                flat_tops[-1]["v_mon"].append(vs[i-1])
                flat_tops.append({"t":[], "v_mon":[]})
                ramps.append({"t":[], "v_mon":[]})
            else:
                print("There is a case in the ramp separation analysis that wasnt considered:")
                print("Ramping: " + str(ramping))
                print("State change: " + str(state_change))
                print("vdiff: " + str(vdiff))
                print("last vdiff: " + str(last_vdiff))
            
            last_vdiff = vdiff
        
        self.ramps = ramps
        self.flat_tops = flat_tops #saved for later. 
                
    #for debugging purposes, plot the ramp data to make
    #sure things are analyzed properly. 
    def plot_ramp_data(self):
        fig, ax = plt.subplots()
        ax.plot(self.ramp_data["t"], self.ramp_data["v_mon"])

        for ft in self.flat_tops:
            ax.plot(ft["t"], ft["v_mon"], 'r')
        for r in self.ramps:
            ax.plot(r["t"], r["v_mon"], 'k')

        plt.show()



    #loads the PMT data and extracts the instantaneous trigger rate
    #as a function of time as a 1D list. Does so by using a histogram method,
    #so a time resolution (or bin width) is provided as input. binwidth in seconds
    def get_rate_curves(self, binwidth):
        #start by getting a 1D list of times and instantaneous rates by
        #looping through all of the PMT files, extracting the timing info, 
        #sorting things in time order. 
        print("Extracting timing info from PMT files...")
        allts = [] #seconds since epoch
        allts_dt = [] #datetime version
        for i, f in enumerate(self.struck_files):
            print("{:d} of {:d}".format(i, len(self.struck_files)), end='\r')
            df, date = pickle.load(open(f, "rb"))
            ts = list(df["Seconds"])
            ts_musec = np.array(list(df["Nanoseconds"])) / 1000 #microseconds
            ts_dt = [datetime.datetime.fromtimestamp(ts[_]) + datetime.timedelta(microseconds=ts_musec[_]) for _ in range(len(ts))]
            allts_dt = allts_dt + ts_dt 
            allts = allts + [ts[_] + ts_musec[_]/1e6 for _ in range(len(ts))]
        print("Binning in time to get rate")
        bins = np.arange(min(allts), max(allts), binwidth)
        n, bin_edges = np.histogram(allts, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        #convert to datetimes for synchronization purposes
        bin_centers = [datetime.datetime.fromtimestamp(_) for _ in bin_centers]

        #convert to Hz. 
        n = np.array(n)/binwidth

        return np.array(bin_centers), np.array(n) #seconds, Hz

    #same as get rate curves, but
    #critically, this function is specific to identifying instantaneous rate
    #while within the bounds of the self.ramps and self.flat_tops regions in time.
    # binwidth in seconds . Modifies the actual ramp and flat top dictionary elements
    # with their respective list of instantaneous rates. 
    def load_rate_curves_into_ramps(self, binwidth):
        pmt_times, rate = self.get_rate_curves(binwidth)

        for i in range(len(self.ramps)):
            if(len(self.ramps[i]["t"]) == 0): continue
            mask, times = self.find_sublist_within_bounds(pmt_times, min(self.ramps[i]["t"]), max(self.ramps[i]["t"]))

            #create new keys in the ramps data for the matching PMT rates
            self.ramps[i]["pmt_rates"] = rate[mask]
            self.ramps[i]["pmt_times"] = times

        for i in range(len(self.flat_tops)):
            if(len(self.flat_tops[i]["t"]) == 0): continue
            mask, times = self.find_sublist_within_bounds(pmt_times, min(self.flat_tops[i]["t"]), max(self.flat_tops[i]["t"]))

            #create new keys in the ramps data for the matching PMT rates
            self.flat_tops[i]["pmt_rates"] = rate[mask]
            self.flat_tops[i]["pmt_times"] = times

        return pmt_times, rate


    #timestamp "Seconds" as input,
    #will find closest HV applied at that time. 
    #Performs some corrections for when the nearest
    #HV log point is farther than 2 seconds
    def get_hv_at_time(self, t):    
        if(self.ramp_data == {}):
            return None
        dt = datetime.datetime.fromtimestamp(t) #cast as datetime to match our ramp_data.

        out_of_bounds = 2 #seconds to perform a different alg to determine HV log points. 
        #find index of closest time in self.ramp_data["t"]
        cl_idx = min(range(len(self.ramp_data["t"])), key=lambda i: abs((self.ramp_data["t"][i] - dt).total_seconds()))
        delta = (self.ramp_data["t"][cl_idx] - dt).total_seconds() #number of seconds off from this logged HV time. 
        if(delta < out_of_bounds):
            return self.ramp_data["v_mon"][cl_idx] #kV #consider linearly interpolating in the future
        else:
            #default for the moment is to just return the closest value for voltage. 
            return self.ramp_data["v_mon"][cl_idx]

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


    #looks at a charge waveform and does a local maximum finding to determine
    #how many peaks there are. If there are more than 1 peak, then it performs
    #the necessary analysis to extract info from the whole waveform about charge, 
    #amplitudes of each peak, etc. 
    def analyze_for_multipeaks_charge(self, row, sw_ch=None, debug=False):
        output = {} #keys are sw channels, but elements are dictionaries with extracted information

        dT = row["dT"]
        #mV, threshold for decided one analysis vs another based on low amplitude. 
        amp_thr = self.config["ad2_reduction"]["glitch"]["fit_amplitude_threshold"]*1000 
        bl_wind = np.array(self.config["ad2_reduction"]["glitch"]["baseline_window"])/(dT*1e6) #in samples
        bl_wind = bl_wind.astype(int)
        #minimum spacing between peaks for rejecting noise
        t_space = 20 #us
        #gaussian smoothing time for finding peaks
        t_smooth = 20 #us


        #a switch in case you only want to process one channel or all
        if(sw_ch is None):
            chs = self.ad2_chmap
        else:
            chs = [sw_ch]

        for sw_ch in chs:
            prered_ind = self.ad2_chmap[sw_ch]
            #get waveform np arrays and time np arrays. 
            v = np.array(row["Data"][prered_ind])

            v = v - np.mean(v[bl_wind[0]:bl_wind[1]]) #baseline subtract
            ts = np.array(np.linspace(0, len(v)*dT*1e6, len(v))) #times in microseconds
            v_sm = gaussian_filter(v, t_smooth/(dT*1e6))
            peaks, _ = scipy.signal.find_peaks(v_sm, distance=int(t_space/(dT*1e6)), height=amp_thr)

            if(debug):
                fig, ax = plt.subplots()
                ax.plot(ts, v)
                ax.plot(ts, v_sm)
                ax.plot(peaks, v[peaks], 'ko')
                plt.show()
            
            output["ch{:d} n peaks".format(sw_ch)] = len(peaks)

            #if this is the case, we need to analyze this whole waveform because
            #the usual fit method won't work. as the code complexifies, this could
            #become more sophisticated. For the moment, things are a bit simplified. 
            #For example, just take the sum of amplitudes as the "amp" - blindly,
            #a real multi-exponential fit would have the second pulse amp a bit less
            #than the voltage measured at the local maximum. Integral is a real integral
            #in this case as well. 
            if(len(peaks) > 1):
                #peak times in the gaussian filtered wave are always
                #going to be a bit after the true peak. Define a window of X
                #microseconds prior to the gaussian filtered peak time that looks
                #for the max sample. 
                peak_window = int(30/(dT*1e6)) #us converted to samples. 
                adjusted_peak_times = []
                peak_amplitudes = []
                for pidx in peaks:
                    if(pidx - peak_window < 0):
                        tmp_start = 0
                    else:
                        tmp_start = pidx - peak_window
                    temp_idx = np.where(v[tmp_start:pidx] == np.max(v[tmp_start:pidx]))[0][0] #index of max value
                    peak_amplitudes.append(v[temp_idx + tmp_start]) #translate to the index referencing the full waveform
                    adjusted_peak_times.append(ts[temp_idx + tmp_start]) #translate to the index referencing the full waveform

                output["ch{:d} amp".format(sw_ch)] = np.sum(peak_amplitudes)
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
        amp_thr = self.config["ad2_reduction"]["glitch"]["fit_amplitude_threshold"]*1000
        bl_wind = np.array(self.config["ad2_reduction"]["glitch"]["baseline_window"])/(dT*1e6) #in samples
        int_wind = np.array(self.config["ad2_reduction"]["glitch"]["integration_window"])/(dT*1e6) #In samples

        bl_wind = bl_wind.astype(int)
        int_wind = int_wind.astype(int)
        rog_cap = float(self.config["rog_cap"]) #pF
        for sw_ch in self.ad2_chmap:
            prered_ind = self.ad2_chmap[sw_ch]
            #get waveform np arrays and time np arrays. 
            v = np.array(row["Data"][prered_ind])
            ts = np.array(np.linspace(0, len(v)*dT*1e6, len(v))) #times in microseconds

            #calculate baselines for use
            output["ch{:d} baseline".format(sw_ch)] = np.mean(v[bl_wind[0]:bl_wind[1]])
            output["ch{:d} postbaseline".format(sw_ch)] = np.mean(v[-1*bl_wind[1]:])

            #change the local waveform variable
            v = v - output["ch{:d} baseline".format(sw_ch)]

            #find the abs-max sample and the sign of that sample. 
            max_abs_index = np.argmax(np.abs(v))
            max_abs_t = ts[max_abs_index]
            trig_idx = max_abs_index
            max_val = v[max_abs_index] #independent of polarity
            
            #if the max val is above the threshold for doing a fit based analysis
            if(np.abs(max_val) >= amp_thr):
                #how many pulses are there in this waveform?
                n_pulses_dict = self.analyze_for_multipeaks_charge(row, sw_ch=sw_ch)
                output["ch{:d} n peaks".format(sw_ch)] = n_pulses_dict["ch{:d} n peaks".format(sw_ch)]
                #if greater than 1, do a multi-pulse analysis 
                if(n_pulses_dict["ch{:d} n peaks".format(sw_ch)] > 1):
                    #then this dict also contains the amplitude reconstruction, sum of
                    #amplitudes of all pulses in the waveform
                    output["ch{:d} amp".format(sw_ch)] = n_pulses_dict["ch{:d} amp".format(sw_ch)]
                #otherwise, fit for one exponential
                else:
                    #get a chunk of data after the trigger time so as 
                    #to fit an exponential with a start time t0. 
                    fit_time_buffer = int(10/(dT*1e6)) #samples, time after the trigger to which to include the fit
                    ts_trim = ts[trig_idx+fit_time_buffer:]
                    v_trim = v[trig_idx+fit_time_buffer:]
                    initial_guess = [max_val, 400] #400 is the estimated tau from previous explorative fits. 

                    #ODR based objects for performing fit
                    odr_model = odr.Model(utilities.exp_wrapper_fixedt0(max_abs_t))
                    odr_data = odr.Data(ts_trim, v_trim)
                    odr_inst = odr.ODR(odr_data, odr_model, beta0=initial_guess)
                    odr_result = odr_inst.run()
                    amp = odr_result.beta[0]
                    tau = odr_result.beta[1]

                    #to confirm with plots
                    if(debug):
                        fig, ax = plt.subplots()
                        ax.plot(ts, v)
                        ax.plot(ts_trim, utilities.fitfunc_exp(odr_result.beta, ts_trim, max_abs_t), label="{:.2f} e^(-t/{:.2f})".format(*odr_result.beta))
                        ax.set_xlabel("[us]")
                        ax.set_ylabel('[mV]')
                        ax.legend()
                        plt.show()

                    output["ch{:d} amp".format(sw_ch)] = amp 

                


            #if the event has no samples above the sample threshold for fit based analysis
            else:
                #just find amplitude as max sample. 
                output["ch{:d} amp".format(sw_ch)] = max_val
                output["ch{:d} n peaks".format(sw_ch)] = 0

            #analysis variables agnostic to multipulse structure, amplitude threshold, and otherwise.

            #integral
            output["ch{:d} full integral".format(sw_ch)] = scipy.integrate.trapezoid(v[int_wind[0]:int_wind[1]], dx=dT*1e6) #mV*us
            #baseline (which was subtracted in prereduction, but recalculated here for completeness)
            output["ch{:d} baseline".format(sw_ch)] = np.mean(v[bl_wind[0]:bl_wind[1]])
            output["ch{:d} postbaseline".format(sw_ch)] = np.mean(v[-1*bl_wind[1]:])
            #noise
            output["ch{:d} noise".format(sw_ch)] = np.std(v[bl_wind[0]:bl_wind[1]])
            #what is kV applied at this time, and save time variables
            output["ch{:d} seconds".format(sw_ch)] = row["Seconds"]
            output["ch{:d} nanoseconds".format(sw_ch)] = row["Nanoseconds"]
            kv = self.get_hv_at_time(row["Seconds"])
            output["ch{:d} hv".format(sw_ch)] = kv 
            output["ch{:d} field".format(sw_ch)] = self.get_field_for_hv(kv)
            #the charge can be reconstructed from the amplitude (or integral, if you calibrate that way)
            reco_charge = self.get_charge_from_mv(output["ch{:d} amp".format(sw_ch)])
            output["ch{:d} charge".format(sw_ch)] = reco_charge
            #if you assume C = 20 pF for the rogowski, then this charge has deposited
            #some quantity of energy, equal to dU = Q(dQ)/C in Joules
            if(kv != None):
                fullQ = rog_cap*kv*1000 #V*pF = pC
                dU = reco_charge*fullQ/rog_cap/1000 #pC*pC/pF = pJ, most are on order nJ hence factor of 1000
                output["ch{:d} energy".format(sw_ch)] = dU
            else:
                output["ch{:d} energy".format(sw_ch)] = None

            output["ch{:d} evidx".format(sw_ch)] = row.name

            
                
        return output

    def analyze_light_waveforms(self, row):
        output = {}
        ph_window = self.config["struck_reduction"]["pulseheight_window"] #samples
        ap_window = self.config["struck_reduction"]["afterpulse_window"] #samples 
        bl_window = self.config["struck_reduction"]["baseline_window"] #samples
        dT = 1e6/float(self.config["struck_reduction"]["clock"]) #us

        for i, sw_ch in enumerate(self.struck_chmap):
            prered_ind = self.struck_chmap[sw_ch]
            v = row["Data"][prered_ind]

            #pulse height and basic integrals
            output["ch{:d} amp".format(sw_ch)] = np.max(v[ph_window[0]:ph_window[1]])
            output["ch{:d} afterpulse integral".format(sw_ch)] = scipy.integrate.trapezoid(v[ap_window[0]:ap_window[1]], dx=dT) #mV*us
            output["ch{:d} trigger integral".format(sw_ch)] = scipy.integrate.trapezoid(v[ph_window[0]:ph_window[1]], dx=dT) #mV*us

            #noise and baseline
            output["ch{:d} noise".format(sw_ch)] = np.std(v[bl_window[0]:bl_window[1]])
            output["ch{:d} baseline".format(sw_ch)] = np.mean(v[bl_window[0]:bl_window[1]])
            #baseline calculated from the back of the waveform
            output["ch{:d} postbaseline".format(sw_ch)] = np.mean(v[-1*bl_window[1]:]) 

            #hv and timing
            kv = self.get_hv_at_time(row["Seconds"]) #returns None if no ramp data 
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
                    output = self.analyze_charge_waveforms(row)
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
