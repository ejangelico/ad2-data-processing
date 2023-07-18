
import os
import yaml
import datetime 
import numpy as np 
import matplotlib.pyplot as plt 
import glob 
import pickle


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
        if(len(self.ad2_files) == 0):
            print("Problem! Found no AD2 files with prefix 'prereduced*.p' in {}".format(self.topdir))
        if(len(self.struck_files) == 0):
            print("Problem! Found no Struck data files with prefix 'prereduced*.p' in {}".format(self.topdir+self.config["struck_folder"]))

        
        

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

        #unfortunately, as of Run5 and Run6, two supplies were used for different datasets
        #and this changes the conversion from v_dac to HV applied. Will become vestigial in the future
        #but I'm putting a single file in the topdir that indicates that conversion, i.e. whether it was
        #the glassman or the SRS supply for that dataset, as I was not smart enough to encode that info
        #in the ramp file. 
        if(os.path.isfile(self.topdir+"dac_conversion.txt")):
            temp = open(self.topdir+"dac_conversion.txt", "r")
            l = temp.readlines()[0]
            dac_conv = float(l)
        else:
            dac_conv = 0.5 #use SRS value for the 5 kV supply


        ad2_epoch = datetime.datetime(1969, 12, 31, 17,0,0)
        
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
            self.ramp_data["e_app"] = v_app/self.config["rog_gap"] #converting to assumed electric field in kV/cm
            self.ramp_data["v_mon"] = v_mon*dac_conv
            self.ramp_data["c_mon"] = c_mon

        else:
            print("no ramp file present at {}, leaving it empty".format(self.topdir+self.config["ramp_name"]))

        #load the g_events data, if it exists
        if(os.path.isfile(self.topdir+self.config["g_events_name"])):
            d = np.genfromtxt(self.topdir+self.config["g_events_name"], delimiter=',', dtype=float)
            #there is a silly thing with genfromtxt where if its a 1 line file, it makes a 1D array instead of the usual
            #2D array. This line forces it into a 2D array so the other lines don't need some other if statement. 
            if(len(d.shape) == 1): d = np.array([d])
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
        ts = self.ramp_data["t"]
        vs = self.ramp_data["v_app"]
        #this is not measured data, rather is applied data, so everything is digitally generated and smooth. 
        #instead of using peak finding, its going to call a ramp a period where derivative is positive. 
        #it will call a flat top a period where the value doesn't change. 
        ramps = [{"t":[], "v_app":[]}]
        flat_tops = [{"t":[], "v_app":[]}]
        ramping = True #always starts with a ramp as opposed to flat top. use this to trigger a transition in state
        last_vdiff = None
        for i in range(1,len(ts)):
            vdiff = vs[i] - vs[i-1]
            #first iteration only
            if(last_vdiff is None): last_vdiff = vdiff

            state_change = np.sign(vdiff) != np.sign(last_vdiff) #if state changes, then this will be true
            if(ramping and (state_change == False)):
                ramps[-1]["t"].append(ts[i-1])
                ramps[-1]["v_app"].append(vs[i-1])
            elif((ramping == False) and (state_change == False) and (vdiff == 0)):
                flat_tops[-1]["t"].append(ts[i-1])
                flat_tops[-1]["v_app"].append(vs[i-1])
            elif(ramping and state_change):
                #what kind of state change is it? Are we going from ramp to a new ramp some time later?
                #or are we going from a ramp to a flat top? 
                if(vdiff < 0):
                    #we are starting a new ramp
                    #add the last value to the last ramp and append a fresh element to the list
                    ramps[-1]["t"].append(ts[i-1])
                    ramps[-1]["v_app"].append(vs[i-1])
                    ramps.append({"t":[], "v_app":[]})
                elif(vdiff == 0):
                    #we are starting a flat top
                    #add this value to the most recent flat top element and change the state flag
                    ramping = False
                    flat_tops[-1]["t"].append(ts[i-1])
                    flat_tops[-1]["v_app"].append(vs[i-1])
            elif((ramping == False) and state_change):
                #we are going from flat top to a new ramp.
                #add the last datapoint to the flat top list and make a fresh
                #flat top element and ramp element, then change the ramping state flag
                ramping = True
                flat_tops[-1]["t"].append(ts[i-1])
                flat_tops[-1]["v_app"].append(vs[i-1])
                flat_tops.append({"t":[], "v_app":[]})
                ramps.append({"t":[], "v_app":[]})
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
        ax.plot(self.ramp_data["t"], self.ramp_data["v_app"])

        for ft in self.flat_tops:
            ax.plot(ft["t"], ft["v_app"], 'r')
        for r in self.ramps:
            ax.plot(r["t"], r["v_app"], 'k')

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
            start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
            ts = list(df["Seconds"])
            ts_dt = [start_of_day + datetime.timedelta(seconds=_) for _ in ts]
            allts_dt = allts_dt + ts_dt 
            allts = [(_ - datetime.datetime(1970, 1, 1)).total_seconds() for _ in allts_dt]
        print("Binning in time to get rate")
        bins = np.arange(min(allts), max(allts), binwidth)
        n, bin_edges = np.histogram(allts, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        #convert to datetimes for synchronization purposes
        bin_centers = [datetime.datetime(1970,1,1) + datetime.timedelta(seconds=_) for _ in bin_centers]

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

    #finding a sublist
    #between two time brounds (chronologically ordered) with list 
    #chronological as well. 
    def find_sublist_within_bounds(self, lst, t0, t1):
        lst = np.array(lst)
        mask = np.where((lst >= t0) & (lst <= t1))
        return mask, lst[mask]

