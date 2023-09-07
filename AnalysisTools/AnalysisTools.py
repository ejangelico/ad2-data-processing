#This class is meant to distinguish tools geared toward waveform analysis
#and pre-reduction with tools geared towards interacting with the
#reduced data dataframe, which contains only quantities after waveform reduction. 


import numpy as np
import matplotlib.pyplot as plt 
import pickle
import os
import yaml
import pandas as pd


class AnalysisTools:
    #reduced_df_pickle is the filename of the pickle file containing reduced dataframe
    def __init__(self, reduced_df_pickle, config_file):
        self.fn = reduced_df_pickle
        self.df = None

        self.config_file = config_file
        self.config = {}
        self.load_config()

        self.ad2_chmap = {}
        self.struck_chmap = {}
        self.load_chmaps()

        self.all_sw_chs = list(self.ad2_chmap.keys()) + list(self.struck_chmap.keys())

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
            for i, sw_ch in enumerate(self.config["ad2_reduction"][ad2]["software_channels"]):
                #for saving these software channels for easier access
                self.ad2_chmap[sw_ch] = self.config["ad2_reduction"][ad2]["active_channels"][i]
        for card in self.config["struck_reduction"]["software_channels"]:
            for i, sw_ch in enumerate(self.config["struck_reduction"]["software_channels"][card]):
                self.struck_chmap[sw_ch] = self.config["struck_reduction"]["active_channels"][card][i]

    #this will get waveforms, from their waveform
    #level pre-reduced files, that pass a mask
    #on the df. Pass a list of sw_chs to grab. 
    def get_waveforms_from_cuts(self, mask, sw_chs):
        dd = self.df[mask] #masked df
        output_events = {}
        for sw_ch in sw_chs:
            output_events[sw_ch] = []
            filenames = list(dd["ch{:d} filename".format(sw_ch)])
            evidx = list(dd["ch{:d} evidx".format(sw_ch)]) 

            filenames_set = list(set(filenames)) #unique filenames only. 
            for f in filenames_set:
                df, date = pickle.load(open(f, "rb"))
                for i in range(len(evidx)):
                    if(filenames[i] == f):
                        event = df.iloc[evidx[i]]
                        output_events[sw_ch].append(event)

        return output_events
    
    
    #takes a dataframe that is input_events, probably
    #formed by a mask such as charge amplitude > 5 mV. 
    #sw_ch is the channel to get time info from input_events
    #Finds events with timestamps (sec and nanosec) within 
    #a provided half-coincidence window. 
    def get_coincidence(self, input_events, sw_ch, coinc, coinc_ns):
        output_dfs = [] #list of dataframes that match coincidence cuts for each event
        for i, ev in input_events.iterrows():
            t0 = ev["ch{:d} seconds".format(sw_ch)]
            t0_ns = ev["ch{:d} nanoseconds".format(sw_ch)]
            
            #look through each other software channel and form masks 
            temp_df = pd.DataFrame()
            for sch in self.all_sw_chs:
                #ignore the channel that is the input channel
                if(sch == sw_ch):
                    continue
                mask = ((self.df["ch{:d} seconds".format(sch)] - (t0 - coinc) + ((self.df["ch{:d} nanoseconds".format(sch)] - (t0_ns - coinc_ns))/1e9)) >= 0) &\
                    ((self.df["ch{:d} seconds".format(sch)] - (t0 + coinc) + ((self.df["ch{:d} nanoseconds".format(sch)] - (t0_ns + coinc_ns))/1e9)) <= 0)
                selected = self.df[mask]
                temp_df = pd.concat([temp_df, selected], ignore_index=True)
                
            output_dfs.append(temp_df)

        return output_dfs
    

