#This class is meant to distinguish tools geared toward waveform analysis
#and pre-reduction with tools geared towards interacting with the
#reduced data dataframe, which contains only quantities after waveform reduction. 


import numpy as np
import matplotlib.pyplot as plt 
import pickle
import os
import yaml


class AnalysisTools:
    #reduced_df_pickle is the filename of the pickle file containing reduced dataframe
    def __init__(self, reduced_df_pickle, config_file):
        self.fn = reduced_df_pickle
        self.df = None

        self.config_file = config_file
        self.config = {}
        self.load_config()

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
    
    
    

