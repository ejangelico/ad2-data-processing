import os
import sys
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt 
from scipy import signal
import random
import pandas as pd
from scipy.ndimage import gaussian_filter
from pathlib import Path
import shutil




def parse_timestamp_from_filename(infile):
    #infile looks like /path/to/data/file/pmt15.23.43.132.csv (hour, minute, second, milli)
    t = infile.split('/')[-1].split('.')[:-1]
    t[0] = t[0][-2:] #hour is always only two digits, this line just ignores file prefix 
    t = [int(_) for _ in t]
    #now is in form [hours, minutes, seconds, millis]
    milliseconds = t[3] + 1e3*t[2] + 1e3*60*t[1] + 1e3*60*60*t[0]
    return milliseconds

#looks at the input directory (a dataset) and
#finds all .csv files, separating them by file prefix
def get_separated_file_lists(indir, file_prefixes, nevents=None):
    #full list of .csv files
    file_list = []
    if(nevents is not None):
        for i, f in enumerate(os.listdir(indir)):
            if(i > nevents):
                break
            if(os.path.isfile(os.path.join(indir, f)) \
                 and f.endswith('.csv')):
                file_list.append(f)
    else:
        file_list = [f for f in os.listdir(indir) if os.path.isfile(os.path.join(indir, f)) \
                 and f.endswith('.csv')]
    
    separate_file_lists = {}
    for pref in file_prefixes:
        #selects filenames by prefix. so separate_file_lists['pmt'] = ['pmt14.53.24.449', 'pmt10.34....', ...]
        separate_file_lists[pref] = list(filter(lambda x: x[:len(pref)] == pref, file_list))  
    
    return separate_file_lists

#converts the dictionary of separated file lists into
#a dictionary of separated timestamps (units milliseconds)
def get_separated_timestamps(separated_file_lists):
    separated_timestamps = {}
    for pref in separated_file_lists:
        separated_timestamps[pref] = [parse_timestamp_from_filename(f) for f\
                                      in separated_file_lists[pref]]
       
        #sort both the timestamps lists and the filelists
        #simultaneously by the timestamps
        separated_timestamps[pref], separated_file_lists[pref] = \
        (list(t) for t in zip(*sorted(zip(separated_timestamps[pref], separated_file_lists[pref]))))
    
    return separated_timestamps, separated_file_lists

def get_sampling_period_from_file(infile):
    #parse header for the timestep
    f = open(infile, 'r', errors='ignore')
    ls = f.readlines()
    raw_sample_rate = ls[4]
    raw_sample_rate = raw_sample_rate.split(' ')[-1]
    raw_sample_rate = float(raw_sample_rate.split('H')[0])
    return (1.0/raw_sample_rate)*1e9 #nanoseconds



#data processing utilities

#median of the first 100 us
#very "stupid" function, just blindly subtracts
def baseline_subtract_1(event_series):
    dt = event_series['SamplingPeriods'][0]/1000.0 #us, [0] and [1] are identical here, anode vs glitch
    median_buffer_duration = 100 #us
    med_buf_didx = int(median_buffer_duration/dt) #number of indices in list for buffer
    
    newdata = []
    for chan in range(len(event_series['Data'])):
        raw_data = event_series['Data'][chan]
        med_buffer = raw_data[:med_buf_didx]
        median = np.median(med_buffer)
        newdata.append(raw_data - median)
        
    event_series['Data'] = newdata
    

#a notch filter tuned to the various
#noise sources in our datasets. only filtering anode
#channel
def notch_filter(event_series):
    ch = 1 #anode channel
    fs = 1.0/event_series['SamplingPeriods'][ch] #GHz
    #hv power supply
    f0 = 107e-6 #100 kHz
    Q = 20 #dB, df = f0/Q
    b, a = signal.iirnotch(f0, Q, fs)
    raw_data = event_series['Data'][ch]
    data_notched = signal.filtfilt(b, a, raw_data)
    event_series['Data'][ch] = data_notched
    
    
    
#returns amplitudes, time constants, and peak indices of exponentials in waveform. 
#does quick thing of returning max-sample and tau=none if max sample is below fit_threshold.
#if above fit threshold, fits to an exponential. 

#integrates about the peak time from a tuple window
#like [-50, 500] which will do -50 us from ptime to 500 us 
#will not integrate if amplitude is lower than threshold in mV
def get_basic_waveform_properties(event_series, fit_amplitude_threshold, window, zero_amplitude_threshold):
    amps = [None]*len(event_series['Data']) #amplitudes
    taus = [None]*len(event_series['Data']) #exp time constants, none usually
    pidx = [None]*len(event_series['Data']) #peak times
    integrals = [None]*len(event_series['Data'])
    
    for i, rawdata in enumerate(event_series['Data']):
        #returns highest absolute value, negative or positive polar
        #i.e. max(-5, 3, key=abs) will return -5 (as opposed to 5)
        maxval = max(rawdata.min(), rawdata.max(), key=abs) 
        maxidx = np.where(rawdata == maxval)[0][0]
        
        #if this is larger than the threshold, do a fit. 
        if(abs(maxval) > fit_amplitude_threshold):
            #do fit later
            #print("PLEASE CODE IN AN EXP FITTER")
            #return None
            amps[i] = maxval
            pidx[i] = maxidx #time in ns
            
        #otherwise, just return peak time, no tau, and amplitude
        else:
            amps[i] = maxval
            pidx[i] = maxidx #time in ns
    
    #integrate based on the time of the ANODE peak only
    for i, rawdata in enumerate(event_series['Data']):
        if(abs(amps[i]*1000) > zero_amplitude_threshold):
            anode_peak = pidx[1]
            dt = event_series['SamplingPeriods'][i] #ns
            lowidx = int(anode_peak + min(window)*1e3/dt)
            hiidx = int(anode_peak + max(window)*1e3/dt) #1e3 because window in us
            integ_data = rawdata[lowidx:hiidx]*1000 #mV
            integrals[i] = (np.trapz(integ_data, dx=dt)/1e3) #in mV*us

                
        
    return amps, taus, pidx, integrals



def reduce_and_process(dataset, topdir, nevents, nevents_per_file):


	#if the folders don't exist, create them.
	Path(topdir+dataset+"rawhdf").mkdir(parents=True, exist_ok=True)
	Path(topdir+dataset+"reduced").mkdir(parents=True, exist_ok=True)
	#WARNING: this next line will delete already processed data
	#sometimes this is desireable, but please double check!
	folder = topdir+dataset+"rawhdf/"
	for filename in os.listdir(folder):
		file_path = os.path.join(folder, filename)
		try:
			if(os.path.isfile(file_path)):
				os.unlink(file_path)
		except Exception as e:
			print('Failed to delete %s. Reason: %s' % (file_path, e))
 	
	folder = topdir+dataset+"reduced/"
	for filename in os.listdir(folder):
		file_path = os.path.join(folder, filename)
		try:
			if(os.path.isfile(file_path)):
				os.unlink(file_path)
		except Exception as e:
			print('Failed to delete %s. Reason: %s' % (file_path, e))
 	

	#the part of the filename before the timestamp. 
	#used to distinguish the two oscilloscopes
	file_prefixes = ["pmt", "anode"]
	

	#load dataset
	t0 = time.time()
	separated_file_lists = get_separated_file_lists(topdir+dataset, file_prefixes, nevents)
	print("Separation by prefix took " + str(time.time() - t0))

	#which channel do you want?
	file_prefix = "anode" 
	t0 = time.time()

	#analysis/processing parameters
	filter_tau = 300 #ns #filtering anode signal gaussian
	zero_amplitude_threshold = 1 #if max-sample is less than this, don't integrate
	fit_amplitude_threshold = 300 #if max-sample is above this, fit the exponential to get tau and amplitude
	integration_window = [-100, 500] #skip integrating the baseline samples, just integrate this window about peak time.

	wf_output_df = pd.DataFrame()
	reduced_output_df = pd.DataFrame()
	if(len(separated_file_lists[file_prefix]) == 0):
		return 
	for i, infile in enumerate(separated_file_lists[file_prefix]):
		if(i % 10 == 0): print("On event " + str(i) + " of " + str(nevents) + " or " + str(len(separated_file_lists[file_prefix])))
		event_series = pd.Series()
		reduced_series = pd.Series()
		event_series['Timestamps'] = [parse_timestamp_from_filename(topdir+dataset+infile)]*2 #in milliseconds since 00:00 (midnight)

		event_series['SamplingPeriods'] = [get_sampling_period_from_file(topdir+dataset+infile)]*2 #nanoseconds

		#load the file
		d = pd.read_csv(topdir+dataset+infile, header=None, skiprows=20, names=['ts','0','1'], encoding='iso-8859-1')
		event_series['Channels'] = ["glitch", "anode"]
		event_series['ChannelTypes'] = ["glitch", "anode"]
		data_map = [d['0'].to_numpy(), d['1'].to_numpy()] 
		event_series['Data'] = data_map
		baseline_subtract_1(event_series)
		#ax = plot_anode_scope(event_series)

		event_series['Data'][1] = gaussian_filter(event_series['Data'][1], filter_tau/float(event_series['SamplingPeriods'][1]))
		#plot_anode_scope(event_series, ax)
		#plt.show()

		#calculate amplitude and integral
		#put processed quantities like amplitude and integral in a reduced series
		amplitudes, taus, peakidx, integrals = get_basic_waveform_properties(event_series, fit_amplitude_threshold, integration_window, zero_amplitude_threshold) #finds tau's if relevant
		"""
		ax = plot_anode_scope(event_series)
		dt = event_series['SamplingPeriods'][0]*1e-6
		ax.scatter([_*dt for _ in peakidx], [_*1000 for _ in amplitudes], color='r', s=30)
		dt = event_series['SamplingPeriods'][1] #ns
		iwindow = [_/1e3 + peakidx[1]*dt/1e6 for _ in integration_window]
		"""

		reduced_series['GlitchAmplitude'] = amplitudes[0] #in mV, negative or positive
		reduced_series['AnodeAmplitude'] = amplitudes[1] #in mV, negative or positive
		reduced_series['GlitchTau'] = taus[0] #time constant of exponential fit
		reduced_series['AnodeTau'] = taus[1] #time constant of exponential fit
		reduced_series['GlitchPeakidx'] = peakidx[0] #index referencing event_series['Data']
		reduced_series['AnodePeakidx'] = peakidx[1] #index referencing event_series['Data']
		reduced_series['GlitchIntegral'] = integrals[0] #mV*us
		reduced_series['AnodeIntegral'] = integrals[1] #mV*us

		wf_output_df = wf_output_df.append(event_series, ignore_index=True)
		reduced_output_df = reduced_output_df.append(reduced_series, ignore_index=True)
		if((i+1) % nevents_per_file == 0):
			print("Took " + str(time.time() - t0) + " seconds for " + str(i) + " events")
			print("Saving now...")

			#if the folders don't exist, create them.
			Path(topdir+dataset+"rawhdf").mkdir(parents=True, exist_ok=True)
			Path(topdir+dataset+"reduced").mkdir(parents=True, exist_ok=True)

			wf_output_df.to_hdf(topdir+dataset+"rawhdf/"+str(i+1)+".h5", key='raw')
			reduced_output_df.to_hdf(topdir+dataset+"reduced/"+str(i+1)+".h5", key='raw')
			print("Done")
			#reinitialize
			wf_output_df = pd.DataFrame()
			reduced_output_df = pd.DataFrame()


	print("Saving now...")

	#if the folders don't exist, create them.
	Path(topdir+dataset+"rawhdf").mkdir(parents=True, exist_ok=True)
	Path(topdir+dataset+"reduced").mkdir(parents=True, exist_ok=True)

	wf_output_df.to_hdf(topdir+dataset+"rawhdf/"+str(i+1)+".h5", key='raw')
	reduced_output_df.to_hdf(topdir+dataset+"reduced/"+str(i+1)+".h5", key='raw')
	print("Done")
	#reinitialize
	wf_output_df = pd.DataFrame()
	reduced_output_df = pd.DataFrame()

if __name__ == "__main__":
	datatopdir_r1 = "/p/lustre1/angelico/hv-test-chamber/Run1/"
	datasets_r1 = {2:"1-29-21/pmt-trig-filling-1800/", \
	                3:"1-29-21/pmt-trig-filling-1920/",\
	                4:"1-30-21/anode-crosstrig-1300/",\
	                6:"1-30-21/ignition-1500/",\
	                7:"1-30-21/ignition-10k-1520/",\
	                8:"1-31-21/glitch-1520/",\
	                9:"2-1-21/anode-100/",\
	                10:"2-1-21/anode-1340/",\
	                11:"2-1-21/glitch-1530/",\
	                12:"2-1-21/glitch-2230/",\
	                13:"2-2-21/anode-1030/",\
	                14:"2-2-21/corona-1300/",\
	                15:"2-2-21/glitch-1320/",\
	                16:"2-2-21/glitch-1430/",\
	                17:"2-2-21/anode-1720/",\
	                18:"2-3-21/glitch-1040/",\
	                19:"2-3-21/anode-1050/",\
	                20:"2-3-21/glitch-1810/",\
	                21:"2-3-21/anode-1820/"}

	#check that the datasets are indexible
	print("Checking for datasets in top directory: " + datatopdir_r1)
	to_remove = [] #list of keys to remove due to not existing
	for dno in datasets_r1:
	    print(str(dno)+": " + datasets_r1[dno] + "\t\t", end = '')
	    isdir = os.path.isdir(datatopdir_r1+datasets_r1[dno]+"/")
	    if(isdir):
	        numfiles = len([_ for _ in os.listdir(datatopdir_r1+datasets_r1[dno]+"/")])
	        print("True, with " + str(numfiles) + " files")
	    else:
	        print("False")
	        to_remove.append(dno)
	        

	for k in to_remove:
	    del datasets_r1[k]
	    

	datatopdir_r2 = "/p/lustre1/angelico/hv-test-chamber/Run2/"
	datasets_r2 = {25:"2-22-2021/cosmics-1725/",\
	              26:"2-22-2021/cosmics-1800/",\
	              28:"2-23-2021/anode-1240/",\
	              29:"2-23-2021/pmts-1240/",\
	              30:"2-23-2021/cosmics-1430/",\
	              31:"2-23-2021/glitch-2000/",\
	              32:"2-24-2021/glitch-1140/",\
	              33:"2-24-2021/glitch-1215/",\
	              34:"2-24-2021/glitch-1425/",\
	              35:"2-24-2021/anode-1730/",\
	              36:"2-25-2021/anode-1030/",\
	              37:"2-26-2021/anode-1430/"}

	#check that the datasets are indexible
	print("Checking for datasets in top directory: " + datatopdir_r2)
	to_remove = [] #list of keys to remove due to not existing
	for dno in datasets_r2:
	    print(str(dno)+": " + datasets_r2[dno] + "\t\t", end = '')
	    isdir = os.path.isdir(datatopdir_r2+datasets_r2[dno]+"/")
	    if(isdir):
	        numfiles = len([_ for _ in os.listdir(datatopdir_r2+datasets_r2[dno]+"/")])
	        print("True, with " + str(numfiles) + " files")
	    else:
	        print("False")
	        to_remove.append(dno)
	        

	for k in to_remove:
	    del datasets_r2[k]


	sets_to_process = range(38)
	nevents = 20000
	nevents_per_file = 1000
	for no in sets_to_process:
		if(not(no in datasets_r1.keys()) and not(no in datasets_r2.keys())): continue
		if(no < 25):
			topdir = datatopdir_r1
			dataset = datasets_r1[no]
		else:
			topdir = datatopdir_r2
			dataset = datasets_r2[no]
	
		print("On dataset " + str(topdir+dataset))
		reduce_and_process(dataset, topdir, nevents, nevents_per_file)
    
