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
	dt = event_series['AnodeSamplingPeriod']/1000.0 #us, [0] and [1] are identical here, anode vs glitch
	median_buffer_duration = 100 #us
	med_buf_didx = int(median_buffer_duration/dt) #number of indices in list for buffer
	
	newdata = []
	for chan in range(len(event_series['Data'])):
		raw_data = event_series['Data'][chan]
		med_buffer = raw_data[:med_buf_didx]
		median = np.median(med_buffer)
		newdata.append(raw_data - median)
		
	event_series['Data'] = newdata
	
#baseline subtract for PMTs, mean subtraction
#using the first and last 2 us of entire buffer
def baseline_subtract_2(event_series):
	dt = event_series['PMTSamplingPeriod']/1000.0 #us, [0] and [1] are identical here, anode vs glitch
	mean_buffer_duration = 1 #us
	mean_buf_didx = int(mean_buffer_duration/dt) #number of indices in list for buffer
	
	newdata = []
	for chan in range(len(event_series['Data'])):
		raw_data = event_series['Data'][chan]
		mean_buffer = raw_data[:mean_buf_didx]
		mean_buffer += raw_data[-mean_buf_didx:]
		mean = np.mean(mean_buffer)
		newdata.append(raw_data - mean)
		
	event_series['Data'] = newdata

#a notch filter tuned to the various
#noise sources in our datasets. only filtering anode
#channel
def notch_filter(event_series):
	ch = 1 #anode channel
	fs = 1.0/event_series['AnodeSamplingPeriod'] #GHz
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

#for PMTs, does no fitting, just max-sample and full integration over the window. 

#integrates about the peak time from a tuple window
#like [-50, 500] which will do -50 us from ptime to 500 us 
#will not integrate if amplitude is lower than threshold in mV
def get_basic_waveform_properties(event_series, fit_amplitude_threshold, window, zero_amplitude_threshold):
	amps = [] #amplitudes
	taus = [] #exp time constants, none usually
	pidx = [] #peak times
	integrals = []
	
	for i, rawdata in enumerate(event_series['Data']):
		#returns highest absolute value, negative or positive polar
		#i.e. max(-5, 3, key=abs) will return -5 (as opposed to 5)
		maxval = max(rawdata.min(), rawdata.max(), key=abs) 
		maxidx = np.where(rawdata == maxval)[0][0]
		
		if(event_series['ChannelTypes'][i] == "anode" or event_series['ChannelTypes'][i] == "glitch"):
			#if this is larger than the threshold, do a fit. 
			#and only fit if its an anode channel. 
			if(abs(maxval) > fit_amplitude_threshold and i == 1):
				#do fit later
				dt = event_series['AnodeSamplingPeriod'][i] #ns
				times = np.array(np.arange(0, len(rawdata)*dt, dt))
				p0 = [maxval, times[maxidx], dt*10, 150e3] #guess for fitter, amp and time of peak. 
				popt, pcov = curve_fit(anode_fit_function, times, rawdata, p0=p0)
				fity = anode_fit_function(times, *popt)

				fig, ax = plt.subplots(figsize=(10, 6))
				ax.plot([_/1e6 for _ in times], fity, label=popt)
				ax.plot([_/1e6 for _ in times], rawdata)
			   #ax.set_xlim([times[maxidx]/1e6 - 0.1, times[maxidx]/1e6 + 0.1])
				ax.legend()
				plt.show()
				pidx.append(maxidx)
				amps.append(maxval)
				taus.append(None)


			#otherwise, just return peak time, no tau, and amplitude
			else:
				amps.append(maxval)
				taus.append(None)
				pidx.append(maxidx) 
				
		elif(event_series['ChannelTypes'][i] == "pmt"):
			amps.append(maxval)
			taus.append(None)
			pidx.append(maxidx)
	
	
	#integration functions
	for i, rawdata in enumerate(event_series['Data']):
		if(event_series['ChannelTypes'][i] == "anode" or event_series['ChannelTypes'][i] == "glitch"):
			anode_peak = pidx[1]
			dt = event_series['AnodeSamplingPeriod'] #ns
			lowidx = int(anode_peak + min(window)*1e3/dt)
			hiidx = int(anode_peak + max(window)*1e3/dt) #1e3 because window in us
			integ_data = rawdata[lowidx:hiidx]*1000 #mV
			integrals.append(np.trapz(integ_data, dx=dt)/1e3) #in mV*us

				
		elif(event_series['ChannelTypes'][i] == "pmt"):
			#check if amplitude is below a threshold, for which we choose not to integrate
			dt = event_series['PMTSamplingPeriod'] #ns
			lowidx = int(pidx[0] + min(window)*1e3/dt)
			hiidx = int(pidx[0] + max(window)*1e3/dt) #1e3 because window in us
			integ_data = rawdata[lowidx:hiidx] #V
			integ_vs = np.trapz(integ_data, dx=dt)/1e9 #V*s
			integ_coul = integ_vs/50.0 #50 ohms
			integ_mega_elec = (integ_coul/1.6e-19)/1e9 #billion electrons
			integrals.append(integ_mega_elec) #billion electrons

	

				
		
	return amps, taus, pidx, integrals




def reduce_and_process(dataset, topdir, nevents, nevents_per_file, rogowski_voltage):


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
	file_prefixes = ["pmts", "anode"]
	

	#load dataset
	t0 = time.time()
	separated_file_lists = get_separated_file_lists(topdir+dataset, file_prefixes, nevents)
	print("Separation by prefix took " + str(time.time() - t0))

	t0 = time.time()

	#analysis/processing parameters
	filter_tau = 300 #ns #filtering anode signal gaussian
	zero_amplitude_threshold = 1 #if max-sample is less than this, don't integrate
	fit_amplitude_threshold = 300 #if max-sample is above this, fit the exponential to get tau and amplitude
	anode_integration_window = [-100, 500] #us, skip integrating the baseline samples, just integrate this window about peak time.
	pmt_integration_window = [0, 85] #us, encompasing the entire event 

	wf_output_df = pd.DataFrame()
	reduced_output_df = pd.DataFrame()

	#which list, anode or pmt, has highest number of events
	max_evts = max(len(separated_file_lists["pmts"]), len(separated_file_lists["anode"]))

	#we are going to blindly group the PMT and anode files into the same "event"
	#even though they are not necessarily timestamp matched. This is just for organizational
	#purposes. Eventually, need to match them up... this means that some events will
	#have only pmt or only anode, but many will have both. 
	for i in range(max_evts):
		if(i % 10 == 0): print("On event " + str(i) + " of " + str(nevents) + " or " + str(max_evts))
		event_series = pd.Series()
		reduced_series = pd.Series()
		event_series['RogowskiVoltage'] = rogowski_voltage
		reduced_series['RogowskiVoltage'] = rogowski_voltage
		#determine if we have both, or only one type of event
		pmtfile = None
		anodefile = None
		if(i < len(separated_file_lists["pmts"])):
			pmtfile = topdir+dataset+separated_file_lists["pmts"][i]
		if(i < len(separated_file_lists["anode"])):
			anodefile = topdir+dataset+separated_file_lists["anode"][i]

		if(pmtfile is not None):
			event_series['PMTTimestamp'] = parse_timestamp_from_filename(pmtfile) #in milliseconds since 00:00 (midnight)
			event_series['PMTSamplingPeriod'] = get_sampling_period_from_file(pmtfile) #ns
		else:
			event_series['PMTTimestamp'] = None
			event_series['PMTSamplingPeriod'] = None
		if(anodefile is not None):
			event_series['AnodeTimestamp'] = parse_timestamp_from_filename(anodefile) #in milliseconds since 00:00 (midnight)
			event_series['AnodeSamplingPeriod'] = get_sampling_period_from_file(anodefile) #ns
		else:
			event_series['AnodeTimestamp'] = None
			event_series['AnodeSamplingPeriod'] = None

		#load the anode file
		if(anodefile is not None):
			d = pd.read_csv(anodefile, header=None, skiprows=20, names=['ts','0','1'], encoding='iso-8859-1')
			event_series['Channels'] = ["glitch", "anode"]
			event_series['ChannelTypes'] = ["glitch", "anode"]
			data_map = [d['0'].to_numpy(), d['1'].to_numpy()] 
			event_series['Data'] = data_map
			baseline_subtract_1(event_series)

			event_series['Data'][1] = gaussian_filter(event_series['Data'][1], filter_tau/float(event_series['AnodeSamplingPeriod']))

			#calculate amplitude and integral
			#put processed quantities like amplitude and integral in a reduced series
			amplitudes, taus, peakidx, integrals = get_basic_waveform_properties(event_series, fit_amplitude_threshold, anode_integration_window, zero_amplitude_threshold) #finds tau's if relevant

			reduced_series['GlitchAmplitude'] = amplitudes[0] #in mV, negative or positive
			reduced_series['AnodeAmplitude'] = amplitudes[1] #in mV, negative or positive
			reduced_series['GlitchTau'] = taus[0] #time constant of exponential fit
			reduced_series['AnodeTau'] = taus[1] #time constant of exponential fit
			reduced_series['GlitchPeakidx'] = peakidx[0] #index referencing event_series['Data']
			reduced_series['AnodePeakidx'] = peakidx[1] #index referencing event_series['Data']
			reduced_series['GlitchIntegral'] = integrals[0] #mV*us
			reduced_series['AnodeIntegral'] = integrals[1] #mV*us
		else:
			reduced_series['GlitchAmplitude'] = None #in mV, negative or positive
			reduced_series['AnodeAmplitude'] = None #in mV, negative or positive
			reduced_series['GlitchTau'] = None #time constant of exponential fit
			reduced_series['AnodeTau'] = None #time constant of exponential fit
			reduced_series['GlitchPeakidx'] = None #index referencing event_series['Data']
			reduced_series['AnodePeakidx'] = None #index referencing event_series['Data']
			reduced_series['GlitchIntegral'] = None #mV*us
			reduced_series['AnodeIntegral'] = None #mV*us

		#load the pmt file
		if(pmtfile is not None):
			d = pd.read_csv(pmtfile, header=None, skiprows=20, names=['ts','0','1'], encoding='iso-8859-1')
			event_series['Channels'] = ["pmt1", "pmt2"]
			event_series['ChannelTypes'] = ["pmt", "pmt"]
			data_map = [d['0'].to_numpy(), d['1'].to_numpy()] 
			event_series['Data'] = data_map
			#baseline_subtract_2(event_series)
			#calculate amplitude and integral
			#put processed quantities like amplitude and integral in a reduced series
			amplitudes, taus, peakidx, integrals = get_basic_waveform_properties(event_series, fit_amplitude_threshold, pmt_integration_window, zero_amplitude_threshold) #finds tau's if relevant

			
			reduced_series['PMT1Amplitude'] = amplitudes[0] #in mV, negative or positive
			reduced_series['PMT2Amplitude'] = amplitudes[1] #in mV, negative or positive
			reduced_series['PMT1Peakidx'] = peakidx[0] #index referencing event_series['Data']
			reduced_series['PMT2Peakidx'] = peakidx[1] #index referencing event_series['Data']
			reduced_series['PMT1Integral'] = integrals[0] #million electrons
			reduced_series['PMT2Integral'] = integrals[1] #million electrons
			reduced_series['PMT1std'] = np.std(event_series['Data'][0])
			reduced_series['PMT2std'] = np.std(event_series['Data'][1])
		else:
			reduced_series['PMT1Amplitude'] = None #in mV, negative or positive
			reduced_series['PMT2Amplitude'] = None #in mV, negative or positive
			reduced_series['PMT1Peakidx'] = None #index referencing event_series['Data']
			reduced_series['PMT2Peakidx'] = None #index referencing event_series['Data']
			reduced_series['PMT1Integral'] = None #million electrons
			reduced_series['PMT2Integral'] = None #million electrons
			reduced_series['PMT1std'] = None
			reduced_series['PMT2std'] = None


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
	datatopdir = "/p/lustre1/angelico/hv-test-chamber/ControlRun0/"
	datasets = {0: "8-13-21/"}

	#check that the datasets are indexible
	print("Checking for datasets in top directory: " + datatopdir)
	to_remove = [] #list of keys to remove due to not existing
	for dno in datasets:
		print(str(dno)+": " + datasets[dno] + "\t\t", end = '')
		isdir = os.path.isdir(datatopdir+datasets[dno]+"/")
		if(isdir):
			numfiles = len([_ for _ in os.listdir(datatopdir+datasets[dno]+"/")])
			print("True, with " + str(numfiles) + " files")
		else:
			print("False")
			to_remove.append(dno)
			

	for k in to_remove:
		del datasets[k]
	

	#for each dataset, what was the voltage applied to
	#the rogowski electrod. NOTE: this dict takes into account
	#the bleed and filter resistors in the filter box
	voltage_dict = {0:0}

	sets_to_process = [0]
	nevents = 95000
	nevents_per_file = 1000
	for no in sets_to_process:
		if(no in datasets.keys()):
			topdir = datatopdir
			dataset = datasets[no]
		else:
			continue
	
		print("On dataset " + str(topdir+dataset))
		rogowski_voltage = voltage_dict[no]
		reduce_and_process(dataset, topdir, nevents, nevents_per_file, rogowski_voltage)
	
