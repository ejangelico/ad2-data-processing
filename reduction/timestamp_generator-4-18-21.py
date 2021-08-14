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
from datetime import datetime



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
		separated_timestamps[pref] = [parse_timestamp_from_filename(f) for f in separated_file_lists[pref]]
		if(len(separated_timestamps[pref]) == 0): 
			continue 
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
	mean_buffer_duration = 2 #us
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


#TRICKY PART: only flagged by "daychange = True".
#because there is often an interface
#between 23:59 and 00:00, this usually indicates
#an increment in the day, rather than 00:00 coming
#before 23:59. We will call 6am the cutoff, where 
#any time before 10am is assumed to be the next day
#(this needs to be changed if someone starts a new
#dataset before 10am)
#global_reference and date_of_data are assumed to be datetime.datetime objects
def convert_timestamps_to_realtime(separated_timestamps, global_reference, date_of_data, daychange = True):
    this_dataset_zero_time = (date_of_data - global_reference).total_seconds() * 1000.0 #milliseconds
    one_day = 24*60*60*1000 #in milliseconds, one day, for doing daychange math
    daychange_time = 10*60*60*1000 #10:00 am in milliseconds
    sep_times_h, sep_times_m, sep_times_s = {}, {}, {}
    for pref in separated_timestamps:
        stamps = np.array(separated_timestamps[pref])
        if(len(stamps) == 0):
            sep_times_h[pref] = []
            sep_times_m[pref] = []
            sep_times_s[pref] = []
            continue
        adjusted_stamps = np.array([])
        if(daychange):
            for time in stamps:
                if(time < daychange_time):
                    adjusted_stamps.append(time + one_day)
                else:
                    adjusted_stamps.append(time)
        else:
            adjusted_stamps = stamps
        zeroed = adjusted_stamps + this_dataset_zero_time #numpy array fast math
        sep_times_h[pref] = zeroed/(1000*60*60.0)
        sep_times_m[pref] = zeroed/(1000*60.0)
        sep_times_s[pref] = zeroed/(1000.0)
        
    return sep_times_h, sep_times_m, sep_times_s


def reduce_and_process(dataset, topdir, rogowski_voltage):

	
	#the part of the filename before the timestamp. 
	#used to distinguish the two oscilloscopes
	file_prefixes = ["pmt", "anode"]
	

	#load dataset
	t0 = time.time()
	separated_file_lists = get_separated_file_lists(topdir+dataset, file_prefixes)
	print("Separation by prefix took " + str(time.time() - t0))

	t0 = time.time()
	separated_timestamps, separated_file_lists = get_separated_timestamps(separated_file_lists)

	print("Saving now...")

	#convert to hours, mins, seconds since the run started
	#get the time of the dataset from the filename/directory name
	daystr = dataset.split('/')[0]
	#unfortunately, run1 years are '21' instead of '2021'
	if(len(daystr.split('-')[-1]) == 4):	
		daystrf = "%m-%d-%Y"
	else:
		daystrf = "%m-%d-%y"
	
	timestr = (dataset.split('/')[1]).split('-')[-1]
	#add a colon
	timestr = timestr[:-2] + ":" + timestr[-2:]	

	run_reference = datetime.strptime(daystr + " " + timestr, daystrf + " " + "%H:%M")
	dataset_date = datetime.strptime(daystr, daystrf)
	#this one you need to be careful and know info on the dataset, 
	#see comment by "convert_timestamps_to_realtime"
	daychange = False
	sep_times_h, sep_times_m, sep_times_s = convert_timestamps_to_realtime(separated_timestamps, run_reference, dataset_date, daychange)

	ds = pd.Series()
	ds['voltage'] = rogowski_voltage
	ds['dataset'] = dataset
	ds['datetime'] = run_reference
	ds['pmt_timestamps'] = separated_timestamps['pmt']
	ds['pmt_times_s'] = sep_times_s['pmt']
	ds['anode_timestamps'] = separated_timestamps['anode']
	ds['anode_times_s'] = sep_times_s['anode']
	return ds 	
		
	


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
	datasets_r2 = {26:"2-22-2021/cosmics-1800/",\
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


	#for each dataset, what was the voltage applied to
	#the rogowski electrod. NOTE: this dict takes into account
	#the bleed and filter resistors in the filter box
	voltage_dict = {2: 2227, 3:2227, 4:205, 5:395.79, \
				   6:6654, 7:9859, 8:9994, 9:9994, 10:7439, \
				   11:9994, 12:15193, 13:15193, 14:15193, \
				   15:19967, 16:19967, 17:20167, 18:24950, \
				   19:24950, 20:27251, 21:27251, 25: 210, 26:210, \
				   27: 13975, 28:13975, 29:208, 30:208, 31:10369, \
				   32:22284, 33:24756, 34:24756, 35:24756, 36:17346, \
				   37:19804}

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
	df = pd.DataFrame()
	for no in sets_to_process:
		if(no in datasets_r1.keys()):
			topdir = datatopdir_r1
			dataset = datasets_r1[no]
		elif(no in datasets_r2.keys()):
			topdir = datatopdir_r2
			dataset = datasets_r2[no]
		else:
			continue
	
		print("On dataset " + str(topdir+dataset))
		rogowski_voltage = voltage_dict[no]
		ds = reduce_and_process(dataset, topdir, rogowski_voltage)
		df = df.append(ds, ignore_index=True)
	

	df.to_hdf("/p/lustre1/angelico/hv-test-chamber/timestamps-4-18-21.h5", key='raw')
	#wf_output_df.to_hdf(topdir+dataset+"rawhdf/"+str(i+1)+".h5", key='raw')
	#reduced_output_df.to_hdf(topdir+dataset+"reduced/"+str(i+1)+".h5", key='raw')
	#print("Done")
	#reinitialize
	#wf_output_df = pd.DataFrame()
	#reduced_output_df = pd.DataFrame()
