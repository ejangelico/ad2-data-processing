# ad2-data-processing
Gratta group code for processing data from analog discovery 2 (AD2)


# Reduced data, structure of input and output

### Input
Datasets are stored on the LLNL cluster inside folders that identify them, containing 100s to 100ks of .csv output files from the analog discoveries. This directory is indexed within a dictionary, for mass processing of data; these are the dictionaries seen in the __main__ function of reduce data scripts. An accompanying dictionary hold info on the high voltage settings. Ultimately, the dictionaries are looped over and a data reduction function is called with the folder location information, the number of events to process in total, and the number of events to save per output hdf5 file. 

The directories containing the data will have a massive number of files in the "timestamp" output format from the AD2s, with a prefix that indicates which AD2 they originate from. For example, in Run1 and Run2 of the HV test chamber, the AD2s were "pmt" and "anode", where the "anode" AD2 files have two channels representing an amplified signal from rogowski anode and a glitch signal from the HV filter. 

### Reduction procedure
The first step in the reduction process is to index the dataset directory file-list and separate by the file prefix. The result is a two (or more, number of AD2s) element dictionary 
`separated_file_lists = {"prefix1": [filename, filename, ...], "prefix2": [filename, filename, ...]}` 

A loop with a cutoff on the number of total events initiates and performs the following reduction tasks:
- Parse the timestamp from the filename
- Get event metadata from the header of the file (the sampling period, for example)
- Get waveform properties, presently amplitude, exponential decay timeconstant, peak time index, and integral

### Output

The processed data is organized into two pandas dataframes, one for the raw waveforms alone and another for reduced data alone. 

`wf_output_df = pd.DataFrame()`:
`event_series = pd.Series`: with elements PMTTimestamp, PMTSamplingPeriod, AnodeTimestamp, AnodeSamplingPeriod, Channels (channel names list), ChannelTypes (what kind of detector), Data (`event_series['Data'] = [[channel 0 digitized signal list], [channel 1 digitized signal list]]`), RogowskiVoltage

`reduced_output_df = pd.DataFrame()`:
`reduced_series = pd.Series`: with elements, GlitchAmplitude, AnodeAmplitude, GlitchTau, AnodeTau, GlitchPeakidx, AnodePeakidx, GlitchIntegral, AnodeIntegral, PMT1Amplitude, PMT2Amplitude, PMT1Peakidx, PMT2Peakidx, PMT1Integral, PMT2Integral, PMT1std, PMT2std, RogowskiVoltage

These are then saved as .h5 files. 

