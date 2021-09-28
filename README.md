# ad2-data-processing
Gratta group code for processing data from analog discovery 2 (AD2)


### Note on installation
I've used a system of progress bars in the Dataset.py class that will only work properly in jupyter notebook with the following installation steps performed

```
pip install tqdm ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

# Data structure and analysis flow


### Dataset
This class organizes functions related to loading data, plotting raw waveforms, and reduction of data. It's attributes hold information about the location of datasets (folders), stores information about which events from one scope correspond to events from another scope based on timestamp, stores filename information, and contains pandas dataframes for storing raw waveforms as well as attributes (integral, amplitude, noise, etc) of reduced data. One can save by pickling the entire dataset object, or saving the dataframes to .h5 files. 

See "analysis/Data\ loading\ tester.ipynb" for a notebook example of how to begin processing data. 

### Reduction
Data reduction code is still in the works, but the planned attributes for the reduced Dataframe are 

`wave_df` pandas dataframe for raw waveforms with columns : pmtTimestamp, pmtSamplingPeriod, anodeTimestamp, anodeSamplingPeriod, Channels (channel names list), ChannelTypes (what kind of detector), pmt0-Data, pmt1-Data, anode0-Data, anode1-Data 

`reduced_df` pandas dataframe for reduced data with columns: GlitchAmplitude, AnodeAmplitude, GlitchTau, AnodeTau, GlitchPeakidx, AnodePeakidx, GlitchIntegral, AnodeIntegral, PMT1Amplitude, PMT2Amplitude, PMT1Peakidx, PMT2Peakidx, PMT1Integral, PMT2Integral, PMT1std, PMT2std, RogowskiVoltage


