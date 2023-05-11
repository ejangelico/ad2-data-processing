import numpy as np 
import os 
import pandas as pd

#this class takes in the hdf5 pandas dataframes from NGMBinaryFile output processing and 
#pre processing the events for integrating with the AD2 data. The operations include
#1. reconstructing timestamps in UTC time and microseconds relative to the clock full timestamp counter
#2. combines and organizes files into reasonable sizes
#3. formats the output dataframe to be consistent with the Dataset.py object
#4. Deletes empty events, i.e. bugginess if both channels have no amplitude relative to baseline. 
