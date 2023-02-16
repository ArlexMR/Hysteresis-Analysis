import sys
import pandas as pd
from importlib import reload
from src.get_data import read_precip, read_QTC
from src.preprocessing import clean_precip





class DataPreprocessor:

    def __init__(self, path_precip, path_QTC):
        
        self.data_Source = [path_precip, path_QTC]
        # read data as is
        self.raw_precip = read_precip(path_precip)
        # read data as is
        self.raw_QTC    = read_QTC(path_QTC)

    
    def clean_data(self):
        # resample to hourly values
        from src.preprocessing import clean_precip

        self.clean_precip = clean_precip(self.raw_precip)
        #remove outliers, interpolate and unify Cond and turb
        # self.clean_QTC    = clean_QTC(self.raw_QTC)

        print("Data Cleaned")


    def get_events(self):
        #Apply splitting criteria and drop tails
        self.QTC_with_event_flags = split_events(self.clean_QTC)

        n_events = self.QTC_with_event_flags['Event_ID'].unique()

        print("{} events identified".format(n_events))

    def refine_splitting(self, event_mapper):
        # modify event flags according to the mapper
        self.QTC_with_event_flags = refine_splitting(self.QTC_with_event_flags, event_mapper) 

    def get_old_new_water(self):

        self.old_new_water_DF = apply_emua(self.clean_QTC)

    def get_table(self):
        
        input_ = join_cleaned_P_and_QTC()
        self.table = calc_event_metrics()

    def plot_PCA(self):
        pass

    def plot_scatter():
        pass




    