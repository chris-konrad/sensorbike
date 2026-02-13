# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 15:40:58 2025

Example script showing how to use this package to load and filter sensor data
captured from the Gazelle/Bosch Balance Assist Bicycle.

This requires a data folder with the structure explained in the README

@author: Christoph M. Konrad
"""

import datetime as dt
import os 
from sensorbike.bikedata import InstrumentedBicycleData, read_yaml
from sensorbike.ukf import parse_filter_settings


def main():
    
    cfd = os.path.dirname(os.path.abspath(__file__))
    config = read_yaml(os.path.join(cfd, 'example_full-pipeline_config.yaml'))
    
    #parse filter settings.
    filter_settings = parse_filter_settings(config['filter_settings'])
    
    #create a bicycle data object
    bikedata = InstrumentedBicycleData(
        os.path.join(cfd, config['general']['dir_base']),
        config['general']['experiment_name'],
        config['general']['trial_name'],
        config['general']['filename_can'],
        config['general']['gnss_position_params'],
        config['general']['t_s'],
        filter_settings=filter_settings)
     
    #define a begin and a end time to crop the data
    t_begin = dt.datetime(2024, 8, 7, 16, 23, 0, tzinfo=dt.timezone.utc)
    t_end = dt.datetime(2024, 8, 7, 16, 23, 30, tzinfo=dt.timezone.utc)
    
    #load the raw data sensor. Performs time synchronization.
    data_raw = bikedata.load_raw(t_begin=t_begin, t_end=t_end, plot=True)

    #Run the Kalman Filter to estimate the bicycle states from the raw sensor data
    states_filtered = bikedata.get_bicyclestates_filtered(plot=True)


if __name__ == "__main__":
    main()