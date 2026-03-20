# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 15:40:58 2025

Example script showing how to use this package to load and filter sensor data
captured from the Gazelle/Bosch Balance Assist Bicycle.

The example is designed to work with the data collected for the Stochastic Balancing Rider 
available at https://doi.org/10.4121/f881dd80-b9f5-4322-9fd5-192034c9717f

Usage:
python example_full-pipelin.py /path/to/toplevel/data/directory

For help, type:
python example_full-pipelin.py --help

@author: Christoph M. Konrad
"""
import argparse
import datetime as dt
import os 

import matplotlib.pyplot as plt

from sensorbike.bikedata import InstrumentedBicycleData, read_yaml
from sensorbike.ukf import get_yaml_filter_settings
from sensorbike.geometry import get_geometry_params


def parse_args():
    parser = argparse.ArgumentParser(description="Run a minimal sensorbike full processing pipeline demo. See example_full-pipeline_config.yaml for configuration.")
    parser.add_argument("--datadir", help="Path to the toplevel directory of the stochastic balancing rider dataset. Get the dataset from https://doi.org/10.4121/f881dd80-b9f5-4322-9fd5-192034c9717f.")
    return parser.parse_args()


def main():
    args = parse_args()

    cfd = os.path.dirname(os.path.abspath(__file__))
    config = read_yaml(os.path.join(cfd, 'example_full-pipeline_config.yaml'))
    
    #parse filter settings.
    filter_settings = get_yaml_filter_settings(config['filter_settings'])
    
    # Geometry parameters (position of GNSS and IMU). In the stochastic balancing rider dataset, no antenna pole was used.
    # Choose 'interaction2024' for data captured with the antenna pole and 'zigzag2024' without antenna pole.
    geometry_params = get_geometry_params('zigzag2024')

    #create a bicycle data object
    bikedata = InstrumentedBicycleData(
        os.path.join(args.datadir, config['general']['subdir_data_raw']),
        config['general']['experiment_name'],
        config['general']['trial_name'],
        filename_can=config['general']['filename_can'],
        geometry_params=geometry_params,
        filter_settings=filter_settings)
     
    #define a begin and a end time to reduce the runtime of this demo.
    t_begin = dt.datetime.strptime(config['general']['t_begin'], "%d.%m.%Y %H:%M:%S").replace(tzinfo=dt.timezone.utc)
    t_end = dt.datetime.strptime(config['general']['t_end'], "%d.%m.%Y %H:%M:%S").replace(tzinfo=dt.timezone.utc)

    #load the raw data sensor. Performs time synchronization.
    data_raw = bikedata.load_raw(t_begin=t_begin, t_end=t_end, plot=True)

    #Run the Kalman Filter to estimate the bicycle states from the raw sensor data
    states_filtered = bikedata.get_bicyclestates_filtered(plot=True)

    plt.show(block=True)

if __name__ == "__main__":
    main()