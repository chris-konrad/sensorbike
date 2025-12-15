# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 17:54:09 2025

canbus
------

Module for decoding data logs from the CAN bus. 

Currently only supports data logged with the CAN Edge 2 logger. The CAN
CL2000 is not supported 

TODO: Copy the CL2000 processing code from Annas Repo to here.

This code is copied from the bicycle_loc_and_state repo created by
Anna Marbus during her Master's Thesis. Original code by Anna Marbus.
Copying and modifications by Christoph Konrad.
https://repository.tudelft.nl/record/uuid:092f3b70-2d97-436e-b193-139a593e09c7

@author: Anna Marbus
@author: Christoph M. Konrad
"""


import os
import asammdf

def process_can_edge(logfiles, databases):
    """
    This function processes CAN log files using DBC files to decode the 
    messages.
    
    ONLY SUPPORTS LOGS CREATED BY THE CAN Edge 2 LOGGER!

    Parameters
    ----------
    
    logfiles : list 
        List of paths to the MDF log files.
    databases : dict
        Dictionary containing the bus types as keys and a list of tuples with 
        DBC file paths and corresponding channel numbers as values.
    
    Returns
    -------
    df_can_edge : pandas.dataframe
        Dataframe of the can decoded can log
    """
    # Concatenate the MDF log files
    mdf = asammdf.MDF.concatenate(logfiles)
    
    # Extract bus logging data using the specified databases
    mdf_scaled = mdf.extract_bus_logging(databases)
    
    # Covert to dataframe. use_interpolation enforces a joint time grid. 
    df_can_edge = mdf_scaled.to_dataframe(use_interpolation=True,
                                          time_as_date=True)
    
    return df_can_edge


def verify_filepath_dbc(filepath):
    """ Check if the given filepath points to a .dbc file.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Can't find file {filepath}.")
    elif not filepath.lower().endswith('.dbc'):
        raise TypeError(f"File {filepath} is not a .dbc file.")
    else:
        return filepath
    

def verify_filepath_mf4(filepath):
    """ Check if the given filepath points to a .mf4 file.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Can't find file {filepath}.")
    elif not filepath.lower().endswith('.mf4'):
        raise TypeError(f"File {filepath} is not a .mf4 file.")
    else:
        return filepath
    

def list_canlogs(directory, verbose=False):
    """ List all CAN log files (ending with .mf4) in 
    a given directory and its subdirectories.

    Parameters
    ----------
    directory : str
        The directory to look in.
    verbose : bool, optional
        Print log file list, default is False.
    
    Returns
    -------

    """
    logfiles = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.mf4'):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, directory)
                logfiles.append(relative_path)
    logfiles = sorted(logfiles)
    
    if verbose:
        print(f'Found {len(logfiles)} CAN logs in {directory}:')
        for i, f in enumerate(logfiles):
            print(f'    {i:<3}: {f}')
    
    return logfiles


def decode_parquet(logfile):
    """ Decode a parquet CAN log file.

    Parameters
    ----------
    logfile : str
        Filepath of the CAN log file

    Returns
    -------
    df : DataFrame
        Logs
    """

    df = pd.read_parquet(logfile)

    keys = {
        'gyro_z_rad/s': 'gyro_z',
        'gyro_y_rad/s': 'gyro_y',
        'gyro_x_rad/s': 'gyro_x', 
        'accel_z_m/s2': 'accel_z', 
        'accel_y_m/s2': 'accel_y', 
        'accel_x_m/s2': 'accel_x', 
        'yaw_rad': 'yaw', 
        'pitch_rad': 'pitch', 
        'roll_rad': 'roll', 
        'wheelspeed_rear_rev/s': 'ws_rear', 
        'steer_deg': 'LWS_ANGLE', 
        'steer_rate_deg/s': 'LWS_SPEED'}
    
    df.rename(columns=keys, inplace=True)

    return df


