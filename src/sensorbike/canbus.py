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

