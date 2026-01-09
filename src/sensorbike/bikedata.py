# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 08:55:01 2025

bikedata
--------

A module to load and filter data collected on the instrumented 
Gazelle/Bosch Balance Assist Bicycles. 

This module was originally part of the rider-bicycle-control identification 
study by Christoph Konrad. The present code is copied and modifed 
to work as a standalone module from rcid.utils.

@author: Christoph M. Konrad
"""

# external imports
import re
import os
import warnings
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import sympy as sm
import sympy.physics.mechanics as me

from scipy.signal import correlate
from sklearn.linear_model import RANSACRegressor
from pathlib import Path

# own imports
from trajdatamanager.datamanager import RTKLibGNSSManager, Track, DataManager
from trajdatamanager.utils import to_finite

# local imports
from sensorbike.ukf import filter_dynamic, get_default_filter_settings
from sensorbike.canbus import process_can, decode_parquet, verify_filepath_dbc, list_decoded_canlogs

class InstrumentedBicycleData():
    """
    A class to load and filter data collected on the instrumented 
    Gazelle/Bosch Balance Assist Bicycles. 
    
    Uses data from the IMU, steerencorder and speedometer captured from 
    the CAN bus as well as data collected from the SWIFTNAV Piksi Multi 
    RTK GNSS mounted on the rack of the bicycle.
    """
    
    TZ = "UTC"

    def __init__(
        self,
        dir_base,
        experiment_name,
        trial_name,
        gnss_position_params = dict(h_gnss=0.94, l_gnss=0.17),
        filename_can = None,
        t_s = 0.01,
        subdir_bike_gnss_solution=None,
        subdir_bike_gnss_report=None,
        subdir_bike_can=None,
        rotation = 67,
        reference_location = [51.999370, 4.370451],
        filter_settings = None,
        gnss_data_settings = {},
        can_data_settings = {},
        transfrom_to_rwcontactpoint = True,
        reference_frame = 'E', 
    ):
        """
        Create a InstrumentedBicycleData object.

        Required file system
        --------------------
        
        The raw data is assumed to be organized in the following file system. 
        The assumed file system can be adapted with the subdir_ properties.

        base_directory
        |---experiment_name
            |---bike-gnss
            |   |---solution
            |   |       |---0000-00000.pos
            |   |       |---...
            |   |       |---0000-000X0.pos
            |   |---report
            |           |---0000-00000
            |           |---...
            |           |---0000-000X0
            |---can-logger
                |---filename_dbc
                |---filename_can


        Parameters
        ----------
        dir_base : str
            Base directory of the data.
        experiment_name : str
            Name of the experiment / subdirectory of the data.
        trial_name : str
            An arbitrary name for this trial.
        gnss_position_params : dict, optional
            A dictionary describing the position of the GNSS antenna:
                hb : height [m] of the GNSS antenna above ground when the bicycle is upright. Default is 0.94 m
                lb : horizontal distance [m] between GNSS antenna and the rear wheel contact patch. Default is 0.17 m
            The default corresponds to the setup used for the zigzag experiment (without antenna post).
        filename_can : str, optional
            The filname (or sub-path) of a specific (coded or decoded) CAN log file. 
            Can be a decoded log in .parquet format or coded logs in .MF4/.txt format. If 
            .MF4/.txt, the CAN bus definition must be given as well (can_data_settings['dbc_file'] = 'path//to//definition.dbc'). 
            If not specified, all CAN files in subdir_bike_can are loaded and stitched together.
            Example for a single file: '00001024//00000001.MF4'
        t_s : float, optional
            Desired sample period of the data after loading. May be as low as the CAN sample time. Set to None to 
            automatically infer the CAN sample time. Default is 0.01 s
        subdir_bike_gnss_solution : str, optional
            Subdirectory of the gnss solution. The default is 
            experiment_name//bike-gnss//solution.
        subdir_bike_gnss_report : str, optional
            Subdirectory of the gnss report. The default is 
            experiment_name//bike-gnss//report.
        subdir_bike_can : str, optional
            Subdirectory of the can-log. The default is 
            experiment_name//can-logger.
        rotation : float, optional
            Rotation in deg of the local reference frame w.r.t North. 
            The default is 67.
        reference_location : list, optional
            Coordinates of the origin of of the local reference coordinate 
            system. The default is [51.999370, 4.370451].
        filter_settings : dict, optional
            Dictionary of Unscented Kalman Filter settings. 
            See the output of sensorbike.ukf.get_default_filter_settings()
            for details on the expected format. If None, 
            the output of this function is used. The default is None.
        transfrom_to_rwcontactpoint : bool, optional
            Transform GNSS locations to the rear-wheel contact point of the 
            bicycle. The default is True.
        reference_frame : str, optional
            The frame to represent the data in. Choose from 'E' and 'N'. The 
            N-frame is the frame with x and y directions fixed to the ground and z pointing
            into the ground (as common in bicycle dynamic research and used for the Carvallo-Whipple model). 
            The E-frame is the frame with x and y directions fixed to the ground 
            and z pointing upwards (as common in traffic engineering and used for 
            cyclistsocialforces). The default is 'E'. 
        """

        # set up paths and directories

        if subdir_bike_gnss_solution is None:
            subdir_bike_gnss_solution = os.path.join(
                experiment_name, "bike-gnss", "solution"
            )

        if subdir_bike_gnss_report is None:
            subdir_bike_gnss_report = os.path.join(
                experiment_name, "bike-gnss", "report"
            )

        if subdir_bike_can is None:
            subdir_bike_can = os.path.join(experiment_name, "can-logger")
            
        # properties
        self.dir_base = dir_base
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.filename_can = filename_can
        self.subdir_bike_gnss_solution = subdir_bike_gnss_solution
        self.subdir_bike_gnss_report = subdir_bike_gnss_report
        self.subdir_bike_can = subdir_bike_can
        self.filename_can = filename_can
        self.t_s = t_s
        self.name = f"{self.experiment_name}/{self.trial_name}"
        self.transfrom_to_rwcontactpoint = transfrom_to_rwcontactpoint
        self.is_filtered = False
        
        # reference frame
        if reference_frame not in ['E', 'N']:
            raise ValueError(f"The reference frame must be 'E' or 'N', instead it was '{reference_frame}'.")
        self.desired_reference_frame = reference_frame
        self.current_reference_frame = 'E'

        # rotation and reference location
        self.rotation = rotation
        self.reference_location = reference_location
        
        # bicycle geometry
        self.bike_geom = InstrumentedBikeGeometry(gnss_position_params)

        # data
        self.bike_gnss_data = None
        self.bike_dynamic_data = None
    
        # measurement filter
        if filter_settings is None:
            self.filter_settings = get_default_filter_settings()
        else:
            self.filter_settings = filter_settings
        
        # additional settings (i.e. keyword arguments for BalanceAssistDataManager and RTKLibDataManager)
        self.gnss_data_settings = gnss_data_settings
        self.can_data_settings = can_data_settings
        
        # data properties
        self.trk = None
        self.trk_filtered = None
        
    
    def _transform_gnss_to_rearwheel(self, trk):
        """
        Transform the gnss measurements to the rearwheel contact patch. 

        Parameters
        ----------
        trk_gnss : trajdatamanager.Track
            A track object holding the loaded can data.

        Returns
        -------
        trk_gnss : trajdatamanager.Track
            The updated track object holding the loaded can data.

        """
       
        x, y = self.bike_geom.transform_gnss2rwcp(trk['x_gnss'], 
                                                    trk['y_gnss'], 
                                                    trk['psi_can'], 
                                                    trk['phi_can'])
        
        trk['x_gnss'] = x
        trk['y_gnss'] = y
        
        trk = update_yaw(trk, keys=('x_gnss', 'y_gnss', 'psi_gnss'))

        self.current_reference_frame = 'E'
        
        return trk
    
    def _combine_datasets(self, trk_gnss, trk_can):
        """
        Combine the datasets from the two sensors into one:
            - Crop to the same timespan
            - Put into the same timeframe
            - Align yaw measurements
            - Create joint Track object.

        Parameters
        ----------
        trk_gnss : trajdatamanager.Track
            A track object holding the loaded can data.
        trk_can : trajdatamanager.Track
            A track object holding the loaded can data.

        Returns
        -------
        trk : trajdatamanager.Track
            A track object holding the combined data.

        """

        def _to_continous_angles(trk):
            for k in trk.data_feature_keys:
                for kk in ['psi', 'phi', 'delta']:
                    pattern = rf"(?<!d){kk}"
                    if re.findall(pattern, k):
                        trk[k] = to_continous_angle(trk[k])
            return trk
        
        def _limit_angles(trk):
            for k in trk.data_feature_keys:
                for kk in ['psi', 'phi', 'delta']:
                    pattern = rf"(?<!d){kk}"
                    if re.findall(pattern, k):
                        trk[k] = limit_angle(trk[k])
            return trk
        
        # combine into one track
        t,  t_span = self._find_time_frame(trk_gnss, trk_can) 

        #convert to continous angles
        trk_can = _to_continous_angles(trk_can)
        trk_can.sample_at_times(np.r_[t, t[-1]+dt.timedelta(seconds=self.t_s)])

        t = trk_can.t
        t_span = (t[0], t[-1])

        trk_can.crop_to_timespan(t_span[0], t_span[1])
        trk_gnss.crop_to_timespan(t_span[0], t_span[1])

        #trk_can = _limit_angles(trk_can)
        
        data_gnss = expand_timebase(trk_gnss.data, trk_gnss.t, t)
        data_can = trk_can.data
        
        n = min(data_gnss.shape[0], data_can.shape[0]) #fix occasional numerical error
        data = np.c_[data_gnss[:n], data_can[:n]]
        t = t[:n]

        feat = [k + "_gnss" for k in trk_gnss.data_feature_keys] + \
               [k + "_can" for k in trk_can.data_feature_keys]

        metadata = {"meta_gnss": trk_gnss.metadata, 
                    "meta_can": trk_can.metadata}   
        
        trk = Track(self.name, 0, t, data, 
                    data_feature_keys = feat, 
                    yaw_feature_index = trk_gnss.yaw_feature_index, 
                    metadata = metadata)
        
        # align yaw
        #offset = np.nanmedian(trk['psi_can']-trk['psi_gnss'])
        #trk['psi_can'] = trk['psi_can'] - offset 

        # limit angles
        #trk = _limit_angles(trk)
        
        return trk
                    
        
    def _find_time_frame(self, trk_gnss, trk_can):
        """
        Find a suitable common time frame for data from different sensors.

        Returns
        -------
        trk_gnss : trajdatamanager.Track
            A track object holding the loaded can data.
        trk_can : trajdatamanager.Track
            A track object holding the loaded can data.
            
        Returns
        -------
        t : array
            Array of datetime timestamps forming the common time frame
        t_span : list
            List of (t_begin, t_end, t_s).

        """
        
        
        dt_gnss = np.median(np.diff(trk_gnss.t)).total_seconds()
        dt_can = np.median(np.diff(trk_can.t)).total_seconds()
        
        if self.t_s is None:
            self.t_s = dt_can * 2
        
        if self.t_s < dt_can:
            msg = (f"This module only supports datasets where twice the CAN sample "
                   f"rate exceeds or equals the requested sample rate "
                   f"2. Instead the double median can sample time was {2*dt_can:.6f} "
                   f"s and the requested sample time was {self.t_s:.6f} s.")
            raise ValueError(msg)
            
        if ((dt_gnss / self.t_s) - int(dt_gnss / self.t_s)) != 0.0:
            msg = (f"The median GNSS is not a multiple of the requested "
                   f"sample time t_s = {self.t_s:.6f} s. Instead it was "
                   f"{dt_gnss:.6f} s")
            
        t_begin = max(trk_gnss.get_begin_allfinite()[0], trk_can.get_begin_allfinite()[0])
        i_gnss_begin = np.argwhere(trk_gnss.t >= t_begin).flatten()[0]
        t_begin = trk_gnss.t[i_gnss_begin]
        
        t_end = min(trk_gnss.get_end_allfinite()[0], trk_can.get_end_allfinite()[0])
        i_gnss_end = np.argwhere(trk_gnss.t <= t_end).flatten()[-1]
        t_end = trk_gnss.t[i_gnss_end]
            
        t = np.array(pd.date_range(start=t_begin, end=t_end, freq=f"{self.t_s:.6f}s").to_pydatetime())

        # check that all gnss times are in the timeframe
        # hacky but efficient solution
        t_float = np.arange(t.size).astype(int) * 10**3
        tgnss64 = trk_gnss.t.astype('datetime64[ns]')
        t_float_gnss = (tgnss64[i_gnss_begin:i_gnss_end+1] - tgnss64[i_gnss_begin]) / np.timedelta64(int(self.t_s * 10**3), 'us')
        t_float_gnss = t_float_gnss.astype(int)
        check = np.all(np.isin(t_float_gnss, t_float))
        if not check:
            msg = ("Error in time frame! Check if GNSS data has time jitter.")
            raise ValueError(msg)
            
        return t, (t_begin, t_end, self.t_s)
    
    
    def _load_can(self):
        """
        Load data captured from the CAN-Bus of the bicycle.

        Returns
        -------
        trk_can : trajdatamanager.Track
            A track object holding the loaded can data.

        """

        # make paths and directorys
        dir_can_log = os.path.join(self.dir_base, self.subdir_bike_can)
        dir_bike_gnss_report = os.path.join(
            self.dir_base, self.subdir_bike_gnss_report
        )

        # create datamanager
        dataman = BalanceAssistLogDataManager(
            dir_can_log, dir_bike_gnss_report, self.bike_geom, **self.can_data_settings
        )

        # load the full track 
        if self.filename_can is None:
            ftypes = ['.parquet', '.mf4', '.txt']
            for ftype in ftypes:
                can_files = list(Path(dir_can_log).rglob(f"*{ftype}"))
                can_files = [str(p) for p in can_files]
                if can_files:
                    break
            if len(can_files) == 0:
                raise FileNotFoundError(f"Didn't find any CAN logs in {dir_can_log}! Searched for: {ftypes}")
        elif os.path.isfile(self.filename_can):
            can_files = [self.filename_can]
        else:
            FileNotFoundError(f"Can't find filename_can: {self.filename_can}")

        trk_can = dataman.load_track(can_files, self.filenames_bike_gnss)

        return trk_can
        

    def _load_gnss(self, t_begin=None, t_end=None, plot_results=False):
        """
        Load data measured by the SWIFTNAV Pixi Mutli GNSS.

        Parameters
        ----------
        t_begin : datetime.datetime, optional
            Crop the available data to begin at this time. The default is None.
        t_end : datetime.datetime, optional
            Crop the available data to end at this time. The default is None.

        Returns
        -------
        trk_gnss : trajdatamanager.Track
            A track object holding the loaded gnss data.

        """
        
        #Derive limits of the experiment area from measurements relative to the
        #target lights.
        #if x_limits is None:
        #    x_limits = np.mean(self.target_locations[:,0]) - np.array([28+8.7, 8.7])

        if plot_results:
            fig, ax = plt.subplots(1, 1)
            ax.set_title("Full GNSS data and extracted runs (rotated).")
            #plt.plot([x_limits[0], x_limits[0]], [-50, 50], color="orange")
            #plt.plot([x_limits[1], x_limits[1]], [-50, 50], color="orange")

        # built paths and directories
        dir_bike_gnss_sol = os.path.join(
            self.dir_base, self.subdir_bike_gnss_solution
        )

        # create a datamanager instance
        dataman = RTKLibGNSSManager(
            dir_bike_gnss_sol, reference_location=self.reference_location,
            **self.gnss_data_settings
        )

        # load the track
        seq = dataman.load_sequence()
        
        # crop
        if t_begin is None:
            t_begin = seq.t_begin
        if t_end is None:
            t_end = seq.t_end
        seq = seq.reduce_to_timespan(t_begin, t_end, criterion="overlap"
        )
        self.filenames_bike_gnss = [trk.track_id for trk in seq.tracks]
        trk_gnss = seq.serialize()
        trk_gnss.crop_to_timespan(t_begin, t_end)

        # rotate the track for convenience and plot again
        trk_gnss.rotate_xy(self.rotation, deg=True)
        if plot_results:
            trk_gnss.plot_xy(ax=ax, color="gray")

        return trk_gnss
    
    
    def load(self, t_begin=None, t_end=None, verbose=True, plot_data=True):
        """
        Load instrumented bicycle data.

        Parameters
        ----------
        t_begin : datetime.datetime, optional
            Crop the available data to begin at this time. The default is None.
        t_end : datetime.datetime, optional
            Crop the available data to end at this time. The default is None.
        verbose : bool, optional
            Verbose output. The default is True.
        plot_data : bool, optional
            Plot the results after filtering. The default is True.
            
        Returns
        -------
        trk : trajdatamanager.Track
            A track object holding the loaded trajectory data.
            
        """
        # run gnss data
        if verbose:
            print("Loading position data ...", end="")
        trk_gnss = self._load_gnss(t_begin=t_begin, t_end=t_end)
        if verbose:
            print("done!")
            
        # run bike dynamics data
        if verbose:
            print("Loading can data ...", end="")
        trk_can = self._load_can()
        if verbose:
            print("done!")
           
        # align time frames
        if verbose:
            print("Combine datasets ...", end="")
        trk = self._combine_datasets(trk_gnss, trk_can)
        if verbose:
            print("done!")
        
        # transform gnss to rear-wheel contact point
        if self.transfrom_to_rwcontactpoint:
            if verbose:
                print(("Transforming gnss data to rear-wheel "
                       "contact point ..."), end="")
            trk = self._transform_gnss_to_rearwheel(trk)
            # desired reference frame
            trk = self._transform_to_desired_referenceframe(trk)
            if verbose:
                print("done!")
                       
        self.trk = trk
        self.data_loaded = True

        #plotting
        if plot_data:
            if verbose:
                print("Plotting data ...", end="")
                axes_t, axes_xy = self.plot_data(self.trk)
            if verbose:
                print("done!")

        return self.trk
    

    def _transform_to_desired_referenceframe(self, trk):

        if self.current_reference_frame == 'E' and self.desired_reference_frame == 'N':
            trk = self.bike_geom.transform_E2N(trk)
            self.current_reference_frame = 'N'
            trk.metadata['reference_frame'] = 'N'
        elif self.current_reference_frame == 'N' and self.desired_reference_frame == 'E':
            trk = self.bike_geom.transform_N2E(trk)
            self.current_reference_frame = 'E'
            trk.metadata['reference_frame'] = 'E'
        
        return trk

    
    def apply_filter(self, measurement_noise_std=None, 
                           process_noise_std=None,
                           integration_method=None,
                           bicycle_parameter_dict=None,
                           plot_data=True,
                           plot_filter_details=False,
                           verbose=True):
        
        """
        Filter the data from the instrumented bike with anUnscented Kalman 
        Filter.

        Parameters
        ----------
        measurement_noise_std : array, optional
            Array of sensor noise variances for filtering in the order
            [std_x, std_y, std_psi, std_v, std_phi, std_delta, std_dpsi, 
             std_dphi, std_ddelta, std_dv]. If None, values are given 
            by sensorbike.ukf.get_default_filter_settings(). The default is 
            None.
        process_noise_std : array, optional
            Array of process noise variances for filtering in the 
            order [std_x, std_y, std_psi, std_v, std_phi, std_delta, std_dpsi, 
             std_dphi, std_ddelta, std_dv]. If None, values are given by 
            sensorbike.ukf.get_default_filter_settings(). The default is None.
        integration_method: str, optional
            The integration method for the integration of the system dynamics 
            in the prediction step. Can be 'backward euler' or 'midpoint'. 
            The default is 'midpoint'.
        bicycle_parameter_dict : dict, optional
            The dictionary of physical bicycle parameters for the prediction
            of the bicycle dynamics. The dictionary is expected to be in the 
            format defined by BicycleParameters (https://github.com/moorepants/
            BicycleParameters/tree/master). The parameters should be measured 
            from the actual bicycle used to collect the data. Uses
            meijaard2007_browser_jason per default which are not correct for the
            balance assist bicycle.
        verbose : bool, optional
            Verbose output. The default is True.
        plot_data : bool, optional
            Plot the results after filtering. The default is True.
        plot_filter_details : bool, optional
            A more detailed plot of the filter and smoother results useful for
            calibrating the filter. 

        Returns
        -------
        trk_filtered : trajdatamanager.Track
            A track object holding the filtered trajectory data.

        """
        
        if verbose:
            print("Running Unscented Kalman Filter ...", end="")

        keys_track = ["x_gnss", "y_gnss", "psi_gnss", "v_can", "phi_can", 
                      "delta_can", "dpsi_can", "dphi_can", "ddelta_can", "a_can"]
        keys_dict = ["x", "y", "psi", "v", "phi", "delta", "dpsi", "dphi", 
                     "ddelta", "a"]
        keys_out = ["x", "y", "psi", "v", "phi", 
                    "delta", "dpsi", "dphi", "ddelta", "a"]
        
        def _parse_settings(sname, sval):
            if sval is None:
                if sname in self.filter_settings.keys():
                    return self.filter_settings[sname]
                else:
                    msg = (f"Filter settings must provide '{sname}'!")
                    raise KeyError(msg)
            else:
                return sval
        
        R_std = _parse_settings('measurement_noise_std', 
                                  measurement_noise_std)
        Q_std = _parse_settings('process_noise_std', process_noise_std)
        int_method = _parse_settings('integration_method', integration_method)
        bparams = _parse_settings('bicycle_parameter_dict', 
                                  bicycle_parameter_dict)

        #the measurements are in the E frame. The bike model of the kalman
        #filter operates in the N frame -> transform
        data_dict = self.trk.to_dict(relative_time=self.trk.t_begin, 
                                     features=keys_track)
        data_dict = {k.split("_")[0]: v for k, v in data_dict.items()}

        if self.current_reference_frame == 'E':
            data_dict = self.bike_geom.transform_E2N(data_dict)
        
        data_dict["psi"] = to_continous_angle(data_dict["psi"])
        
        measurements = np.zeros((len(data_dict["t"]), len(keys_dict)))
        for i, k in enumerate(keys_dict):
            measurements[:,i] = data_dict[k]
        
        R = np.diag(R_std)
        Q = np.diag(Q_std) 
        
        data_filtered = filter_dynamic(measurements, R, Q, 
                                       integration_method=int_method,
                                       bicycle_parameter_dict=bparams,
                                       plot=plot_filter_details)
        
        #the smoothed results are in the N frame. The measurements
        #are assumed to be in the E frame -> transform
        #TODO: This should be able to process dicts.
        data_dict_filt = {}
        for i, k in enumerate(keys_out):
            data_dict_filt[k] = data_filtered[:,i]
 
        if self.desired_reference_frame == 'E':
            data_dict_filt = self.bike_geom.transform_N2E(data_dict_filt)
        data_dict_filt["psi"] = limit_angle(data_dict_filt["psi"])
        
        
        for i, k in enumerate(keys_out):
            data_filtered[:,i] = data_dict_filt[k]
        
        metadata_filtered = dict(self.trk.metadata)
        metadata_filtered['filter_settings'] = {
            'measurement_noise_std': R_std,
            'process_noise_std': Q_std,
            'integration_method': int_method,
            'bicycle_parameter_dict': bparams}
        
        self.trk_filtered = Track(self.name+'/filtered', 0, self.trk.t, 
                                  data_filtered, data_feature_keys=keys_out,
                                  metadata = metadata_filtered, 
                                  yaw_feature_index=2)
        
        if verbose:
            print("done!")
            
        if plot_data:
            if verbose:
                print("Plotting data ...", end="")
                axes_t, axes_xy = self.plot_data(self.trk)
            if verbose:
                print("done!")
    
        return self.trk_filtered
    
    
    def load_and_filter(self, 
                        t_begin=None, 
                        t_end=None, 
                        measurement_noise_std=None, 
                        process_noise_std=None,
                        integration_method=None,
                        bicycle_parameter_dict=None,
                        verbose=True, 
                        plot_data=True):
        """
        Load instrumented bicycle data and filter the loaded data with a
        Unscented Kalman Filter.

        Parameters
        ----------
        t_begin : datetime.datetime, optional
            Crop the available data to begin at this time. The default is None.
        t_end : datetime.datetime, optional
            Crop the available data to end at this time. The default is None.
        measurement_noise_std : array, optional
            Array of sensor noise variances for filtering in the order
            [std_x, std_y, std_psi, std_v, std_phi, std_delta, std_dpsi, 
             std_dphi, std_ddelta, std_dv]. If None, values are given 
            by sensorbike.ukf.get_default_filter_settings(). The default is 
            None.
        process_noise_std : array, optional
            Array of process noise noise variances for filtering in the 
            order [std_x, std_y, std_psi, std_v, std_phi, std_delta, std_dpsi, 
             std_dphi, std_ddelta, std_dv]. If None, values are given by 
            sensorbike.ukf.get_default_filter_settings(). The default is None.
        integration_method: str, optional
            The integration method for the integration of the system dynamics 
            in the prediction step. Can be 'backward euler' or 'midpoint'. 
            The default is 'midpoint'.
        bicycle_parameter_dict : dict, optional
            The dictionary of physical bicycle parameters for the prediction
            of the bicycle dynamics. The dictionary is expected to be in the 
            format defined by BicycleParameters (https://github.com/moorepants/
            BicycleParameters/tree/master). The parameters should be measured 
            from the actual bicycle used to collect the data. The parameters
            of the instrumented bicycle used by defaul are stored in 
            sensorbike.bicycleparameters.bike_with_rider. 
        verbose : bool, optional
            Verbose output. The default is True.
        plot_data : bool, optional
            Plot the results after filtering. The default is True.

        Returns
        -------
        trk_filtered : trajdatamanager.Track
            A track object holding the filtered trajectory data.

        """
    
        self.load(t_begin=t_begin, t_end=t_end, 
                  verbose=verbose, plot_data=False)
        self.apply_filter(measurement_noise_std=measurement_noise_std,
                          process_noise_std=process_noise_std,
                          integration_method=integration_method,
                          bicycle_parameter_dict=bicycle_parameter_dict,
                          plot_data=plot_data,
                          verbose=verbose)
        
        return self.trk_filtered
    
        
    def plot_data(self, plot_raw = True, plot_filtered=True):
        """
        Plot the loaded data over time and in the xy-plane.

        Parameters
        ----------
        plot_raw : bool, optional
            If available, plot the raw data. The default is True.
        plot_filtered : bool, optional
            If available, plot the filtered data. The default is True.

        Returns
        -------
        axes_t : array
            Array of axes of the time plot.
        axes_xy : TYPE
            Axis of the xy plot.
            
        """
        
        col_gnss = 'blue'
        col_can = 'green'
        col_fil = 'orange'
        
        marker = '.'
        markersize = 3
        linewidth = 0.5
        plot_kwargs = {"marker": marker, "linewidth": linewidth,
                       "markersize": markersize}
        
        fig_t, axes_t = plt.subplots(10,1, sharex=True)
        fig_xy, axes_xy = plt.subplots(1,1)
        
        if plot_raw and self.trk is not None:
            
            #gnss
            feat_gnss = ['x_gnss', 'y_gnss', 'psi_gnss', 'v_gnss']
            self.trk.plot(features=feat_gnss, axes = axes_t[[0, 1, 2, 8]],
                          plot_over_timestamps=True, label='gnss', 
                          color=col_gnss, **plot_kwargs)
            axes_xy = self.trk.plot_xy(ax=axes_xy, label='gnss', 
                                       color=col_gnss, **plot_kwargs)
            
            #can
            
            feat_can = ['psi_can', 'dpsi_can', 'delta_can', 'ddelta_can',
                        'phi_can', 'dphi_can', 'v_can', 'a_can']
            self.trk.plot(features=feat_can, axes = axes_t[[2, 3, 4, 5, 
                                                            6, 7, 8, 9]],
                          plot_over_timestamps=True, label='can', 
                          color=col_can, **plot_kwargs)
        
        if plot_filtered and self.trk_filtered is not None:
            
            #gnss
            feat_filt = ["x", "y", "psi", "dpsi", "delta", "phi", "dphi", "v"]
            self.trk_filtered.plot(features=feat_filt, axes = axes_t[[0, 1, 2, 
                                                                      3, 4, 6, 
                                                                      7, 8, ]],
                          plot_over_timestamps=True, label='filtered', 
                          color=col_fil, **plot_kwargs)
            axes_xy = self.trk_filtered.plot_xy(ax=axes_xy, label='filtered', 
                                       color=col_fil, **plot_kwargs)
            
        axes_t[-2].legend()
        axes_t[0].set_title(self.name)
        axes_xy.legend()
        axes_xy.set_ylabel('y [m]')
        axes_xy.set_xlabel('x [m]')
        axes_xy.set_title(self.name)
        axes_xy.set_aspect('equal')
        
        return axes_t, axes_xy
    
class BalanceAssistLogDataManager(DataManager):
    """
    Manage the CAN bus log data from the Balance Assist Bikes.
    """

    NUM_SKIP_ROWS = 6  # Number of rows to skip at the bginning of a .txt file
    GPS_LEAP_SECONDS = 18  # GPS leap seconds for GPS time to UTC conversion
    G = 9.81  # gravity

    def __init__(self, 
                 path_can_log, 
                 path_gnss_report,
                 bike_geometry, 
                 dbc_file=None, 
                 steer_angle_bias = 19.5,
                 ins_filename_suffix = "-ins"):
        """
        Create a BalanceAssistLogDataManager.

        Parameters
        ----------
        path_can_log : str
            Path to the MF4 file.
        path_gnss_report : TYPE
            Path to the GNSS report directory.
        dbc_file : TYPE
            Path to the DBC file.
        bike_geometry : InstrumentedBikeGeometry
            InstrumentedBikeGeometry for coordinate transformations.
        steer_angle_bias : float, optional
            A constant bias of the steer angle measurement in deg. 
            The default is 19.5.

        Returns
        -------
        None.

        """
        
        super().__init__(path_can_log)
        if dbc_file is None:
            self.dbc_file = None
        else:
            self.dbc_file = verify_filepath_dbc(dbc_file)
        self.dir_gnss_report = path_gnss_report
        self.bike_geom = bike_geometry
        self.steer_angle_bias = steer_angle_bias
        self.ins_filename_suffix = ins_filename_suffix


    def _load_can_logs(self, can_files):
        """ Load a list of can_files into a single data frame. """
        if can_files[0].lower().endswith('.mf4') or can_files[0].lower().endswith('.txt'):
            if self.dbc_file is None:
                raise ValueError("Encoded CAN logs given but no .dbc CAN database definition supplied!")
            can_files = [os.path.join(self.dir, f) for f in can_files]
            data = process_can(can_files, self.dbc_file)
        elif can_files[0].lower().endswith('.parquet'):
            data = pd.concat([decode_parquet(f) for f in can_files])
        else:
            raise NotImplementedError(f"Loading CAN logs of filetype {can_files[0]} is not supported!")
        
        return data


    def load_track(self, can_files, filenames_bike_gnss):
        """
        Load a Track object holding the trajectories captured from the CAN bus.

        Parameters
        ----------
        can_files : str
            A list containing the can log files. If the list contains multiple files, the logs are stitched together.
            Can be .mf4 or decoded .parquet files. example: ['00001024//00000001.MF4']
        filenames_bike_gnss : list
            List of gnss solution filenames. 

        Returns
        -------
        trk : trajdatamanager.Track
            A track object holding the loaded trajectory data.
        """

        # extract CAN data
        df = self._load_can_logs(can_files)

        t_can = np.array((df.index - df.index[0]).total_seconds())

        a_can = (
            np.sqrt(
                df["accel_x"] ** 2 + df["accel_y"] ** 2 + df["accel_z"] ** 2
            )
            - self.G
        )
        a_can = np.array(a_can)
        t_a_can, a_can, mask = to_finite(t_can, test=a_can, return_mask=True)
        
        # roll and yaw and linear acceleration
        df = df.interpolate(method='time', limit=5)
        yaw, dyaw, roll, droll, accel = self.bike_geom.transform_imu2bike(
            df["yaw"],
            df["roll"],
            df["gyro_x"],
            df["gyro_z"],
            df["accel_x"],
            )

        # speed
        #   Derived from the rear wheelspeed assuming a circumfence of 221cm
        #   With rider (approx 70kg) and 3.0 bar tirepressure, the effective
        #   circumfence reduces to 219.25cm
        speed = np.array(df["ws_rear"]) * (219.25/221)
        
        # steer angle
        #   The steer assist biccle assumes a wrong sensor bias. This is 
        #   corrected here. Additionally direction definitions varies.
        #   Definition on the balance-assist bikes: right+, left-
        #   Definition in cyclistsocialforces: right-, left+
        steer = -(
            np.pi * (np.array(df["LWS_ANGLE"]) - self.steer_angle_bias) / 180
        )
        dsteer = -(np.pi * np.array(df["LWS_SPEED"]) / 180)

        # extract GNSS IMU data for time synchronisation of the CAN data
        t_gnss_global, a_gnss, path_timesync_source = self.load_gnss_imu(
            filenames_bike_gnss
        )
        t_gnss_global_begin = t_gnss_global[0]
        t_gnss_local = np.array(
            [(ti - t_gnss_global[0]).total_seconds() for ti in t_gnss_global]
        )

        #plot for validation
        #fig0, ax0 = plt.subplots(2,1, sharex=True)
        #ax0[0].plot(t_gnss_local, a_gnss)
        #ax0[1].plot(t_can, a_can)

        # identify time offset between local times of CAN and GNSS data
        t_ss = 0.001
        t_gnss_100 = np.arange(0, t_gnss_local[-1], t_ss)
        t_can_100 = np.arange(0, t_a_can[-1], t_ss)

        a_can_interp = np.interp(t_can_100, t_a_can, a_can)
        a_gnss_interp = np.interp(t_gnss_100, t_gnss_local, a_gnss)
        
        a_gnss_interp_med = np.median(a_gnss_interp)
        a_can_interp_med = np.median(a_can_interp)

        a_corr = correlate(a_can_interp-a_can_interp_med, a_gnss_interp-a_gnss_interp_med, mode="full")
        n_offset = (np.argmax(a_corr) - len(a_gnss_interp) + 1)
        t_offset = n_offset * t_ss
            #t_offset = -np.argmax(a_corr) * 0.005
        print(f"Time offset: {t_offset} s", end="")

        #plot for validation
        fig, ax = plt.subplots(1,1)
        if n_offset < 0: 
            ax.plot(a_gnss_interp[abs(n_offset):]-a_gnss_interp_med, label='gnss')
            ax.plot(a_can_interp-a_can_interp_med, label='can')
        else:
            ax.plot(a_gnss_interp-a_gnss_interp_med, label='gnss')
            ax.plot(a_can_interp[abs(n_offset):]-a_can_interp_med, label='can')
        ax.legend()
        
        #identify drift
        if n_offset < 0: 
            get_drift = find_drift(a_gnss_interp[abs(n_offset):], a_can_interp,
                                   t_ss, plot=True)
            drift = get_drift(t_can[:,np.newaxis])
        else:
            get_drift = find_drift(a_gnss_interp, a_can_interp[abs(n_offset):], 
                                   t_ss, plot=True)
            drift = get_drift(t_can[:,np.newaxis]-t_offset)
            
        # plot for validation
        fig2, ax2 = plt.subplots(1,1)
        ax2.plot(a_corr)

        # derive timstamps for CAN data from GNSS time, offset and drift
        t_can_global = t_gnss_global_begin + dt.timedelta(seconds=1) * \
            (t_can - (t_offset - drift)) 

        # plot for validation
        fig3, ax3 = plt.subplots(1, 1)
        ax3.plot(t_gnss_global, a_gnss-a_gnss_interp_med, label='GNSS')
        ax3.plot(t_can_global[mask], a_can-a_can_interp_med, label='can')
        ax3.set_title(("Time synchronization based on total linear"
                       "acceleration"))

        metadata = {
            "track_type": "BalanceAssistLogData",
            "source":can_files,
            "source_timesync": path_timesync_source,
            "dbc_file": self.dbc_file,
            "time_offset": t_offset,
        }

        # create a track with the CAN data
        trk = Track(
            'can',
            2,
            t_can_global,
            np.c_[steer, dsteer, roll, droll, yaw, dyaw, speed, accel],
            data_feature_keys=["delta", "ddelta", "phi", "dphi", "psi", 
                               "dpsi", "v", "a"],
            metadata=metadata,
        )

        return trk

    def load_gnss_imu(self, run_name):
        """
        Load IMU measurements from the RTK GNSS for time-synchronisation.

        Parameters
        ----------
        run_name : str
            Name of the GNSS file.

        Returns
        -------
        t_out : array-like
            Timestamps.
        a_out : array-like
            Total linear acceleration.
        path_timesync_source : str
            path of the file the data was taken from.

        """

        if not isinstance(run_name, list):
            run_name = [run_name]

        t_out = np.array([])
        a_out = np.array([])

        for rname in run_name:
            if rname[-4] == ".":
                rname = rname[:-4]

            
            path_timesync_source = os.path.join(
                self.dir_gnss_report, rname, rname + self.ins_filename_suffix + ".csv"
            )
            
            #try alternative folder structure.
            if not os.path.isfile(path_timesync_source):
                path_timesync_source_alt = os.path.join(
                    self.dir_gnss_report, rname + self.ins_filename_suffix + ".csv"
                )
                
                if not os.path.isfile(path_timesync_source_alt):
                    msg = (f"Can't find INS file in the given report directory!"
                           f"Directory: {self.dir_gnss_report}"
                           f"Looked for {path_timesync_source} and "
                           f"{path_timesync_source_alt}")
                    raise FileNotFoundError(msg)
                else:
                    path_timesync_source = path_timesync_source_alt 

            df = pd.read_table(
                path_timesync_source,
                sep=",",
            )

            gps_epoch = dt.datetime(1980, 1, 6, tzinfo=dt.UTC)
            gps_weeks = df["GPS Week"]
            gps_tow = df["GPS TOW [s]"]

            t = []
            valid_times = np.ones(len(gps_tow), dtype=bool)
            i = -1
            for si, wi in zip(gps_tow, gps_weeks):
                i += 1
                if (np.isnan(si) or np.isnan(wi)) or (si == 0 or wi == 0):
                    valid_times[i] = False
                    continue
                t.append(
                    gps_epoch
                    + dt.timedelta(
                        weeks=wi, seconds=si - self.GPS_LEAP_SECONDS
                    )
                )

            t = np.array(t)

            a = np.sqrt(
                df["Acc X [g]"][valid_times] ** 2
                + df["Acc Y [g]"][valid_times] ** 2
                + df["Acc Z [g]"][valid_times] ** 2
            )
            a = (a * self.G) - self.G
            a = np.array(a)

            valid_accels = np.logical_not(np.isnan(a))
            a = a[valid_accels]
            t = t[valid_accels]

            t_out = np.r_[t_out, t]
            a_out = np.r_[a_out, a]

        return t_out, a_out, path_timesync_source
    
class InstrumentedBikeGeometry:
    """ A class that describes the geometry of the instrumented bicycle and 
    provides functions to transform sensor measurements to bicycle coordiantes.
    
    This class was copied and modified from rcid.utils by Christoph M. Konrad.
    """
    
    def __init__(self, bike_params):
        """
        Create and InstrumentedBikeGeometry object. 
        
        bike_params : dict
            A dictionary containing the bicycle dimensions. Must contain the 
            dimensions: 
                h_gnss : height of the GNSS antenna above ground when the
                bicycle is upright.
                l_gnss : horizontal distance between GNSS antenna and the rear
                wheel contact patch. 
        """
        
        self.params = bike_params
        
        # Create symbols and reference frames
        E, B, C = sm.symbols('E B C ', cls=me.ReferenceFrame)
        psi, phi = me.dynamicsymbols(r'\psi \phi')
        
        h_gnss, l_gnss = sm.symbols('h_gnss, l_gnss')
        gyro_z, gyro_x = sm.symbols('gyro_z gyro_x')
        
        # Set rotations of reference frames
        B.orient_body_fixed(E, (psi, sm.pi + phi, 0), 'ZXY') # Bike frame 
        C.orient_axis(B, 0, B.x) # GNSS sensor frame equals bike frame
        
        # transformation from GNSS coordinates to bike coordinates.
        
        if not np.all([k in self.params.keys() for k in ['l_gnss', 'h_gnss']]):
            msg = ("The calibration must include measurements of the antenna"
                   "height over ground 'h_gnss' and horizontal distance"
                   "to the rear-wheel contanct point 'l_gnss'.")
            raise ValueError(msg)

        P = me.Point('P')
        P1 = P.locatenew('P_1', - h_gnss * B.z)
        Pgnss = P1.locatenew('P_{GNSS}', - l_gnss * B.x)
        

        param_vals = {l_gnss: bike_params['l_gnss'],
                      h_gnss: bike_params['h_gnss']}
        r_P_Pgnss = P.pos_from(Pgnss).express(E)
        r_P_Pgnss = r_P_Pgnss.subs(param_vals)
        
        self.eval_r_P_Pgnss = sm.lambdify((psi, phi), r_P_Pgnss.to_matrix(E))
        
        # transformation of IMU measurements to roll/yaw
        Cz_w_E_exprB = C.ang_vel_in(E).dot(B.z)
        Cz_w_E_exprE = sm.trigsimp(C.ang_vel_in(E).dot(E.z))
        Cz_w_E_exprE = sm.solve(Cz_w_E_exprB - gyro_z, Cz_w_E_exprE)[0]
        
        Cx_w_E_exprB = C.ang_vel_in(E).dot(B.x)
        Cx_w_E_exprB = sm.solve(Cx_w_E_exprB - gyro_x, Cx_w_E_exprB)[0]
        
        self.eval_C_w_E = sm.lambdify((phi, gyro_x, gyro_z), 
                                      sm.Matrix([[Cz_w_E_exprE],
                                                 [Cx_w_E_exprB]]))
        

    def transform_gnss2rwcp(self, p_x_gnss, p_y_gnss, psi, phi):
        """ Transform the coordinates into the location of the 
        rear-wheel contact patch.
        """
        
        translation = self.eval_r_P_Pgnss(psi, phi)
        
        p_x = p_x_gnss + translation[0,0,:]
        p_y = p_y_gnss + translation[1,0,:]
        
        return p_x, p_y
    
    
    def transform_imu2bike(self, psi_m, phi_m, gyro_x_m, gyro_y_m, a):
        """
        Transform data measured from the IMU on the rack to the bicycle 
        reference frame.
        """
        
        rotation = self.eval_C_w_E(phi_m, gyro_x_m, gyro_y_m)
        
        dpsi = rotation[0,0,:]
        dphi = rotation[1,0,:]
        
        psi = - psi_m
        
        return psi, dpsi, phi_m, dphi, a   
            
    def transform_N2E(self, data_dict):
        """
        Transform a data dictionary from the N to the E frame.
        """

        keys_to_mirror = ['y', 'psi', 'dpsi', 'delta', 'ddelta']
        
        if isinstance(data_dict, dict):
            keys = list(data_dict.keys())
        elif isinstance(data_dict, Track):
            keys = data_dict.data_feature_keys
        else:
            raise ValueError(f"data_dict must be 'dict' or 'Track'. Instead it was '{type(data_dict)}'.")
        
        for k in keys_to_mirror:
            pattern = r"^"+k+r"(?:_|$).*"
            found_k = False
            for kk in keys:
                if re.findall(pattern, kk):
                    data_dict[kk] = - data_dict[kk]
                    found_k = True
            if not found_k:
                raise KeyError(f"Couldn't find data feature corresponding to key '{k}' in data_dict with keys {keys}.")
        
        return data_dict
    
    def transform_E2N(self, data_dict):
        """
        Transform a data dictionary from the E to the N frame.
        """
        
        return self.transform_N2E(data_dict)
  

def to_continous_angle(angle, tol = 0.75):
    """
    Convert an array of angles limited within [-pi, pi] to a continous array
    across multiples of pi. 

    Parameters
    ----------
    angle : array
        Limited array.
    tol : float
        Tolerance to detect a jump from one side of the unit circle to the
        other. A jump is detected if |angle[i+1] - angle[i]| > 2pi * tol. 
        Increase if true dynamics are detected as direction jumps. Decrease if
        direction jumps are missed because of low sample rates. Optional.
        The default is 0.75.

    Returns
    -------
    angle_out : array
        Continous array.

    """
    
    finite = np.isfinite(angle)
    angle_fin = angle[finite]
    
    th = 2 * np.pi * tol
    
    dangle = np.abs(np.diff(angle_fin))#angle_fin[2:] - angle_fin[:-2]
    idx_jumps = np.argwhere(np.abs(dangle) > th).flatten()
    idx_jumps = np.r_[[0], idx_jumps+1, [len(angle_fin)]]
    
    sign = np.sign(angle_fin)
    
    k = 0
    for i in range(len(idx_jumps)-1):
        angle_fin[idx_jumps[i]:idx_jumps[i+1]] += k 
        
        k += sign[idx_jumps[i+1]-1] * 2 * np.pi
    
    angle_out = np.nan * np.ones_like(angle)
    angle_out[finite] = angle_fin
    
    return angle_out


def limit_angle(angle):
    """
    Limit a data series of angles in radians to [-pi, pi] 

    Parameters
    ----------
    angle : array-like
        The array of angles to be limited.

    Returns
    -------
    angle_out : array-like
        The array of limited angles.

    """
    
    finite = np.isfinite(angle)
    angle_fin = angle[finite]
    
    i = 0
    while np.any(np.abs(angle_fin) > np.pi):
        idx = min(np.argwhere(np.abs(angle_fin) > np.pi).flatten())
        
        if idx == 0:
            sign = -1 * np.sign(angle_fin[idx])
        else:
            if np.gradient(angle_fin)[idx] > 0:
                sign = -1
            else:
                sign = 1
        
        angle_fin[idx:] += sign * 2 * np.pi
        
        i+=1
        if i > 100000:
            raise RuntimeError('Endless loop!?')
    
    angle_out = np.nan * np.ones_like(angle)
    angle_out[finite] = angle_fin
    
    return angle_out


def estimate_yaw(x, y):
    """ Estimates the yaw angle from x-y trajectories in the world
    reference frame (z-axis into the air). 
    
    yaw = arctan(y(t+1) - y(t-1) / x(t+1) - x(t-1))
    
    Positive yaw corresponds to positive rotation about the z-axis. Zero yaw
    corresponds to an orientation in x-direction.
    
    Copied and modified from rcid.utils by Christoph Konrad. 
    
    Parameters
    ----------
    
    x : array-like,
        X trajectory
    y : array-like
        Y trajectory
        
    Returns
    -------
    
    psi : array-like
        Yaw trajectory
    """
    
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    
    if x.size != y.size:
        msg = (f"x and y must be the same size! Instead x was size {x.size} " 
               f"and y was size {y.size}!")
        raise ValueError(msg)

    finite = np.logical_and(np.isfinite(x), np.isfinite(y))

    x_fin = x[finite]
    y_fin = y[finite]

    t = np.arange(x.size)
    t_fin = t[finite][1:-1]
    
    with warnings.catch_warnings(action="ignore"): 
        #ignore div/0 warning that occurs when someone is stationary
        temp, psi = cart2polar(x_fin[2:] - x_fin[:-2], y_fin[2:] - y_fin[:-2])
    
    # stationary (i.e. dx = 0) causes nan values. Replace them with the prev.
    # orientation. 
    i_stationary = np.where(np.isnan(psi))[0]
    psi[i_stationary] = psi[i_stationary-1] 
    
    # interpolate to time base of x and y
    psi = np.interp(t, t_fin, psi)
    psi[np.logical_not(finite)]=np.nan
    
    return psi


def update_yaw(trk, keys=("x", "y", "psi")):
    """ Wrapper around estimate_track() to update the yaw for a Track object
    containing an x and a y trajectory.
    
    The track must already contain a yaw trajectory 
    (which will be overwritten.)
    
    Parameters
    ----------
    
    trk : trajdatamanger.datamanager.Track
        A track containing x, y and yaw features. 
    
    keys : list-like, optional
        A tuple of keys that retrive the x, y and psi features of the track. 
        The default is ("x", "y", "psi")
    
    Returns
    -------
    trk : trajdatamanger.datamanager.Track
        The input track with updated yaw trajectory. 
    """
    
    x = trk[keys[0]]
    y = trk[keys[1]]
        
    psi = estimate_yaw(x, y)
    
    trk[keys[2]] = psi
    
    return trk


def expand_timebase(data, time_data, time_target):
    """ 
    Given a data series, its timestamps and a series of timestamps with 
    higher temporal resolution, expand fill the data series with NaN values 
    at times where the original series doesn't have values.
    
    The current time stamps of must be a subset of the requested time stamps.

    Parameters
    ----------
    data : array
        Data series to be expanded.
    time_data : array
        Timestamps of the data series.
    time_target : array
        New desired time stamps.

    Returns
    -------
    data_new : array
        Data series expanded with NaN.

    """
    positions = np.searchsorted(time_target, time_data)
    data_new = np.full((len(time_target), data.shape[1]), np.nan)
    data_new[positions, :] = data

    return data_new 

def find_drift(x1, x2, t_s, plot=False):
    """
    Given two measurements x1 and x2 of the same signal that are (rougly) 
    aligned in time, identify the time drift of one signal with respect to 
    the other. 

    Parameters
    ----------
    x1 : array-like
        First signal.
    x2 : array-like
        Second signal.
    t_s : float
        Sampling time of both signals.
    plot : bool, optional
        Plot the drift function. 

    Returns
    -------
    drift : function
        Function calculating the time offset of signal 2 relative to signal 1 
        as a function of signal time: drift(t)

    """
    
    subsignal_length = 1
    signal_length = min(len(x1) * t_s, len(x2) * t_s)
    n = int(subsignal_length / t_s)
    
    drift_offsets = []
    drift_offset_times = []
    
    for i in range(int(signal_length / subsignal_length)):
        x1i = x1[i*n:(i+1)*n]
        x2i = x2[i*n:(i+1)*n]
        
        corr = correlate(x1i-np.median(x1i), x2i-np.median(x2i), mode="full")
        
        drift_offset_times.append(t_s*(i*n+((i+1)*n - i*n)/2))
        drift_offsets.append(np.argmax(corr) - len(x2i) + 1)
    
    #estimate drift function
    drift_offsets = np.array(drift_offsets) * t_s
    drift_offset_times = np.array(drift_offset_times)[:,np.newaxis]
    reg = RANSACRegressor(max_trials=1000).fit(drift_offset_times, drift_offsets)           
    drift_offsets_fitted = reg.predict(drift_offset_times)
    
    #plot for validation
    if plot:
        fig, ax = plt.subplots(layout='constrained')
        ax.plot(drift_offset_times, drift_offsets, label="drift samples")
        ax.plot(drift_offset_times, drift_offsets_fitted, 
                label="fitted drift function")
        ax.set_xlabel("signal time [s]")
        ax.set_ylabel("time offset [s]")
        ax.set_title((f"Time offset of signal 2 relative to signal 1 \n "
                      f"drift(t) = ({reg.estimator_.coef_[0]*1000:.4f}) ms/s "
                      f"* t + ({reg.estimator_.intercept_:.4f}) s"))
    
    return reg.predict


def cart2polar(x, y):
    """
    Transfrom cartesian coordinates into polar coordinates with the angle psi
    in the range [-pi, pi]
    
    Function copied and modified from cyclistsocialforce.utils by Christoph 
    Konrad (MIT License). 

    Parameters
    ----------
    x : array-like
    y : array-like

    Returns
    -------
    rho : array-like
    psi : array-like.

    """
    rho = np.sqrt(np.power(x, 2) + np.power(y, 2))

    psi = np.arccos(x / rho)
    if type(psi) is not np.ndarray:
        psi = np.array(psi)

    psi[y < 0] = -psi[y < 0]

    return rho, psi


def read_yaml(filepath):
    """
    Read a yaml config file and return config.
    
    Function copied from rcid.utils by Christoph Konrad.

    Parameters
    ----------
    filepath : str
        The path of the yaml config file.

    Returns
    -------
    config : dict
        (Nested) dictionary of the configration.

    """
    
    #check path
    if not os.path.isfile(filepath):
        raise ValueError(f"Not a file: {filepath}")
    if not filepath[-5:] == ".yaml":
        raise ValueError(f"Not a .yaml file: {filepath}")
        
    with open(filepath, "r") as f:
        config = yaml.safe_load(f)

    return config