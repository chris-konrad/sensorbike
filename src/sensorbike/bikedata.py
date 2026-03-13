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

from matplotlib.offsetbox import AnchoredText

from scipy.signal import correlate
from sklearn.linear_model import RANSACRegressor
from pathlib import Path

# own imports
from trajdatamanager.datamanager import Track, DataManager
from trajdatamanager.gnss import RTKLibGNSSManager, RTKLibGNSSTrack
from trajdatamanager.utils import to_finite

# local imports
from sensorbike.ukf import filter, get_default_filter_settings, make_Q, make_R, plot_filter_result_t, plot_filter_result_xy
from sensorbike.canbus import process_can, decode_parquet, verify_filepath_dbc, list_decoded_canlogs
from sensorbike.geometry import InstrumentedBikeGeometry

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
        geometry_params = None,
        filename_can = None,
        t_s = 0.01,
        subdir_bike_gnss_solution=None,
        subdir_bike_gnss_report=None,
        subdir_bike_can=None,
        rotation = 64.318889,
        reference_location = [51.999370, 4.370451, 43.72],
        filter_settings = None,
        gnss_data_settings = {},
        can_data_settings = {},
        desired_reference_frame = 'N', 
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
        geometry_params : dict, optional
            A dictionary describing the position of the GNSS antenna and bicycle IMU:
                h_gnss : distance [m] between GNSS antenna and the rear-wheel contact point in B.z direction (vertical). Default is -1.08 m
                l_gnss : distance [m] between GNSS antenna and the rear-wheel contact point in B.x direction (horizontal). Default is -0.16 m
                h_imu  : distance [m] between the onboard IMU and the rear-wheel contact point in B.z direction (vertical). Default is -0.82 m
                l_imu : distance [m] between the onboard IMU and the rear-wheel contact point in B.x direction (horizontal). Default is 0 m
            The default corresponds to the setup used for the interaction experiment (GNSS with wooden antenna post). 
            Use sensorbike.geometry.get_geometry_params() for different configurations that have been used before. 
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
            system given as (lat[deg], long[deg], height[m]). The height is interpreted as the
            ellipsoidal height of the WGS84 reference ellipsoid. Visit https://www.unavco.org/software/
            geodetic-utilities/geoid-height-calculator/geoid-height-calculator.html to look up the 
            height of a desired reference location on the map. The default is (51.999370, 4.370451, 43.72).
        filter_settings : dict, optional
            Dictionary of Unscented Kalman Filter settings. 
            See the output of sensorbike.ukf.get_default_filter_settings()
            for details on the expected format. If None, 
            the output of this function is used. The default is None.
        desired_reference_frame : str, optional
            The frame to represent filtered bicycle states in. Choose from 'E' and 'N'. The 
            N-frame is the frame with x and y directions fixed to the ground and z pointing
            into the ground (as common in bicycle dynamic research and used for the Carvallo-Whipple model). 
            The E-frame is the frame with x and y directions fixed to the ground 
            and z pointing upwards (as common in traffic engineering and used for 
            cyclistsocialforces). The default is 'N'. 
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
        self.is_filtered = False
        
        # reference frame
        if desired_reference_frame not in ['E', 'N']:
            raise ValueError(f"The reference frame must be 'E' or 'N', instead it was '{desired_reference_frame}'.")
        self.desired_reference_frame = desired_reference_frame

        # rotation and reference location
        self.rotation = rotation
        self.reference_location = reference_location
        
        # bicycle geometry
        self.bike_geom = InstrumentedBikeGeometry(geometry_params=geometry_params)

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
        self.data_raw = None
        self.states_filtered = None
        
        # plot params
        marker = '.'
        markersize = 0.5
        linewidth = 0.5
        self.lineplot_kwargs = {"marker": marker, "linewidth": linewidth, "markersize": markersize}

        self.colors = dict(
            gnss = '#EC6842', 
            imu = '#A50034',
            steer_encoder = '#00A6D6',
            wheelspeed = '#0076C2',
            ukf_filter='#FFB81C',
            rts_smoother='#009B77')
        
    
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
        
        def _limit_angle(trk):
            for k in trk.data_feature_keys:
                for kk in ['psi', 'phi', 'delta']:
                    pattern = rf"(?<!d){kk}"
                    if re.findall(pattern, k):
                        trk[k] = limit_angle(trk[k])
            return trk

        
        # combine into one track
        t,  t_span = self._find_time_frame(trk_gnss, trk_can) 

        #interpolate
        trk_can = _to_continous_angles(trk_can)
        trk_can.sample_at_times(np.r_[t, t[-1]+dt.timedelta(seconds=self.t_s)])
        trk_can = _limit_angle(trk_can)

        t = trk_can.t
        t_span = (t[0], t[-1])

        trk_can.crop_to_timespan(t_span[0], t_span[1])
        trk_gnss.crop_to_timespan(t_span[0], t_span[1])
        
        data_gnss = expand_timebase(trk_gnss.data, trk_gnss.t, t)
        data_can = trk_can.data
        
        n = min(data_gnss.shape[0], data_can.shape[0]) #fix occasional numerical error
        data = np.c_[data_gnss[:n], data_can[:n]]
        t = t[:n]

        feat = [k + "_gnss" for k in trk_gnss.data_feature_keys] + \
               [k + "_can" for k in trk_can.data_feature_keys]

        metadata = {"meta_gnss": trk_gnss.metadata, 
                    "meta_can": trk_can.metadata,
                    "reference_frame": "sensor"}   
        
        trk = Track(self.name, 0, t, data, 
                    data_feature_keys = feat, 
                    yaw_feature_index = trk_gnss.yaw_feature_index, 
                    metadata = metadata)
        
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
        with warnings.catch_warnings():     #ignore timezone cast warning.
            warnings.simplefilter("ignore")
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
            trk_gnss.plot_xy(color="gray")

        #trk_gnss.plot_uncertainties()

        return trk_gnss
    
    
    def load_raw(self, t_begin=None, t_end=None, verbose=True, plot=True):
        """
        Load raw sensor data from the instrumented bicycle. This loads GNSS
        measurements and CAN logs, aligns them into the same timeframe and
        stores them in a single track object. 

        The measurements are not transformed to bicycle states. Instead they 
        are left in their original sensor reference frame. The only transformation is
        applied to GNSS, which is transformed from Lat/Long/Height (LLH) to XY rotated
        by 'rotation' (specified in the constructor) relative to East/North around  Up. 

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
            print("Loading gnss data ... ", end="")
        trk_gnss = self._load_gnss(t_begin=t_begin, t_end=t_end)
        if verbose:
            print("done!")
            
        # run bike dynamics data
        if verbose:
            print("Loading can data ... ", end="")
        trk_can = self._load_can()
        if verbose:
            print("done!")
           
        # align time frames
        if verbose:
            print("Combine datasets ... ", end="")
        trk_raw = self._combine_datasets(trk_gnss, trk_can)
        if verbose:
            print("done!")
                       
        self.data_raw = trk_raw
        self.data_loaded = True

        #plotting
        if plot:
            if verbose:
                print("Plotting raw sensor data ... ", end="")
            fig, axes = self.plot_raw_data()

            self.figures = {"raw": fig} | trk_can.figures

            if verbose:
                print("done!")

        return self.data_raw

    def get_bicyclestates_raw(self, plot=False, verbose=True):
        """ Get bicycle states using the naive mapping from measurements to states.

        Fast and sufficient for coarse applications.
        
        TODO
        """
        raise NotImplementedError()
    
    
    def get_bicyclestates_filtered(self, plot=False, verbose=True):
        """ Apply an Unscented Kalman Filter to fuse GNSS, IMU, steer encoder and
        wheelspeed sensor into an optimal estimate of the Carvallo-Whipple bicycle
        states. 

        Returns a sensorbike.bikedata.BicycleStates(Track) object with the states
        [x, y, psi, v, phi, delta, psidot, phidot, deltadot]. 

        Parameters
        ----------
        verbose : bool, optional
            Verbose output. The default is True.
        plot : bool, optional
            Plot the filter results. The default is True

        Returns
        -------
        trk_filtered : sensorbike.bikedata.BicycleStates(Track)
            The filtered bicycle states.
        """
        
        if verbose:
            print(f"Getting UKF bicycle state estimates ... ", end="")
        
        def _parse_settings(sname):
            if sname in self.filter_settings.keys():
                return self.filter_settings[sname]
            else:
                msg = (f"Filter settings must provide '{sname}'!")
                raise KeyError(msg)
        
        R = make_R(_parse_settings('sensor_std'))
        Q = make_Q(_parse_settings('process_std'))
        
        int_method = _parse_settings('integration_method')
        bparams = _parse_settings('bicycle_parameter_dict')
        
        features_track_gnss = ["x_gnss", "y_gnss", "vx_gnss", "vy_gnss"]
        idx_gnss = [self.data_raw.data_feature_keys.index(k) for k in features_track_gnss]
        measurements_gnss = self.data_raw.data[:,idx_gnss]

        uncertainties_track_gnss = ["varx_gnss", "vary_gnss", "varvx_gnss", "varvy_gnss", "covxy_gnss", "covvxvy_gnss"] 
        idx_gnss_uncert = [self.data_raw.data_feature_keys.index(k) for k in uncertainties_track_gnss]
        uncertainties_gnss = self.data_raw.data[:,idx_gnss_uncert]

        features_track_can = ["delta_can", "ddelta_can", "gyrox_can", "gyroz_can", "ay_can", "az_can", "vrws_can"]
        idx_can = [self.data_raw.data_feature_keys.index(k) for k in features_track_can]
        measurements_can = self.data_raw.data[:,idx_can]

        measurements = np.c_[measurements_gnss, measurements_can]

        results = filter(measurements, uncertainties_gnss, 
                         R, Q, 
                         integration_method=int_method,
                         bicycle_parameter_dict=bparams,
                         plot=False,
                         plot_meas_error=plot,
                         bicycle_geometry=self.bike_geom,
                         estimate_gyro_biases=False)
        states_smoothed = results[2]
        
        if plot: 
            figxy, axxy = plot_filter_result_xy(measurements, *results[:4])

            idx_orient = [self.data_raw.data_feature_keys.index(k) for k in ['psi_can', 'phi_can']]
            figt, axest, figauxt, axesauxt = plot_filter_result_t(measurements, *results[:4], measurements_imuorient=self.data_raw.data[:,idx_orient])

            figs = {'states_xy': (figxy, axxy), 
                    'states_t': (figt, axest), 
                    'states_aux_t': (figauxt, axesauxt), 
                    'measurements': tuple(results[4:6])}

        # analyze steer-angle bias
        convergence_period = int(round(self.filter_settings['convergence_period'] / self.t_s))
        steer_bias = np.mean(states_smoothed[convergence_period:,-3])
        steer_bias_std = np.std(states_smoothed[convergence_period:,-3])
        steer_bias_converged = steer_bias_std < np.deg2rad(0.5)
        if steer_bias_converged:
            convergence_msg = " (converged)"
        else:
            convergence_msg = ""
        if verbose:
            print(f"\n    steer angle bias: {np.rad2deg(steer_bias):.2f}+/-{np.rad2deg(steer_bias_std):.2f} deg"+convergence_msg, end="")


        # pack into track object
        metadata = self.data_raw.metadata
        metadata['steer_angle_bias'] = {'mean_rad': float(steer_bias), 'std_rad': float(steer_bias_std), 'converged': bool(steer_bias_converged)}


        convergence_period = int(round(self.filter_settings['convergence_period'] / self.t_s))
        self.states_filtered = BicycleStates(f"Filtered states {self.name}", self.data_raw.t[convergence_period:],
                                             states_smoothed[convergence_period:,:9], reference_frame='N', metadata=metadata)
        self.states_filtered = self.states_filtered.transform_reference(self.desired_reference_frame)
        self.states_filtered.figures = figs
        
        if verbose:
            print("done!")
    
        return self.states_filtered
        

    def plot_raw_data(self):
        """
        Plot the raw data per sensor over time.
        """
        
        plot_kwargs = self.lineplot_kwargs
        
        fig_t, axes_t = plt.subplots(15,1, sharex=True, layout='constrained')
        for ax in axes_t:
            ax.grid()
            
        #gnss
        axes_t[0].set_title('RTK GNSS (SwiftNav Piki Multi)')
        feat_gnss = ['x_gnss', 'y_gnss', 'psi_gnss', 'v_gnss', 'vx_gnss', 'vy_gnss'] 
        self.data_raw.plot(features=feat_gnss, axes = axes_t[0:6],
                        plot_over_timestamps=True, 
                        color=self.colors['gnss'], **plot_kwargs)
        
        #imu
        axes_t[6].set_title('Onboard IMU (BNO086)')
        feat_imu = ['psi_can', 'gyroz_can', 'phi_can', 'gyrox_can', 'ay_can', 'az_can']
        self.data_raw.plot(features=feat_imu, axes = axes_t[6:12],
                plot_over_timestamps=True, 
                color=self.colors['imu'], **plot_kwargs)
        
        #steer encoder
        axes_t[12].set_title('Steer Encoder')
        feat_enc = ['delta_can', 'ddelta_can']
        self.data_raw.plot(features=feat_enc, axes = axes_t[12:14],
                plot_over_timestamps=True, 
                color=self.colors['steer_encoder'], **plot_kwargs)

        #wheelspeed sensor
        axes_t[14].set_title('Wheelspeed sensor (rear)')
        feat_wsp = 'vrws_can'
        self.data_raw.plot(features=feat_wsp, axes = axes_t[14],
                        plot_over_timestamps=True, 
                        color=self.colors['wheelspeed'], **plot_kwargs)
            
        axes_t[0].set_title(f"Raw Sensor Data: {self.name}")
        fig_t.set_size_inches(10.5, 9)
        
        return fig_t, axes_t
    

    def plot_filtered_states(self):
        """Plot the filtered states (xy-plot and t-plot)
        """
        if not hasattr(self, 'states_filtered'):
            raise RuntimeError("No filtered bicycle states available. Call 'get_bicyclestates_filtered()' before plotting!")
        
        axxy = self.states_filtered.plot_xy(color=self.colors['rts_smoother'])
        figxy = axxy.get_figure()

        axest = self.states_filtered.plot(color=self.colors['rts_smoother'], plot_over_timestamps=True)
        figt = axest.get_figure()

        return figxy, axxy, figt, axest


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
                 steer_angle_bias = 0,
                 circumfence_rear_wheel = 221,
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
            The steer encoder typically shows a constant bias. If this is 
            known, it can be specified here in degrees. The default is 0 deg.
        circumfence_rear_wheel : float, optional
            The circumfence of the rear wheel in cm. The default is 
            221 cm. This can be adjusted to account for rider weight and
            tire pressure, for example, for a 70 kg rider and 3 bar, the
            effective circumfence reduces to 219.25 cm. 
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
        self.circumfence_rear_wheel = circumfence_rear_wheel
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
        
        # roll and yaw and linear acceleration from the IMU
        df['yaw'] = to_continous_angle(df['yaw'])
        df = df.interpolate(method='time', limit=5)
        yaw = limit_angle(df["yaw"])
        roll = df["roll"]
        gyro_x = df["gyro_x"]
        gyro_z = df["gyro_z"]
        ay = df["accel_y"]
        az = df["accel_z"]

        # speed
        #   Derived from the rear wheelspeed assuming a nominal circumfence of 221cm
        speed = np.array(df["ws_rear"]) * (self.circumfence_rear_wheel/221)
        
        # steer angle
        #   The steer assist biccle assumes a wrong sensor bias. This is 
        #   corrected here. 
        #   Definition on the balance-assist bikes: right+, left-
        steer = np.deg2rad(df["LWS_ANGLE"].to_numpy() - self.steer_angle_bias)
        dsteer = np.deg2rad(df["LWS_SPEED"].to_numpy() - self.steer_angle_bias)

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
        print(f"\n    Time offset: {t_offset:.4f} s ")

        #plot for validation
        #fig, ax = plt.subplots(1,1)
        #if n_offset < 0: 
        #    ax.plot(a_gnss_interp[abs(n_offset):]-a_gnss_interp_med, label='gnss')
        #    ax.plot(a_can_interp-a_can_interp_med, label='can')
        #else:
        #    ax.plot(a_gnss_interp-a_gnss_interp_med, label='gnss')
        #    ax.plot(a_can_interp[abs(n_offset):]-a_can_interp_med, label='can')
        #ax.legend()
        
        #identify drift
        if n_offset < 0: 
            get_drift, drift_rate, figdrift, _ = find_drift(a_gnss_interp[abs(n_offset):], a_can_interp,
                                   t_ss, plot=True)
            drift = get_drift(t_can[:,np.newaxis])
        else:
            get_drift, drift_rate, figdrift, _ = find_drift(a_gnss_interp, a_can_interp[abs(n_offset):], 
                                   t_ss, plot=True)
            drift = get_drift(t_can[:,np.newaxis]-t_offset)
        print(f"    Time drift: {drift_rate:.4f} ms/s ", end="")
            
        # plot for validation
        #fig2, ax2 = plt.subplots(1,1)
        #ax2.plot(a_corr)

        # derive timstamps for CAN data from GNSS time, offset and drift
        t_can_global = t_gnss_global_begin + dt.timedelta(seconds=1) * \
            (t_can - (t_offset - drift)) 

        # plot for validation
        fig3, ax3 = plt.subplots(1, 1)
        ax3.plot(t_gnss_global, a_gnss-a_gnss_interp_med, label='gnss')
        ax3.plot(t_can_global[mask], a_can-a_can_interp_med, label='can')
        ax3.set_xlabel("time t")
        ax3.set_ylabel("total linear acceleration |a| [m/s^2]")
        ax3.legend()
        ax3.set_title(("Time synchronization based on total linear acceleration"))

        metadata = {
            "track_type": "BalanceAssistLogData",
            "source": can_files,
            "source_timesync": path_timesync_source,
            "dbc_file": self.dbc_file,
            "time_offset": float(t_offset),
            "drift_ms-per-s": float(drift_rate),
        }

        # create a track with the CAN data
        trk = Track(
            'can',
            2,
            t_can_global,
            np.c_[steer, dsteer, roll, gyro_x, yaw, gyro_z, speed, ay, az],
            data_feature_keys=["delta", "ddelta", "phi", "gyrox", "psi", 
                               "gyroz", "vrws", "ay", "az"],
            metadata=metadata,
        )
        trk.figures = {"timesync": fig3, "drift": figdrift}

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
  

class BicycleStates(Track):
    """Store and process the state trajectories of the Carvallo-Whipple bicycle model."""

    UNITS = ['m', 'm', 'rad', 'm/s', 'rad', 'rad', 'rad/s', 'rad/s', 'rad/s']
    FEATURES = ['x', 'y', 'psi', 'v', 'phi', 'delta', 'psidot', 'phidot', 'deltadot']
    REF_FRAMES = ['E', 'N']

    def __init__(self, track_id, t, data, reference_frame, data_feature_keys=None, metadata=None):
        """Create a BicycleStates Track object.
        
        Parameters
        ----------
        track_id : str
            The ID of this state track.
        t : array
            The timestamps of the N samples in the trajectory. Must be shaped (N,). Must be an array of Python datetime objects.
        data : array
            The data array shaped (N, 9) with the nine states [x, y, psi, v, phi, delta, psidot, phidot, deltadot]. Distances in m, 
            speed in m/s, angles in rad and rates in rad/s. A subset of the nine states may be supplied. In this case, the states in
            the second dimension must correspond to the lables of data_feature_keys.
        reference_frame : str
            The reference frame of the given data array. Data may be given in the 'N' frame, typically used for bicycle dynamics
            (N.x and N.y forming the ground plane, N.z pointing downwards), or the 'E' frame, common in traffic simulation (E.x and E.y 
            forming the ground plane, E.z pointing downwards). N is rotated by pi around E.x with respect to E. Additionally, 
            Steer angles and rates are mirrored: E_delta(dot) = - N_delta(dot).
        data_feature_keys : list
            List of states if data contains only a subset of the bicycle states. Must be a subset of [x, y, psi, v, phi, delta, psidot, phidot, deltadot]. 
            Order must correspond to the order of columns in data. May be used if all 9 states are available but in different order. 
        """

        class_id = 2
        yaw_feature_index = 2

        if data_feature_keys is not None:

            if not np.all([f in self.FEATURES for f in data_feature_keys]):
                raise ValueError((f"A BicycleStates Track object can only hold the features {self.FEATURES}. At least one of {data_feature_keys} is not allowed."))
            if len(data_feature_keys) != data.shape[1]:
                raise ValueError((f"Must supply the same number of data_feature_keys as columns in the data array."))
        
            data_full = np.full((data.shape[0], len(self.FEATURES)), np.nan)
            for i, f in enumerate(data_feature_keys):
                data_full[:, self.FEATURES.index(f)] = data[:,i]
            data = data_full

        if not data.shape[1] == len(self.FEATURES):
            raise ValueError((f"A BicycleStates Track object must have {len(self.FEATURES)} features/states. ",
                              f"The data array must be shaped (N, {len(self.FEATURES)}) or a list of the subset of states in data must be supplied in data_feature_keys. Data shape was {data.shape}. "
                              f"Please provide {self.FEATURES} or data_feature_keys!"))
        
        super().__init__(track_id, class_id, t, data, metadata, yaw_feature_index=yaw_feature_index, data_feature_keys=self.FEATURES)

        if reference_frame not in self.REF_FRAMES:
            raise ValueError(f"'frame' must be one of {self.FRAMES}. Instead it was '{frame}'.")
        self.reference_frame=reference_frame


    def transform_reference(self, frame):
        """Transform the bicycle states to reference a different reference frame (in place).

        Available frames are the 'N' frame, typically used for bicycle dynamics
        (N.x and N.y forming the ground plane, N.z pointing downwards), or the 'E' frame, 
        common in traffic simulation (E.x and E.y forming the ground plane, E.z pointing downwards). 
        N is rotated by pi around E.x with respect to E. Additionally, steer angles and 
        rates are mirrored: E_delta(dot) = - N_delta(dot). In the N-frame the positive steer axis point 
        downwards along the steer column. 

        Parameters
        ----------
        frame : str
            The target frame of the transformation. If the target frame equals the current frame, 
            nothing will happen. 

        Return
        ------
        BicycleStates
            The transformed object.
        """

        if frame not in self.REF_FRAMES:
            raise ValueError(f"'frame' must be one of {self.FRAMES}. Instead it was '{frame}'.")

        if frame != self.reference_frame:
            states_flip = ['psi', 'psidot', 'delta', 'deltadot']
            idx_flip = [self.data_feature_keys.index(s) for s in states_flip]

            self.data[:,idx_flip] *= -1

            self.reference_frame = frame

        return self
    
    
    def plot(self, axes=None, features=None, plot_over_timestamps=False, t_plot=None, **plot_kwargs):
        """Plot the bicycle states with units over time.

        Parameters
        ----------
        axes : Axes or list of Axes, optional
            Axes to be plotted in. Must be the same number of axes as features 
            selected for plotting. If None, creates a new figure.
        features : str or list, optional
            List of feature names to plot. Must be the same number as the given axes. If
            None, all features are plotted.
        plot_over_timestamps : bool, optional
            If true, the data is plotted over timestamps. If false, the data
            is plotted over sample number. The default is False.
        t : array-like, optional
            Force plotting over a given array of t values. t must be the length of 
            the sample number. Overwrites plot_over_timestamps. Default is None.
        plot_kwargs : dict
            Keyword arguments passed to matplotlib.pyplot.plot(_,_,**plot_kwargs).

        Returns
        -------
        axes : list of axes
            The axes that the plot was created in.
        """

        if features is None:
            features = self.data_feature_keys
        units = [self.UNITS[self.FEATURES.index(f)] for f in features]
        
        super().plot(axes=axes, features=features, plot_over_timestamps=plot_over_timestamps,
                     t_plot=t_plot, **plot_kwargs)
        
        for i, ax in enumerate(axes):
            ax.set_ylabel(f"{features[i]} [{units[i]}]")

        frametext = AnchoredText(rf'Reference Frame: {{{self.reference_frame}}}', 'lower right')
        axes[0].add_artist(frametext)       

        return axes
    

    def plot_xy(self, ax=None, **kwargs):

        ax = super().plot_xy(ax, **kwargs)

        frametext = AnchoredText(rf'Reference Frame: {{{self.reference_frame}}}', 'lower right')
        ax.add_artist(frametext)
        return 


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
    if isinstance(angle, (pd.DataFrame, pd.Series)):
        angle = angle.to_numpy()
    
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
    x = np.cos(angle)
    y = np.sin(angle)
    angle_out = np.atan2(y, x)
    
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
        
    drift = reg.estimator_.coef_[0]*1000
    
    if plot:
        return reg.predict, drift, fig, ax
    else:
        return reg.predict, drift, None, None


def cart2polar(x, y):
    """
    Transform cartesian coordinates into polar coordinates with the angle psi
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