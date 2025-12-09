# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 17:14:11 2025

ukf
---

Apply an Unscented Kalman Filter to data gathered from an instrumented bicycle. 

This module was copied and modifed from rcid.ukf by Christoph Konrad.

@author: Christoph M. Konrad
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sm
import warnings

from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

from bicycleparameters.parameter_dicts import meijaard2007_browser_jason
from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
from bicycleparameters.models import Meijaard2007Model


def warn_wrong_parameters():
    warnings.warn((f"Using bicycleparameters default 'meijaard2007_browser_jason' parameters. These"
                f" are not the parameters of the Balance Assist Bikes. Replace 'bicycle_parameter_dict'" 
                f" in the filter settings with the parameters found in https://github.com/moorepants/BicycleParameters/blob/master/data/riders/Jason/Parameters/JasonBalanceassistv1Benchmark.txt"
                f" to use the Balance Assist Bicycle Parameters."))


def get_statespace_matrices(bp_model, v):
    """Get the statespace matrices of the Whipple-Carvallo bicycle after
    Meijaard et. al. (2007).
    
    xdot = A*x + B*u
    
    with 
    
    x = [phi, delta, phidot, deltadot, psi]^T
    u = [0, Tdelta]
    
    Function copied and modified from cyclistsocialforces.dynamics.Whipple
    CarvalloDynamics by Christoph Konrad (MIT License).
    
    Parameters
    ----------
    bp_model : Meijaard2007Model
        A Meijaard2007Model object from the bicycleparameters toolbox.
    v : float
        The speed of the bicycle
    
    Literature
    ----------
    
    Meijaard, J. p, Papadopoulos, J. M., Ruina, A., & Schwab, A. l. (2007). 
        Linearized dynamics equations for the balance and steer of a bicycle: 
        A benchmark and review. Proceedings of the Royal Society A: 
        Mathematical, Physical and Engineering Sciences, 463(2084), 1955–1982. 
        https://doi.org/10.1098/rspa.2007.1857

    """
    
    # get geometry parameters
    w = bp_model.parameter_set.parameters["w"]
    c = bp_model.parameter_set.parameters["c"]
    coslam = np.cos(bp_model.parameter_set.parameters["lam"])

    # pre-calc yaw state matrix coefficients
    A41_over_v = coslam / w
    A43 = coslam * c / w
    
    Awc, Bwc = bp_model.form_state_space_matrices(v=v)

    # add yaw dynamics
    A = np.zeros((5, 5))
    A[:4, :4] = Awc
    A[4, 1] = A41_over_v * v
    A[4, 3] = A43

    B = np.zeros((5, 2))
    B[:4, :] = Bwc

    # output
    C = np.zeros((1, A.shape[1]))
    C[0, 4] = 1
    D = np.zeros((C.shape[0], B.shape[1]))

    return A, B, C, D


def make_Rscale_from_sensorchar(sensor_characteristics):
    """
    Creates the measurement noise scale array from a nested dict containting 
    the standard errors of each measurement sorted by sensor.
    
    The dict must be structured:
        GNSS:
            x: sigma_x
            y: sigma_y
        IMU: 
            roll: sigma_roll
            yaw: sigma_yaw
            gyro: sigma_gyro
            accel: sigma_accel
        speedometer:
            v: sigma_v
        steerencoder:
            rate: sigma_rate
            angle: sigma_angle
            
    Designed for easy parsing of yaml config files. 
    
    Parameters
    ----------
    measurement_noise_dict : dict
        Nested dictionary of measurement noise standard errors.

    Returns
    -------
    measurement_noise_std : array
        Array of measurement noise standard errors.
    """
    
    #sensor_characteristics
    meas = sensor_characteristics
    
    # X and Y variance
    var_x = meas['GNSS']['x']**2
    var_y = meas['GNSS']['y']**2
    
    # PSI variance
    # estimated mean change of y an x between timesteps
    dy = 0
    dx = 11 / 3.6 * 0.02
    
    # error propagation through arctan
    psi, x, y = sm.symbols('psi x y')
    psi = sm.atan2(x,y)

    dpsidx = sm.lambdify((x,y), psi.diff(x).simplify())
    dpsidy = sm.lambdify((x,y), psi.diff(y).simplify())
    
    var_psi = 2 * dpsidx(dx, dy)**2 * var_x + 2 * dpsidy(dx, dy)**2 * var_y
    
    # OTHER
    var_phi = meas['IMU']['roll']**2
    #var_psi = meas['IMU']['yaw']**2
    var_delta = meas['steerencoder']['angle']**2
    var_dpsi = meas['IMU']['gyro']**2
    var_dphi = meas['IMU']['gyro']**2
    var_ddelta = meas['steerencoder']['rate']**2
    var_v = meas['speedometer']['v']**2
    var_dv = meas['IMU']['accel']**2
    
    measurement_noise_std = np.array([var_x, var_y, var_psi, 
                                        var_v, var_phi, var_delta, 
                                        var_dpsi, var_dphi, var_ddelta, 
                                        var_dv])  
    
    return measurement_noise_std
    
def make_Qscale_from_dict(process_noise_dict):
    """
    Creates the process noise scale array from a dict containting the 
    standard errors of each state.
    
    Must contain x, y, psi, v, dv, phi, dphi, psi, dpsi, delta, and ddelta.
    
    Parameters
    ----------
    process_noise_dict : dict
        Dictionary of process noise standard errors.

    Returns
    -------
    process_noise_std : array
        Array of process noise standard errors.
    """
    
    #process noise level
    prcs = process_noise_dict
    
    process_noise_std = np.array([prcs['x'], prcs['y'], prcs['psi'], 
                                    prcs['v'], prcs['phi'], prcs['delta'],
                                    prcs['dpsi'], prcs['dphi'], 
                                    prcs['ddelta'], prcs['dv']])
    
    return process_noise_std


def get_default_filter_settings():
    """
    Create the default filter settings based on sensor noise estimates.
    
    IMU standard errors are based on datasheet values from 
    https://docs.sparkfun.com/SparkFun_VR_IMU_Breakout_BNO086_QWIIC/
    assets/component_documentation/BNO080_085-Datasheet_v1.16.pdf
    
    GNSS, speedometer and steerencoder values are guesses. 

    Returns
    -------
    None.

    """
    
    filter_settings = {"integration_method": "midpoint",
                       "bicycle_parameter_dict": meijaard2007_browser_jason}
    warn_wrong_parameters()
    
    sensor_char = {"GNSS": {"x": 0.1, "y": 0.1},
                   "IMU": {"roll": 0.061,
                           "yaw": 0.061,
                           "gyro": 0.054,
                           "accel": 0.30},
                   "speedometer": {"v": 0.05},
                   "steerencoder": {"rate": 0.05, "angle": 0.03}}         
    filter_settings['measurement_noise_std'] = \
        make_Rscale_from_sensorchar(sensor_char)  

     
    process_noise_dict = {"x": 0.001, "y": 0.001, "psi": 0.001,
                           "dpsi": 0.01, "phi": 0.00001 ,
                           "dphi": 0.00005, "delta": 0.01, 
                           "ddelta": 0.05, "v": 0.001, "dv": 0.05}
    
    filter_settings['process_noise_std'] = \
        make_Qscale_from_dict(process_noise_dict)   

    return filter_settings


def parse_filter_settings(filter_settings_yaml_dict, 
                          bicycle_parameter_dict=None):
    """
    Parse a yaml dict of the filter settings. Return the dict format required
    by the filter functions. 
    
    The yaml must be structured:
        integration_method: "midpoint" or "backward euler"
        measurement_noise_std:
            GNSS:
                x: sigma_x
                y: sigma_y
            IMU: 
                roll: sigma_roll
                yaw: sigma_yaw
                gyro: sigma_gyro
                accel: sigma_accel
            speedometer:
                v: sigma_v
            steerencoder:
                rate: sigma_rate
                angle: sigma_angle
        process_noise_std:
            x: sigma_x
            y:sigma_y
            psi: sigma_psi
            dpsi: sigma_dpsi 
            phi: sigma_phi
            dphi: sigma_dphi
            delta: sigma_delta 
            ddelta: sigma_ddelta
            v: sigma_v
            dv: sigma_dv 

    Parameters
    ----------
    filter_settings_yaml_dict : dict
        See above.
    bicycle_parameter_dict : dict, optional
        Dictionary of physical bicycle parameters following the conventions
        of the bicycleparameters toolbox. If None, the 
        parameters of the balanceassit bicycle will be used.
        The default is None.

    Returns
    -------
    filter_settings : dict.
    """
    
    
    if bicycle_parameter_dict is None:
        bicycle_parameter_dict = meijaard2007_browser_jason
        warn_wrong_parameters()
    
    filter_settings = {"integration_method": 
                           filter_settings_yaml_dict["integration_method"],
                       "bicycle_parameter_dict": bicycle_parameter_dict}
        
    filter_settings['measurement_noise_std'] = \
        make_Rscale_from_sensorchar(
            filter_settings_yaml_dict["measurement_noise_std"])
    
    filter_settings['process_noise_std'] = \
        make_Qscale_from_dict(
            filter_settings_yaml_dict['process_noise_std'])
        
    return filter_settings
    

def move_dynamic(x, t_s, bp_model, integration_method='euler'):
    """
    Predict the next step of the dynamic whipple-carvallo bicycle model
    assuming zero steer and roll torque. 
    
    The state vector is [p_x, p_y, psi, v, phi, delta, dpsi, dphi, ddelta, dv].
    
    Uses euler integration of the state-space formulation.

    Parameters
    ----------
    x : array-like
        Current state vector.
    t_s : float
        Time step.
    bp_model : Meijaard2007Model
        A Meijaard2007Model object from the bicycleparameters toolbox    
    integration_method: str, optional
        The integration method for the integration of the system dynamics in 
        the prediction step. Can be 'backward euler' or 'midpoint'. The default
        is 'backward euler'.

    Returns
    -------
    x_pred : array-like
        Predicted state vector.

    """
    #extract for readibility
    p_x = x[0]
    p_y = x[1]
    psi = x[2]
    v = x[3]
    phi = x[4]
    delta = x[5]
    dpsi = x[6]
    dphi = x[7]
    ddelta = x[8]
    dv = x[9]
    
    #bicycle lateral dynamics
    x_lat = np.array([phi, delta, dphi, ddelta, psi])
    
    if integration_method == 'euler':
        
        #constant acceleration
        v_pred = dv * t_s + v
        dv_pred = dv

        #bicycle lateral dynamics
        A, B, C, D = get_statespace_matrices(bp_model, v)
        x_lat_pred = x_lat + t_s * A @ x_lat
        A_pred, B, C, D = get_statespace_matrices(bp_model, v_pred)
        dpsi_pred = (A_pred @ x_lat_pred)[-1]
        
        # forward dynamics
        p_x_pred =  v * np.cos(psi) * t_s + p_x
        p_y_pred =  v * np.sin(psi) * t_s + p_y
        
        #pack
        x_pred = [p_x_pred, p_y_pred, x_lat_pred[4], v_pred, 
                  x_lat_pred[0], x_lat_pred[1], dpsi_pred,
                  x_lat_pred[2], x_lat_pred[3], dv_pred]
        
    elif integration_method == 'midpoint':
        
        #constant acceleration
        v_pred = dv * t_s + v
        v_pred_h2 = dv * t_s / 2 + v
        dv_pred = dv
        
        #bicycle lateral dynamics
        A_h2, B, C, D = get_statespace_matrices(bp_model, v_pred_h2)
        I = np.eye(x_lat.size)        
        x_lat_pred = np.linalg.inv(I - t_s/2 * A_h2) \
            @ (I + t_s/2 * A_h2) @ x_lat
        psi_pred = x_lat_pred[-1]
        
        A_h, B, C, D = get_statespace_matrices(bp_model, v_pred)
        dpsi_pred = (A_h @ x_lat_pred)[-1]

        # forward dynamics
        p_x_pred = ((v_pred + v) / 2) * np.cos((psi_pred + psi) / 2) * t_s \
            + p_x
        p_y_pred = ((v_pred + v) / 2) * np.sin((psi_pred + psi) / 2) * t_s \
            + p_y 
        
        x_pred = [p_x_pred, p_y_pred, x_lat_pred[4], v_pred, 
                  x_lat_pred[0], x_lat_pred[1], dpsi_pred,
                  x_lat_pred[2], x_lat_pred[3], dv_pred]
        
    else:
        raise ValueError("Unknown integration method!")
    
    return x_pred

def move_kinematic(x, t_s, dv=None, dpsi = None):
    """
    Kinematic constant acceleration and yaw rate model.
    
    The state vector is [p_x, p_y, psi, dpsi, v].

    Parameters
    ----------
    x : array-like
        Current state vector.
    t_s : float
        Time step.
    dv : float (optional)
        Acceleration input
    dpsi : float (optional)
        Yaw rate input
        
    Returns
    -------
    x_pred : array-like
        Predicted state vector.
    """
    
    #extract for readibility
    p_x = x[0]
    p_y = x[1]
    psi = x[2]
    v = x[4]
    
    if dv is None:
        dv = x[5]
    if dpsi is None:
        dpsi = x[3]
    
    #predict
    p_x_pred = v * np.cos(psi) * t_s + p_x
    p_y_pred = v * np.sin(psi) * t_s + p_y
             
    psi_pred = dpsi * t_s + psi
    dpsi_pred = dpsi 

    v_pred = dv * t_s + v
    
    dv_pred = dv 
    
    #pack
    x_pred = [p_x_pred, p_y_pred, psi_pred, dpsi_pred, v_pred, dv_pred]
    x_pred = [float(x_i) for x_i in x_pred]
    
    return x_pred

def measure_dynamic(x, gnss_available):
    """
    Extract the measured states of the dynamic whipple-carvallo model.These are
    - p_x: Measured by GNSS
    - p_y: Measured by GNSS
    - psi: Derived from GNSS x and y
    - dpsi: Yaw rate from IMU
    - v: speed from IMU
    - dv: acceleration from IMU
    - delta: steer angle from the steer encoder
    - ddelta: steer rate from the steer encoder
    - phi: roll angle from the IMU
    - dphi: roll rate from the IMU
    
    The state vector is [p_x, p_y, psi, v, phi, delta, dpsi, dphi, ddelta, dv].

    Parameters
    ----------
    x : array-like
        State vector.
    gnss_available : bool 
        Flag indicating if GNSS is available at the current time step. 

    """
    
    C = np.eye(10)
    if not gnss_available:
        C[:3,:3] = np.zeros((3,3))
    return (C @ x).flatten()

def measure_kinematic(x, gnss_available):
    """
    Extract the measured states. These are
    - p_x: Measured by GNSS
    - p_y: Measured by GNSS
    - psi: Derived from GNSS p_x and p_y
    - dpsi: Yaw rate from IMU
    - v: speed from IMU
    - dv: acceleration from IMU

    Parameters
    ----------
    x : array-like
        State vector [p_x, p_y, psi, dpsi, v].
    gnss_available : bool 
        Flag indicating if GNSS is available at the current time step. 

    """
    C = np.eye(6)
    if not gnss_available:
        C[:3,:3] = np.zeros(3)
    return (C @ x).flatten()


def filter_dynamic(measurements, R, Q, t_s=0.01, smooth = True, plot = True, 
                   integration_method = "backward euler", 
                   bicycle_parameter_dict=None):
    """
    Filter instrumented bicycle measurements from different sensors using
    an Unscented Kalman Filter.
    
    This requires N equally spaced samples of the following ten measurements:
        - p_x: Measured by GNSS
        - p_y: Measured by GNSS
        - psi: Derived from GNSS p_x and p_y
        - dpsi: Yaw rate from IMU
        - v: speed from IMU
        - dv: acceleration from IMU
        - delta: steer angle from the steer encoder
        - ddelta: steer rate from the steer encoder
        - phi: roll angle from the IMU
        - dphi: roll rate from the IMU
        
    NaN values indicate where the GNSS does not return a value due to lower
    sampling rate. 
        
    Based on the dynamic whipple-carvallo bicycle model with assuming zero 
    steer and roll torque.

    Parameters
    ----------
    measurements : array-like
        Measurement array of shape (N, 10).
    R : array-like
        Measurement noise matrix (10, 10).
    Q : array-like
        Process noise matrix (10, 10).
    t_s : float, optional
        Sample time. The default is 0.01.
    smooth : bool, optional
        Optionally smooth the filter result with a Rauch-Tung-Striebel smoother
        (recommended). The default is True.
    plot : bool, optional
        Plot the filter results. The default is False.
    integration_method: str, optional
        The integration method for the integration of the system dynamics in 
        the prediction step. Can be 'backward euler' or 'midpoint'. The default
        is 'backward euler'.
    bicycle_parameter_dict : dict
        A dictionary of bicycle parameters as returned by the bicycleparameters
        toolbox. Use this to customize the bike. 

    Returns
    -------
    states_filtered/smoothed : array-like
        Filtered (or smoothed) measurements (N, 10)

    """
    

    state_labels = ['p_x', 'p_y', 'psi', 'v', 'phi', 'delta', 'dpsi', 'dphi', 
                    'ddelta', 'dv']
    n_states = len(state_labels)
    n_samples = measurements.shape[0]-1
    n_measurements = n_states

    #make a time vector
    t = np.arange(0, n_samples+1) * t_s
    
    # intial conditions
    x0 = measurements[0,:]
    
    if bicycle_parameter_dict is None:
        bicycle_parameter_dict = meijaard2007_browser_jason
        warn_wrong_parameters()
        
    bp_param = Meijaard2007ParameterSet(bicycle_parameter_dict, True)
    bp_model = Meijaard2007Model(bp_param)
    
    # setup filter
    points = MerweScaledSigmaPoints(n_states, alpha=.1, beta=2., kappa=-1)
    
    def move(x, t_s):
        return move_dynamic(x, t_s, bp_model, 
                            integration_method=integration_method)

    ukf = UnscentedKalmanFilter(dim_x=n_states, dim_z=n_measurements, 
                                fx=move, hx=measure_dynamic, 
                                points=points, dt=t_s)
    
    ukf.x = x0
    ukf.P *= 0.2
    ukf.R = R
    ukf.Q = Q
    
    states_filtered = np.zeros((n_samples+1, n_states))
    states_filtered[0,:] = x0
    
    covs_filtered = np.zeros((n_samples+1, ukf.P.shape[0], ukf.P.shape[1]))
    covs_filtered[0,:,:] = ukf.P

    # filter measurements 
    for i in range(n_samples):
        #predict
        ukf.predict()
        
        #update
        m = measurements[i+1,:]

        gnss_available = np.all(np.isfinite(m))
        if not gnss_available:
            m[np.logical_not(np.isfinite(m))] = 0
            
        ukf.update(m, gnss_available = gnss_available)
        
        states_filtered[i+1,:] = ukf.x
        covs_filtered[i+1,:,:] = ukf.P

    states_filtered = np.array(states_filtered)
        
    # smooth filtered states using an Rauch-Tung-Striebel smoother. 
    if smooth:
        states_smoothed, covs_smoothed, K = ukf.rts_smoother(states_filtered, 
                                                             covs_filtered)
    
    #plot filter results
    if plot:
        
        #x-y plot
        fig0, ax0 = plt.subplots(1,1)
        ax0.set_aspect('equal')
        finite = np.logical_not(measurements[:,1]==0)
        ax0.plot(measurements[:,0][finite], measurements[:,1][finite], 
                 color='blue', label='measured')
        if smooth:
            ax0.plot(states_smoothed[:,0], states_smoothed[:,1], 
                     color = 'green', label='smoothed')
        
        #time plot
        fig1, axes1 = plt.subplots(n_states, 1, sharex=True)
        ax0.plot(states_filtered[:,0], states_filtered[:,1], 
                 color = 'orange', label='filtered')
            
        for i in range(n_states):
            axes1[i].set_ylabel(state_labels[i])
            
            m = measurements[:,i]
            finite = np.logical_not(m==0)
            axes1[i].plot(t[finite], m[finite], 
                          color = 'blue', label = 'data')
            axes1[i].plot(t, states_filtered[:,i], 
                          color = 'orange', label = 'filtered')
            if smooth:
                axes1[i].plot(t, states_smoothed[:,i], 
                              color = 'green', label = 'smoothed')
            
    if smooth:
        return states_smoothed
    else:
        return states_filtered
        

def filter_kinematic(measurements, R, Q, t_s=0.01):
    """ UNTESTED, PROBABLY DOESN'T WORK
    """
    
    state_labels = ['p_x', 'p_y', 'psi', 'dpsi', 'v', 'dv']
    n_states = 6
    n_samples = measurements.shape[0]-1

    t = np.arange(0, (n_samples+1)*t_s, t_s)
    
    # figures 
    fig0, ax0 = plt.subplots(1,1)
    ax0.set_aspect('equal')
    finite = np.isfinite(measurements[:,0])
    ax0.plot(measurements[:,0][finite], measurements[:,1][finite], 
             color='blue', label='measured')
    
    fig1, axes1 = plt.subplots(n_states, 1, sharex=True)

    # inital state
    x0 = measurements[0,:]
    
    points = MerweScaledSigmaPoints(6, alpha=.1, beta=2., kappa=-1)

    ukf = UnscentedKalmanFilter(dim_x=6, dim_z=6, fx=move_kinematic, 
                                hx=measure_kinematic, points=points, dt=t_s)

    ukf.x = x0
    ukf.P *= 0.2
    ukf.R = R
    ukf.Q = Q        

    states_filtered = [x0]
    covs_filtered = ukf.P[np.newaxis,:,:]

    for i in range(n_samples):
        #predict
        ukf.predict()
        
        #update
        m = measurements[i+1,:]

        gnss_available = np.all(np.isfinite(m))
        if not gnss_available:
            m[np.logical_not(np.isfinite(m))] = 0
            
        ukf.update(m, gnss_available = gnss_available)
        
        states_filtered.append(ukf.x)
        covs_filtered = np.concatenate((covs_filtered, ukf.P[np.newaxis,:,:]), 
                                       axis=0)

    states_filtered = np.array(states_filtered)
    ax0.plot(states_filtered[:,0], states_filtered[:,1], 
             color = 'orange', label='filtered')
        
    for i in range(n_states):
        axes1[i].set_ylabel(state_labels[i])
        
        m = measurements[:,i]
        finite = np.logical_not(m==0)
        axes1[i].plot(t[finite], m[finite], 
                      color = 'blue', label = 'data')
        axes1[i].plot(t, states_filtered[:,i], 
                      color = 'orange', label = 'filtered')
        axes1[i].grid()
        
    #smooth

    states_smoothed, covs_smoothed, K = ukf.rts_smoother(states_filtered, 
                                                         covs_filtered)

    ax0.plot(states_smoothed[:,0], states_smoothed[:,1], 
             color = 'green', label='smoothed')
        
    for i in range(n_states):
        axes1[i].plot(t, states_smoothed[:,i], 
                      color = 'green', label = 'smoothed')