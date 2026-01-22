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

from sensorbike.params.balanceassist_bikeparams import balanceassistv1_with_averagerider
from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
from bicycleparameters.models import Meijaard2007Model


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
    
    # OTHER
    std_phi = meas['IMU']['roll']
    std_delta = meas['steerencoder']['angle']
    std_dpsi = meas['IMU']['gyro']
    std_dphi = meas['IMU']['gyro']
    std_ddelta = meas['steerencoder']['rate']
    std_v = meas['speedometer']['v']
    std_dv = meas['IMU']['accel']
    
    measurement_noise_std = np.array([0, 0, 0, 
                                    std_v, std_phi, std_delta, 
                                    std_dpsi, std_dphi, std_ddelta, 
                                    std_dv])  
    
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
                                    prcs['ddelta'], prcs['dv'], prcs['b_steer']])
    
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
                       "bicycle_parameter_dict": balanceassistv1_with_averagerider}
    
    sensor_std = {"IMU": {"roll":float(np.deg2rad(3.5)), # datasheet values
                           "yaw": float(np.deg2rad(3.5)),
                           "gyro": float(np.deg2rad(3.1)),
                           "accel": 1.5},   # inflated, true data seems more noisy then datasheet suggests
                   "speedometer": {"v": 0.05},  #guess
                   "steerencoder": {"rate": float(np.deg2rad(0.5)), "angle": float(np.deg2rad(1))}}  #guess    
        
    filter_settings['measurement_noise_std'] = \
        make_Rscale_from_sensorchar(sensor_std)  

     
    process_std = {"x": 1e-6, "y": 1e-6,           # no additional uncertainty in position dynamics
                    "psi": float(np.deg2rad(1)),   # moderate uncertainties to account for model simplifications
                    "phi": float(np.deg2rad(1)),
                    "delta":float(np.deg2rad(2)),    
                    "dpsi": float(np.deg2rad(10)), # large uncertainties in rates due to zero roll/steer torque assumption
                    "dphi": float(np.deg2rad(15)), 
                    "ddelta": float(np.deg2rad(20)),
                    "v": 0.5, #large uncertainties in speed and acceleration due to const. acceleration assumption                              
                    "dv": 1,
                    "b_steer": float(np.deg2rad(0.05))} # steer bias 
    
    filter_settings['process_noise_std'] = \
        make_Qscale_from_dict(process_std)   

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
        bicycle_parameter_dict = balanceassistv1_with_averagerider
    
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
    b_steer = x[10]
    
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
                  x_lat_pred[2], x_lat_pred[3], dv_pred, b_steer]
        
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
                  x_lat_pred[2], x_lat_pred[3], dv_pred, b_steer]
        
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

def measure_dynamic(x):
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
    """
    
    C = np.eye(10)
    C = np.c_[C, np.zeros(10)]

    #steer bias
    C[5,10] = 1 
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


def filter_dynamic(measurements, uncertainties, R, Q, t_s=0.01, smooth = True, plot = True, 
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
    uncertainties : array-like
        GNSS measurement uncertainties of shape (N, 4) with [var_x, var_y, cov_xy, var_psi].
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
                    'ddelta', 'dv', 'b_delta']
    n_states = len(state_labels)
    n_samples = measurements.shape[0]-1
    n_measurements = n_states

    #make a time vector
    t = np.arange(0, n_samples+1) * t_s

    # construct measurement noise matrix
    def get_measurement(i):
        #new measurements and uncertainties
        m_i = measurements[i,:].copy()
        R_i = R.copy()

        # check gnss availability
        meas_unavilable = np.argwhere(np.logical_not(np.isfinite(m_i))).flatten()

        m_i[meas_unavilable] = 0
        R_i[meas_unavilable, meas_unavilable] = 1e10
        
        if not (0 in meas_unavilable or 1 in meas_unavilable):
            R_i[:2,:2] = [[uncertainties[i,0], uncertainties[i,2]],
                          [uncertainties[i,2], uncertainties[i,1]]]
            
        if not (2 in meas_unavilable):
            R_i[2,2] = uncertainties[i,3]

        R_i[~np.isfinite(R_i)] = 1e10

        return m_i, R_i

    
    # intial conditions
    x0, P0_meas = get_measurement(0)
    x0 = np.r_[x0, 0]
    P0 = np.zeros((x0.size, x0.size))
    P0[:10, :10] = P0_meas
    P0[5,5] = np.pi**2   #any valid angle is a reasonable initial guess for the steer angle (due to potential bias)
    P0[-1, -1] = np.pi**2   #any valid angle is reasonable guess for the bias
    
    # setup bicycle parameters
    if bicycle_parameter_dict is None:
        bicycle_parameter_dict = balanceassistv1_with_averagerider
        
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
    ukf.P = P0
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

        # update
        m_i, R_i = get_measurement(i+1)
        ukf.update(m_i, R=R_i)
        
        states_filtered[i+1,:] = ukf.x
        covs_filtered[i+1,:,:] = ukf.P

    states_filtered = np.array(states_filtered)
        
    # smooth filtered states using an Rauch-Tung-Striebel smoother. 
    if smooth:
        states_smoothed, covs_smoothed, _ = ukf.rts_smoother(states_filtered, 
                                                             covs_filtered)
        
        states_smoothed[:, 3] = states_filtered[:, 3]
    
    #plot filter results
    if plot:
        
        #x-y plot
        fig0, ax0 = plt.subplots(1,1, layout='constrained')
        ax0.set_aspect('equal')
        finite = np.isfinite(measurements[:,1])
        ax0.plot(measurements[:,0][finite], measurements[:,1][finite], color='blue', label='measured')

        ax0.plot(states_filtered[:,0], states_filtered[:,1], color = 'orange', label='filtered')
        ax0.plot(states_filtered[:,0] + covs_filtered[:,0,0], 
                 states_filtered[:,1] + covs_filtered[:,1,1], color = 'orange', linestyle='--')
        ax0.plot(states_filtered[:,0] - covs_filtered[:,0,0], 
                 states_filtered[:,1] - covs_filtered[:,1,1], color = 'orange', linestyle='--')
        if smooth:
            ax0.plot(states_smoothed[:,0], states_smoothed[:,1], 
                     color = 'green', label='smoothed')
            ax0.plot(states_smoothed[:,0] + covs_smoothed[:,0,0], 
                     states_smoothed[:,1] + covs_smoothed[:,1,1], color = 'green', linestyle='--')
            ax0.plot(states_smoothed[:,0] - covs_smoothed[:,0,0], 
                     states_smoothed[:,1] - covs_smoothed[:,1,1], color = 'green', linestyle='--')
            
        ax0.set_xlabel('x [m]')
        ax0.set_ylabel('y [m]')
        ax0.set_title('Filtered xy-trajectories')
        ax0.legend()
        
        #time plot
        fig1, axes1 = plt.subplots(n_states, 1, sharex=True, layout='constrained')
            
        for i in range(n_states):
            axes1[i].set_ylabel(state_labels[i])

            minval = np.inf
            maxval = -np.inf
            
            if i < n_states-1:
                m = measurements[:,i]
                finite = np.isfinite(m)
                axes1[i].plot(t[finite], m[finite], color = 'blue', label = 'data')
                minval = min(minval, np.min(m))
                maxval = max(maxval, np.max(m))

            axes1[i].plot(t, states_filtered[:,i], color = 'orange', label = 'filtered')
            axes1[i].fill_between(t, states_filtered[:,i]+np.sqrt(covs_filtered[:,i,i]), 
                                     states_filtered[:,i]-np.sqrt(covs_filtered[:,i,i]), alpha=0.2, color='orange')
            minval = min(minval, np.min(states_filtered[:,i]))
            maxval = max(maxval, np.max(states_filtered[:,i]))

            if smooth and state_labels[i] != 'v':
                axes1[i].plot(t, states_smoothed[:,i], 
                              color = 'green', label = 'smoothed')
                axes1[i].fill_between(t, states_smoothed[:,i]+np.sqrt(covs_smoothed[:,i,i]), 
                                         states_smoothed[:,i]-np.sqrt(covs_smoothed[:,i,i]), alpha=0.2, color='green')
                minval = min(minval, np.min(states_smoothed[:,i]))
                maxval = max(maxval, np.max(states_smoothed[:,i]))

            rng = maxval - minval
            axes1[i].set_ylim(minval - 0.2 * rng, maxval + 0.2 * rng)
        axes1[0].legend()
        axes1[-1].set_xlabel('t [s]')
        axes1[0].set_title("Filtered Trajectories")
        fig1.set_size_inches(10.5, 9)
            
    if smooth:
        steer_angle_bias = np.median(states_smoothed[:,-1])
        states_smoothed = states_smoothed[:,:-1]
        return states_smoothed, steer_angle_bias
    else:
        steer_angle_bias = np.median(states_smoothed[:,-1])
        states_filtered = states_filtered[:,:-1]
        return states_filtered, steer_angle_bias
        

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