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
from sensorbike.geometry import InstrumentedBikeGeometry
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


def make_R(sensor_characteristics):
    """
    Creates the constant part of the measurement noise matrix R from a nested dict 
    containting the standard errors of each measurement sorted by sensor.
    
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
    
    The GNSS is omitted/ignored. Instead, we later plug in dynamic uncertainty estimates.
    Designed for easy parsing of yaml config files. 

    The 12 measurements are
    |    gnss    | steer encoder  |          imu          |ws|  |          
    [x, y, vx, vy, delta, deltadot, phi, gyrox, psi, gyroz, v, a]

    
    Parameters
    ----------
    measurement_noise_dict : dict
        Nested dictionary of measurement noise standard errors.

    Returns
    -------
    R : array
        Array (12x12) of static measurement variances. 
    """

    R = np.zeros((12,12), dtype=float)

    # steer encoder
    R[4,4] = sensor_characteristics['steerencoder']['angle']**2
    R[5,5] = sensor_characteristics['steerencoder']['rate']**2

    # imu
    R[6,6] = sensor_characteristics['IMU']['roll']**2
    R[7,7] = sensor_characteristics['IMU']['gyro']**2
    R[8,8] = sensor_characteristics['IMU']['yaw']**2
    R[9,9] = sensor_characteristics['IMU']['gyro']**2
    R[11,11] = sensor_characteristics['IMU']['accel']**2

    #wheelspeed sensor
    R[10,10] = sensor_characteristics['speedometer']['v']**2
    
    return R
    
def make_Q(process_noise_dict):
    """
    Creates the process noise matrix Q from a dict containting the 
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
                                    prcs['ddelta'], prcs['dv'], 
                                    prcs['ang_bias'], prcs['ang_bias']])
    
    Q = np.diag(process_noise_std**2)
    
    return Q


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
    
    sensor_std = {"gnss": {"var_inflation_factor": 10},
                   "IMU": {"roll":float(np.deg2rad(60)),  # little confidence and potential frame misalignment -> inflate
                           "yaw": float(np.deg2rad(3.5)), # datasheet values
                           "gyro": float(np.deg2rad(3.1)),
                           "accel": 1.5},   # inflated, true data seems more noisy then datasheet suggests
                   "speedometer": {"v": 0.05},  #guess
                   "steerencoder": {"rate": float(np.deg2rad(0.5)), "angle": float(np.deg2rad(1))}}  #guess   
     
        
    #filter_settings['R'] = make_R(sensor_std)  
     
    process_std = {"x": 1e-6, "y": 1e-6,                # no additional uncertainty in position dynamics
                   "psi": float(np.deg2rad(1)),         # moderate uncertainties to account for model simplifications
                   "phi": float(np.deg2rad(0.1)),
                   "delta":float(np.deg2rad(2)),    
                   "dpsi": float(np.deg2rad(10)),       # large uncertainties in rates due to zero roll/steer torque assumption
                   "dphi": float(np.deg2rad(10)), 
                   "ddelta": float(np.deg2rad(20)),
                   "v": 0.5,                            # large uncertainties in speed and acceleration due to const. accel. assumption                              
                   "dv": 1,
                   "ang_bias": float(np.deg2rad(0.01))} # small -> roll/steer bias should converge to const.
    
    #filter_settings['Q'] = make_Q(process_std)   

    filter_settings['process_std'] = process_std
    filter_settings['sensor_std'] = sensor_std

    return filter_settings


def parse_filter_settings(filter_settings_yaml_dict, 
                          bicycle_parameter_dict=None):
    """
    Parse a yaml dict of the filter settings. Return the dict format required
    by the filter functions. 
    
    The yaml must be structured:
        integration_method: "midpoint" or "backward euler"
        measurement_noise_std:
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
        
    filter_settings['R'] = make_R(filter_settings_yaml_dict["measurement_noise_std"])
    filter_settings['Q'] = make_Q(filter_settings_yaml_dict['process_noise_std'])
        
    return filter_settings
    

def move(x, t_s, bp_model, integration_method='euler'):
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
    b_psi_imu = x[10]
    #b_phi = x[11]
    b_delta = x[11]
    
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
                  x_lat_pred[2], x_lat_pred[3], dv_pred, b_psi_imu, b_delta]
        
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
                  x_lat_pred[2], x_lat_pred[3], dv_pred, b_psi_imu, b_delta]
        
    else:
        raise ValueError("Unknown integration method!")
    
    return x_pred


def measure(x, bicycle_geometry):
    """ Apply the measurement model to the current state estimate.
    
    The state vector is [x, y, psi, v, phi, delta, dpsi, dphi, ddelta, dv].

    Parameters
    ----------
    x : array-like
        State vector.

    Returns
    -------
    measurement : array
        Measrurement 
    """

    meas_gnss = bicycle_geometry.transform_statesN2gnss(x[:10])
    meas_can = bicycle_geometry.transform_statesN2can(x[:10])
    
    #add biases 
    meas_can[0] += x[11]    # delta
    #meas_can[2] += x[11]    # phi
    meas_can[4] += x[10]    # psi

    return np.r_[meas_gnss, meas_can]


def filter(measurements, uncertainties_gnss, 
                   R, Q,
                   t_s=0.01, smooth = True, plot = True, 
                   integration_method = "backward euler", 
                   bicycle_parameter_dict=None, 
                   bicycle_geometry=None,
                   color_filtered='#'):
    """
    Filter instrumented bicycle measurements from different sensors using
    an Unscented Kalman Filter.
    
    This requires N equally spaced samples of the following ten measurements:
        - p_x: Measured by GNSS
        - p_y: Measured by GNSS
        - psi_gnss: Derived from GNSS p_x and p_y
        - psi_imu: Derived from IMU but heavily biased.
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
        Measurement array of shape (N, 11).
    uncertainties : array-like
        GNSS measurement uncertainties of shape (N, 4) with [var_x, var_y, cov_xy, var_psi].
    R : array-like
        Measurement noise matrix (11, 11).
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

    if bicycle_geometry is None:
        bicycle_geometry = InstrumentedBikeGeometry()

    state_labels = ['p_x', 'p_y', 'psi', 'v', 'phi', 'delta', 'dpsi', 'dphi', 
                    'ddelta', 'dv', 'b_psi_imu', 'b_delta_enc'] #'b_phi_imu', 
    n_states = len(state_labels)
    n_samples = measurements.shape[0]-1
    n_measurements = n_states

    # make a time vector
    t = np.arange(0, n_samples+1) * t_s

    # R update function
    var_vwsp = R[10,10].copy()
    var_psiimu = R[8,8].copy()

    def update_R(i, R):
        """Update the measurement variance matrix according to GNSS availablility at
        time step i.
        """

        # check availability
        pos_gnss_available = np.all(np.isfinite(measurements[i,0:2]))
        vel_gnss_available = np.all(np.isfinite(measurements[i,2:4]))

        # if gnss position is available plug in RTKLib covariance estimates. Else, make unreliable
        if pos_gnss_available:
            R[:2,:2] = [[uncertainties_gnss[i,0], uncertainties_gnss[i,4]],
                        [uncertainties_gnss[i,4], uncertainties_gnss[i,1]]]
        else:
            R[:2,:2] = 1e10 * np.eye(2)

        # if gnss velocity is available plug in RTKLib variance estimates and make wheelspeed/imu orientation unreliable. 
        if vel_gnss_available:
            R[2:4,2:4] = [[uncertainties_gnss[i,2], uncertainties_gnss[i,5]],
                          [uncertainties_gnss[i,5], uncertainties_gnss[i,3]]]
        else: 
            R[2:4,2:4] = 1e10 * np.eye(2)      
            R[8,8] = var_psiimu
            R[10,10] = var_vwsp
        
        return R


    def transform_measurement2state(m_i):
        """Build a state vector from measurements. Select GNSS orientation over CAN and 
        wheelspeed over GNSS speed."""

        states_E_can = np.zeros(10)
        states_E_can[2] = np.arctan2(m_i[3], m_i[2])
        states_E_can[4] = m_i[4]
        states_N_can = bicycle_geometry.transform_statesE2statesN(states_E_can)

        states_N_can = bicycle_geometry.transform_can2statesN(m_i[4:], states_N_can)

        states_N_can[2] = - np.arctan2(m_i[3], m_i[2])
        states_N_gnss = bicycle_geometry.transform_gnss2statesN(m_i[:4], states_N_can)

        states_N = np.r_[states_N_gnss[:3], states_N_can[3:]]
        meas_gnss = bicycle_geometry.transform_statesN2gnss(states_N)

        #states_N = np.r_[states_N_gnss[:2], states_N_can[2:]]
        #meas_can = bicycle_geometry.transform_statesN2can(states_N)

        return states_N

    # intial state from first measurement
    x0 = transform_measurement2state(measurements[0,:])
    x0 = np.r_[x0, [0, 0]]       # add yaw, roll, and steer angle bias      

    P0 = np.zeros((x0.size, x0.size), dtype=float)
    P0[:2,:2] = [[uncertainties_gnss[0,0], uncertainties_gnss[0,4]],
                 [uncertainties_gnss[0,4], uncertainties_gnss[0,1]]]
    
    P0[2:10, 2:10] = R[4:12, 4:12]
    P0[4,4] = np.deg2rad(3.5)**2         # should be small although confidence in sensor is not big. Otherwise, filter will jump till first GNSS.
    P0[5,5] = np.deg2rad(30)**2

    P0[[10,11], [10,11]] = (3*np.pi)**2  # inflate var for yaw/steer bias: could be any valid angle 

    # setup bicycle parameters
    if bicycle_parameter_dict is None:
        bicycle_parameter_dict = balanceassistv1_with_averagerider
    bp_param = Meijaard2007ParameterSet(bicycle_parameter_dict, True)
    bp_model = Meijaard2007Model(bp_param)

    # setup filter
    points = MerweScaledSigmaPoints(n_states, alpha=.1, beta=2., kappa=-1)
    
    def wrap_move(x, t_s):
        return move(x, t_s, bp_model, integration_method=integration_method)

    ukf = UnscentedKalmanFilter(dim_x=n_states, dim_z=n_measurements, 
                                fx=wrap_move, hx=measure, 
                                points=points, dt=t_s)
    
    ukf.x = x0
    ukf.P = P0
    ukf.Q = Q
    
    states_filtered = np.zeros((n_samples+1, n_states))
    states_filtered[0,:] = x0
    
    covs_filtered = np.zeros((n_samples+1, ukf.P.shape[0], ukf.P.shape[1]))
    covs_filtered[0,:,:] = ukf.P

    state_measurements = []

    # filter measurements 
    for i in range(n_samples):
        #predict
        ukf.predict()

        # update
        m = measurements[i+1,:].copy()
        m[~np.isfinite(m)] = 0

        R = update_R(i+1, R)

        ukf.update(m, R=R, bicycle_geometry=bicycle_geometry)

        state_measurements.append(measure(ukf.x, bicycle_geometry))
        
        states_filtered[i+1,:] = ukf.x
        covs_filtered[i+1,:,:] = ukf.P

    states_filtered = np.array(states_filtered)
    state_measurements = np.array(state_measurements)
        
    # smooth filtered states using an Rauch-Tung-Striebel smoother. 
    if smooth:
        states_smoothed, covs_smoothed, _ = ukf.rts_smoother(states_filtered, 
                                                             covs_filtered)
        
        # never used smoothed speed. May be unreliable
        states_smoothed[:, 3] = states_filtered[:, 3]
    
    if plot:
        _plot_measurement_error(measurements, state_measurements)
        figxy, axxy = plot_filter_result_xy(measurements, states_filtered, covs_filtered, bicycle_geometry, states_smoothed=states_smoothed, covs_smoothed=covs_smoothed)
        figt, axest = plot_filter_result_t(measurements, states_filtered, covs_filtered, bicycle_geometry, states_smoothed=states_smoothed, covs_smoothed=covs_smoothed)
        
    out = [states_filtered, covs_filtered]
    if smooth:
        out += [states_smoothed, covs_smoothed]
    if plot:
        out += [figt, axest, figxy, axxy]

    return out
    

def _plot_measurement_error(measurements, state_measurements):

    features_track_gnss = ["x_gnss", "y_gnss", "vx_gnss", "vy_gnss"]
    features_track_can = ["delta_can", "ddelta_can", "phi_can", "gyrox_can", "psi_can", "gyroz_can", "vrws_can", "ax_can"]
    features = features_track_gnss + features_track_can

    t = np.arange(0, measurements.shape[0])

    fig, axes = plt.subplots(measurements.shape[1], 1, sharex=True, layout='constrained')
    for i in range(measurements.shape[1]):
        fin = np.isfinite(measurements[:,i])
        axes[i].plot(t[fin], measurements[fin,i])
        axes[i].plot(state_measurements[:,i])
        axes[i].set_ylabel(features[i])

    axes[0].set_title("gnss measurements")
    axes[len(features_track_gnss)].set_title("can measurements")

def plot_filter_result_xy(measurements, 
                         states_filtered, covs_filtered, bicycle_geometry,
                         states_smoothed=None, covs_smoothed=None, 
                         color_meas_gnss='#EC6842',
                         color_state_gnss='#5C5C5C',
                         color_filtered='#FFB81C', color_smoothed='#009B77', ax=None):
    """Plot the filtered/smoothed states into an x/y plot.
    """

    smooth = not(states_smoothed is None) and not(covs_smoothed is None)

    plotstyle_filt = dict(linewidth=1)
    plotstyle_cov = dict(linewidth=1, linestyle="--")
    plotstyle_meas = dict(marker='.', markersize=2, linestyle='None')

    #remove biases
    #if smooth:
    #    measurements[:, (4, 6, 8)] -= states_smoothed
    meas_states_can = np.zeros((measurements.shape[0], 10), dtype=float)
    meas_states_can[:,2] = measurements[:,8]
    meas_states_can[:,4] = measurements[:,6]
    meas_states_can = bicycle_geometry.transform_can2statesN(measurements[:,4:], states_smoothed[:,:10])
    meas_states_gnss = bicycle_geometry.transform_gnss2statesN(measurements[:,:4], states_smoothed[:,:10])

    if ax is None:
        fig, ax = plt.subplots(1,1, layout='constrained')
        ax.set_aspect('equal')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title("filtered and smoothed bicycle position")
    else:
        fig = ax.get_figure()

    #x-y plot
    fin_gnss = np.logical_and(np.isfinite(meas_states_gnss[:,0]), np.isfinite(meas_states_gnss[:,1]))
    ax.plot(meas_states_gnss[fin_gnss,0], meas_states_gnss[fin_gnss,1], color= color_state_gnss, label = 'gnss (transf.)', **plotstyle_meas)
    ax.plot(measurements[fin_gnss, 0], -measurements[fin_gnss, 1], color= color_meas_gnss, label = 'gnss (meas.)', **plotstyle_meas)

    ax.plot(states_filtered[:,0], states_filtered[:,1], color = color_filtered, label='ukf', **plotstyle_filt)
    ax.plot(states_filtered[:,0] + np.sqrt(covs_filtered[:,0,0]), 
            states_filtered[:,1] + np.sqrt(covs_filtered[:,1,1]), color = color_filtered, **plotstyle_cov)
    ax.plot(states_filtered[:,0] - np.sqrt(covs_filtered[:,0,0]), 
            states_filtered[:,1] - np.sqrt(covs_filtered[:,1,1]), color = color_filtered, **plotstyle_cov)
    if smooth:
        ax.plot(states_smoothed[:,0], states_smoothed[:,1], color = color_smoothed, label='rts', **plotstyle_filt)
        ax.plot(states_smoothed[:,0] + np.sqrt(covs_smoothed[:,0,0]), 
                states_smoothed[:,1] + np.sqrt(covs_smoothed[:,1,1]), color = color_smoothed, **plotstyle_cov)
        ax.plot(states_smoothed[:,0] - np.sqrt(covs_smoothed[:,0,0]), 
                states_smoothed[:,1] - np.sqrt(covs_smoothed[:,1,1]), color = color_smoothed, **plotstyle_cov)
        
    ax.legend()

    return fig, ax


def plot_filter_result_t(measurements, 
                         states_filtered, covs_filtered, bicycle_geometry,
                         states_smoothed=None, covs_smoothed=None, 
                         color_meas_can='#A50034',
                         color_meas_gnss='#EC6842',
                         color_filtered='#FFB81C', color_smoothed='#009B77', axes=None, t_s=0.01):
    """Plot the filtered/smoothed states into an time plot.
    """
    state_labels = ['p_x', 'p_y', 'psi', 'v', 'phi', 'delta', 'dpsi', 'dphi', 
                'ddelta', 'dv', 'b_psi_imu', 'b_phi_imu', 'b_delta_enc']

    smooth = not(states_smoothed is None) and not(covs_smoothed is None)

    plotstyle_filt = dict(linewidth=1)
    plotstyle_meas = dict(marker='.', markersize=1, linestyle='None')

    #remove biases
    #if smooth:
    #    measurements[:, (4, 6, 8)] -= states_smoothed

    meas_states_can = np.zeros((measurements.shape[0], 10), dtype=float)
    meas_states_can[:,2] = measurements[:,8]
    meas_states_can = bicycle_geometry.transform_can2statesN(measurements[:,4:], meas_states_can)
    meas_states_gnss = bicycle_geometry.transform_gnss2statesN(measurements[:,:4], meas_states_can)

    t = np.arange(0, measurements.shape[0]) * t_s

    n_states = meas_states_can.shape[1]
    n_biases = states_filtered.shape[1] - n_states

    if axes is None:
        fig, axes = plt.subplots(n_states, 1, layout='constrained', sharex=True)
        fig_biases, axes_biases = plt.subplots(n_biases, 1, layout='constrained', sharex=True)
        axes = np.r_[axes, axes_biases]
    else:
        fig = axes[0].get_figure()

    fig_biases, axes_biases = plt.subplots(n_biases, 1, layout='constrained', sharex=True)
    axes = np.r_[axes, axes_biases]
    
    for i in range(n_states+n_biases):
        minval = np.inf
        maxval = -np.inf

        if i < n_states:
            fin_can = np.isfinite(meas_states_can[:,i])
            if np.any(fin_can):
                axes[i].plot(t[fin_can], meas_states_can[fin_can,i], color= color_meas_can, label = 'can (transf.)', **plotstyle_meas)
                minval = min(minval, np.min(meas_states_can[fin_can,i]))
                maxval = max(maxval, np.max(meas_states_can[fin_can,i]))

            fin_gnss = np.isfinite(meas_states_gnss[:,i])
            if np.any(fin_gnss):
                axes[i].plot(t[fin_gnss], meas_states_gnss[fin_gnss,i], color= color_meas_gnss, label = 'gnss (transf.)', **plotstyle_meas)
                minval = min(minval, np.min(meas_states_gnss[fin_gnss,i]))
                maxval = max(maxval, np.max(meas_states_gnss[fin_gnss,i]))

        axes[i].plot(t, states_filtered[:,i], color = color_filtered, label = 'ukf', **plotstyle_filt)
        axes[i].fill_between(t, states_filtered[:,i]+np.sqrt(covs_filtered[:,i,i]), 
                                states_filtered[:,i]-np.sqrt(covs_filtered[:,i,i]), alpha=0.2, color=color_filtered)
        minval = min(minval, np.min(states_filtered[:,i]))
        maxval = max(maxval, np.max(states_filtered[:,i]))

        if smooth:
            axes[i].plot(t, states_smoothed[:,i], color = color_smoothed, label = 'rts', **plotstyle_filt)
            axes[i].fill_between(t, states_smoothed[:,i]+np.sqrt(covs_smoothed[:,i,i]), 
                                    states_smoothed[:,i]-np.sqrt(covs_smoothed[:,i,i]), alpha=0.2, color=color_smoothed)
            minval = min(minval, np.min(states_smoothed[:,i]))
            maxval = max(maxval, np.max(states_smoothed[:,i]))

        rng = maxval - minval
        axes[i].set_ylim(minval - 0.2 * rng, maxval + 0.2 * rng)
        axes[i].set_ylabel(state_labels[i])
        axes[i].grid()

    axes[n_states-1].legend()
    axes[-1].set_xlabel('t [s]')
    if smooth:
        axes[0].set_title("filtered and smoothed bicycle states")
    else:
        axes[0].set_title("filtered and smoothed bicycle states")
    axes[n_states].set_title("biases")
    fig.set_size_inches(10.5, 9)

    return fig, axes
        
