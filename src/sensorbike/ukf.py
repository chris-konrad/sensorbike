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
    Creates the time-constant part of the measurement noise matrix R from a nested dict 
    containting the standard errors of each measurement sorted by sensor.
    
    The dict must be structured:
        gnss:
            position: sigma_pos     (Will be added to the dynamic uncertainties)
            velocity: sigma_vel     (Will be added to the dynamic uncertainties)
        imu: 
            gyro: sigma_gyro
            accely: sigma_accel_y
            accelz: sigma_accel_z
        speedometer:
            v: sigma_v
        steerencoder:
            rate: sigma_rate
            angle: sigma_angle
    
    All sigmas refer to time-constant measurement errors in the
    unit of their respective measurement.

    The 11 measurements are
    |    gnss    | steer encoder  |          imu        |ws|          
    [x, y, vx, vy, delta, deltadot, gyrox, gyroz, ay, ay, v]

    Use get_default_filter_settings() for defaults.
    
    Parameters
    ----------
    measurement_noise_dict : dict
        Nested dictionary of measurement noise standard errors.

    Returns
    -------
    R : array
        Array (11x11) of time-constant measurement variances. 
    """

    R = np.zeros((11,11), dtype=float)

    #gnss
    R[0,0] = sensor_characteristics['gnss']['position']**2
    R[1,1] = sensor_characteristics['gnss']['position']**2
    R[2,2] = sensor_characteristics['gnss']['velocity']**2
    R[3,3] = sensor_characteristics['gnss']['velocity']**2

    # steer encoder
    R[4,4] = sensor_characteristics['steerencoder']['angle']**2
    R[5,5] = sensor_characteristics['steerencoder']['rate']**2

    # imu
    R[6,6] = sensor_characteristics['imu']['gyro']**2
    R[7,7] = sensor_characteristics['imu']['gyro']**2
    R[8,8] = sensor_characteristics['imu']['accely']**2
    R[9,9] = sensor_characteristics['imu']['accelz']**2

    #wheelspeed sensor
    R[10,10] = sensor_characteristics['speedometer']['v']**2
    
    return R
    

def make_Q(process_noise_dict):
    """
    Creates the process noise matrix Q from a dict containting the 
    standard errors of each state.
    
    Must contain the process noise of the states [x, y, psi, v, dv, phi, dphi, psi, dpsi, delta, ddelta],
    as well as:
        eps : process noise of the IMU misalignment (should be very small)
        b_delta : process noise of the steer bias (should be very small)
        b_gyro : process noise of gyro biases (should be very small)

    Use get_default_filter_settings() for defaults.
    
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
                                  prcs['ddelta'], 
                                  prcs['eps'], prcs['eps'], prcs['eps'], 
                                  prcs['b_delta'], prcs['b_gyro'], prcs['b_gyro']])
    
    Q = np.diag(process_noise_std**2)
    
    return Q


def get_default_filter_settings():
    """
    Create the default filter settings based on sensor noise estimates.
    
    IMU standard errors are based on datasheet values from 
    https://docs.sparkfun.com/SparkFun_VR_IMU_Breakout_BNO086_QWIIC/
    assets/component_documentation/BNO080_085-Datasheet_v1.16.pdf
    
    GNSS, speedometer and steerencoder values are guesses. 

    See inline comments for reasoning about the default settings.

    Result can be passed to make_R and make_Q

    Returns
    -------
    filter_settings : dict
    """
    
    filter_settings = {"integration_method": "midpoint",
                       "bicycle_parameter_dict": balanceassistv1_with_averagerider,
                       "convergence_period": 0.1,          # time in s at the begining of the signal to discard to give the filter time to coverge.
                       }
    
    sensor_std = {"gnss": {"position": 0.015,             # additionaly noise due to wobbly pole, guess; Will be added to RTKLib uncertainties 
                           "velocity": np.sqrt((0.015**2)/(2*0.1**2))}, # error propagation of wobbly pole noise
                   "imu": {"roll":float(np.deg2rad(60)),  # unused
                           "yaw": float(np.deg2rad(3.5)), # unused
                           "gyro": float(np.deg2rad(3.1)),# datasheet value
                           "accely": 0.7,                 # datasheet value 0.35 slightly increased due to road roughness
                           "accelz": 5},                  # datasheet value 0.35 strongly increased due to road roughness
                   "speedometer": {"v": 0.1},             # guess
                   "steerencoder": {"rate": float(np.deg2rad(0.5)), "angle": float(np.deg2rad(1))}}  # guess   
     
        
    #filter_settings['R'] = make_R(sensor_std)  
     
    process_std = {"x": 1e-6, "y": 1e-6,                # no additional uncertainty in position dynamics
                   "psi": float(np.deg2rad(1)),         # moderate uncertainties to account for model simplifications
                   "phi": float(np.deg2rad(0.2)),
                   "delta":float(np.deg2rad(1)),    
                   "dpsi": float(np.deg2rad(20)),       # large uncertainties due to missing angular rates in gyroz measurement model
                   "dphi": float(np.deg2rad(15)),       # large uncertainties in roll and steer rates due to zero roll/steer torque assumption
                   "ddelta": float(np.deg2rad(40)),
                   "v": 1,                              # large uncertainties in speed and acceleration due to const. accel. assumption                              
                   "b_delta": float(np.deg2rad(0.01)),  # tiny, steer bias should be const.
                   "b_gyro": float(np.deg2rad(0.01)),   # unusedtiny, gyro bias should be const.
                   "eps": float(np.deg2rad(0.01))}      # tiny, imu<->bike rotation should be constant
    
    #filter_settings['Q'] = make_Q(process_std)   

    filter_settings['process_std'] = process_std
    filter_settings['sensor_std'] = sensor_std

    return filter_settings


def get_yaml_filter_settings(filter_settings_yaml_dict, 
                          bicycle_parameter_dict=None):
    """
    Parse a yaml dict of the filter settings. Return the dict format required
    by the filter functions. 
    
    The yaml must be structured:
        integration_method: "midpoint" or "backward euler"
        measurement_noise_std:
            IMU: 
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
            eps: sigma_epsx/y/z     
            b_steer: sigma_bsteer

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
    
    The state vector is [x, y, psi, v, phi, delta, dpsi, dphi, ddelta, epsx, epsy, epsz, bias_steer].
    
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
    eps = x[9:12]
    biases = x[12:15]
    
    #bicycle lateral dynamics
    x_lat = np.array([phi, delta, dphi, ddelta, psi])
    
    if integration_method == 'euler':
        
        #speed random walk
        v_pred = v

        #bicycle lateral dynamics
        A, B, C, D = get_statespace_matrices(bp_model, v)
        x_lat_pred = x_lat + t_s * A @ x_lat
        A_pred, B, C, D = get_statespace_matrices(bp_model, v_pred)
        dpsi_pred = (A_pred @ x_lat_pred)[-1]
        
        # forward dynamics
        p_x_pred =  v * np.cos(psi) * t_s + p_x
        p_y_pred =  v * np.sin(psi) * t_s + p_y
        
    elif integration_method == 'midpoint':
        
        #speed random walk
        v_pred = v 
        
        #bicycle lateral dynamics
        A_h2, B, C, D = get_statespace_matrices(bp_model, v_pred)
        I = np.eye(x_lat.size)        
        x_lat_pred = np.linalg.inv(I - t_s/2 * A_h2) \
            @ (I + t_s/2 * A_h2) @ x_lat
        psi_pred = x_lat_pred[-1]
        
        A_h, B, C, D = get_statespace_matrices(bp_model, v_pred)
        dx_lat_pred = (A_h @ x_lat_pred)
        dpsi_pred = dx_lat_pred[4]

        # forward dynamics
        p_x_pred = v_pred * np.cos((psi_pred + psi) / 2) * t_s \
            + p_x
        p_y_pred = v_pred * np.sin((psi_pred + psi) / 2) * t_s \
            + p_y 
        
    else:
        raise ValueError("Unknown integration method!")
    
    #pack
    x_pred = np.array([p_x_pred, p_y_pred, x_lat_pred[4], v_pred, 
                x_lat_pred[0], x_lat_pred[1], dpsi_pred,
                x_lat_pred[2], x_lat_pred[3]])
    x_pred = np.r_[x_pred, eps, biases]
    
    return x_pred


def measure(x, bicycle_geometry, estimate_gyro_biases, estimate_imu_orient_misalignment):
    """ Apply the measurement model to the current state estimate.
    
    The state vector is [x, y, psi, v, phi, delta, dpsi, dphi, ddelta, epsx, epsy, epsz, bias_steer].

    Parameters
    ----------
    x : array-like
        State vector.

    Returns
    -------
    measurement : array
        Measrurement 
    """
    if estimate_imu_orient_misalignment:
        eps = x[9:12]
    else:
        eps = np.zeros(3)

    meas_gnss = bicycle_geometry.transform_states2gnss(x[:9])
    meas_can = bicycle_geometry.transform_states2can(x[:9], eps)
    
    #add biases 
    meas_can[0] += x[12]    # delta
    if estimate_gyro_biases:
        meas_can[2] += x[13]    # gyrox
        meas_can[3] += x[14]    # gyroz

    return np.r_[meas_gnss, meas_can]


def filter(measurements, uncertainties_gnss, 
            R, Q,
            t_s=0.01, smooth = True, plot = True, plot_meas_error = True,
            integration_method = "backward euler", 
            bicycle_parameter_dict=None, 
            bicycle_geometry=None,
            estimate_gyro_biases=False,
            estimate_imu_orient_misalignment=False):
    """
    Filter instrumented bicycle measurements from different sensors using
    an Unscented Kalman Filter.

    Model: 
    - Based on the carvallo-whipple bicycle model.
    - Uses fixed, given bicycle parameters
    - Random walks for steer torque, roll torque and velocity. 
    - Estimate constant rotation between IMU and bike frame due to mechanical misalignment,
    - Estimate constant steer angle bias.

    The 13 states are:
    |               bicycle states                     | IMU<->bike rot  |  bias  |
    [x, y, psi, v, phi, delta, psidot, phidot, deltadot, epsx, epsy, epsz, b_steer]

    Measurements:
    - RTK-GNSS on the bicycle rack measuring x,y and velocity vector of the antenna. 
    - Steer encoder for steer angle and steer rate
    - Use gyrox, gyroz, ay and az from the onboard IMU. Omit uninformative gyroy and ax.
    - Omit IMU orientation estimates (psi, phi) due to unknown Notch filter and reference frame.
    - Rear wheelspeed sensor for longitudinal speed.
    
    The 11 measurements are:
    |    gnss    | steer encoder  |          imu        |ws|          
    [x, y, vx, vy, delta, deltadot, gyrox, gyroz, ay, az, v]

    Parameters
    ----------
    measurements : array-like
        Measurement array of shape (N, 11). NaN values indicate where a sensor does not return a value due to lower sampling rate. 
    uncertainties : array-like
        GNSS measurement uncertainties of shape (N, 6) with [varx, vary, varvx, varvy, covxy, covvxvy].
    R : array-like
        Constant measurement noise matrix (11, 11). Entries R[:4,:4] will be overwritten with the gnss uncertainties.
        Use make_R and get default_filter_setttings() to get the default R.
    Q : array-like
        Process noise matrix (12, 12). Use make_Q and get default_filter_setttings() to get the default Q.
    t_s : float, optional
        Sample time. The default is 0.01.
    smooth : bool, optional
        Optionally smooth the filter result with a Rauch-Tung-Striebel smoother (recommended). The default is True.
    plot : bool, optional
        Plot the filter results. The default is False.
    integration_method: str, optional
        The integration method for the integration of the system dynamics in 
        the prediction step. Can be 'backward euler' or 'midpoint'. The default
        is 'backward euler'.
    bicycle_parameter_dict : dict
        A dictionary of bicycle parameters as returned by the bicycleparameters
        toolbox. Use this to customize the bike. 
    estimate_gyro_biases : bool
        If true, adds gyro bias states to the filter. Use with caution! May destabilize the filter.
        Default is False.
    estimate_imu_orient_misalignment : bool
        If true, estimates the misalignment of the imu with respect to it's nominal mounting position. 
        Use with caution! May destabilize the filter.
    """

    if bicycle_geometry is None:
        bicycle_geometry = InstrumentedBikeGeometry()

    bicyclestate_labels = ['x', 'y', 'psi', 'v', 'phi', 'delta', 'dpsi', 'dphi', 'ddelta']
    imu_orient_labels = ['eps_x', 'eps_y', 'eps_z']
    bias_labels = ['b_delta', 'b_gyrox', 'b_gyroz']

    n_bikestates = len(bicyclestate_labels)
    n_states = len(bicyclestate_labels)+len(imu_orient_labels)+len(bias_labels)
    n_samples = measurements.shape[0]-1
    n_measurements = n_states

    def update_R(i):
        """Update the measurement variance matrix according to GNSS availablility at
        time step i.
        """
        R_i = R.copy()

        # check availability
        pos_gnss_available = np.all(np.isfinite(measurements[i,0:2]))
        vel_gnss_available = np.all(np.isfinite(measurements[i,2:4]))

        # if gnss position is available plug in RTKLib covariance estimates. Else, make unreliable
        if pos_gnss_available:
            R_i[:2,:2] += [[uncertainties_gnss[i,0], uncertainties_gnss[i,4]],
                           [uncertainties_gnss[i,4], uncertainties_gnss[i,1]]]
        else:
            R_i[:2,:2] = 1e10 * np.eye(2)

        # if gnss velocity is available plug in RTKLib variance estimates and make wheelspeed/imu orientation unreliable. 
        if vel_gnss_available:
            R_i[2:4,2:4] += [[uncertainties_gnss[i,2], uncertainties_gnss[i,5]],
                             [uncertainties_gnss[i,5], uncertainties_gnss[i,3]]]
        else: 
            R_i[2:4,2:4] = 1e10 * np.eye(2)      
        
        return R_i
    

    def make_initial_state(m, u):
        """ Make the initial state from first measurement """
        R0 = update_R(0)

        x0 = np.zeros(n_states)

        x0[0] = m[0]    #x 
        x0[1] = -m[1]   #y: flip due to E<->N conversion
        x0[2] = np.arctan2(-m[3], m[2]) #psi: flip vy due to E<->N conversion
        x0[3] = m[10]   #v
        x0[4] = 0       #phi: could be anything
        x0[5] = m[4]    #delta
        x0[6] = m[7]    #psidot: approx. gyro_z
        x0[7] = m[6]    #phidot: approx. gyro_x 
        x0[8] = m[5]    #deltadot


        P0 = np.zeros((x0.size, x0.size), dtype=float) 

        P0[:2,:2] = R0[:2,:2] + [[u[0], u[4]], [u[4], u[1]]] #x,y: gnss measurement uncertainty
        P0[2,2] = (m[2]**2 * P0[0,0] + m[3]**2 * P0[1,1] - 2 * m[2] * m[3] * P0[0,1]) /  (m[2]**2+m[3]**2 + 1e-12) #psi: error propagation through arctan
        P0[3,3] = R0[10,10]  #         v
        P0[4,4] = np.deg2rad(5)**2     #phi: could be anything
        P0[5,5] = R0[4,4]              #delta
        P0[6,6] = 3 * R0[7,7]          #psidot: approx. gyro_z + error from phi=0
        P0[7,7] = 3 * R0[6,6]          #phidot: approx. gyro_x + error from phi=0
        P0[8,8] = R0[5,5]              #deltadot

        P0[9:12, 9:12] = np.deg2rad(2)**2 * np.eye(3) # IMU<->bicycle orientation is small.
        P0[12:15,12:15] = (3*np.pi)**2 *np.eye(3)   # large, bias could be any valid angle  

        return x0, P0

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
    
    x0, P0 = make_initial_state(measurements[0,:], uncertainties_gnss[0,:])
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

        R_i = update_R(i+1)

        ukf.update(m, R=R_i, bicycle_geometry=bicycle_geometry, 
                   estimate_gyro_biases=estimate_gyro_biases,
                   estimate_imu_orient_misalignment=estimate_imu_orient_misalignment)

        state_measurements.append(measure(ukf.x, bicycle_geometry, 
                                          estimate_gyro_biases=estimate_gyro_biases,
                                          estimate_imu_orient_misalignment=estimate_imu_orient_misalignment))
        
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
    
    if plot_meas_error:
        fige, axese = plot_measurement_error(measurements, state_measurements)
    if plot:
        figxy, axxy = plot_filter_result_xy(measurements, states_filtered, covs_filtered, states_smoothed=states_smoothed, covs_smoothed=covs_smoothed)
        fig_bike, axes_bike, fig_supp, axes_supp = plot_filter_result_t(measurements, states_filtered, covs_filtered,states_smoothed=states_smoothed, covs_smoothed=covs_smoothed)
        
    out = [states_filtered, covs_filtered]
    if smooth:
        out += [states_smoothed, covs_smoothed]
    if plot:
        out += [fig_bike, axes_bike, fig_supp, axes_supp, figxy, axxy]
    if plot_meas_error:
        out += [fige, axese]

    return out
    

def plot_measurement_error(measurements, state_measurements):

    features = ['x', 'y', 'vx', 'vy', 'delta', 'deltadot', 'gyrox', 'gyroz', 'ay', 'az', 'v']

    t = np.arange(0, measurements.shape[0])

    style = dict(marker='.', markersize=2, linewidth=0.5)

    fig, axes = plt.subplots(measurements.shape[1], 1, sharex=True, layout='constrained')
    for i in range(measurements.shape[1]):
        axes[i].grid()
        fin = np.isfinite(measurements[:,i])
        axes[i].plot(t[fin], measurements[fin,i], **style, label='measured')
        axes[i].plot(state_measurements[:,i], **style, label='estimated')
        axes[i].set_ylabel(features[i])

    axes[0].legend()
    axes[0].set_title("gnss")
    axes[4].set_title("steer encoder")
    axes[6].set_title("imu")
    axes[10].set_title("wheelspeed sensor")
    fig.suptitle("Measurements")

    fig.set_size_inches(10.5, 9)

    return fig, axes


def plot_filter_result_xy(measurements, 
                         states_filtered, covs_filtered,
                         states_smoothed=None, covs_smoothed=None, 
                         color_gnss='#EC6842',
                         color_filtered='#FFB81C', color_smoothed='#009B77', ax=None):
    """Plot the filtered/smoothed states into an x/y plot.
    """

    smooth = not(states_smoothed is None) and not(covs_smoothed is None)

    plotstyle_filt = dict(linewidth=1)
    plotstyle_cov = dict(linewidth=1, linestyle="--")
    plotstyle_meas = dict(marker='.', markersize=2, linestyle='None')

    if ax is None:
        fig, ax = plt.subplots(1,1, layout='constrained')
        ax.set_aspect('equal')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title("filtered and smoothed bicycle position")
        ax.yaxis.set_inverted(True)
    else:
        fig = ax.get_figure()

    #x-y plot
    fin_gnss = np.logical_and(np.isfinite(measurements[:,0]), np.isfinite(measurements[:,1]))
    ax.plot(measurements[fin_gnss,0], -measurements[fin_gnss,1], color= color_gnss, label = 'gnss', **plotstyle_meas)

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
                         states_filtered, covs_filtered, 
                         states_smoothed=None, covs_smoothed=None, measurements_imuorient=None,
                         color_meas_can='#A50034',
                         color_meas_gnss='#EC6842',
                         color_filtered='#FFB81C', color_smoothed='#009B77', t_s=0.01):
    """Plot the filtered/smoothed states into an time plot.
    """

    smooth = not(states_smoothed is None) and not(covs_smoothed is None)

    plotstyle_filt = dict(linewidth=1)
    plotstyle_meas = dict(marker='.', markersize=2, linestyle='None')

    bicyclestate_labels = ['x', 'y', 'psi', 'v', 'phi', 'delta', 'dpsi', 'dphi', 'ddelta']
    imu_orient_labels = ['eps_x', 'eps_y', 'eps_z']
    bias_labels = ['b_delta', 'b_gyrox', 'b_gyroz']
    state_labels = bicyclestate_labels+imu_orient_labels+bias_labels

    t = np.arange(0, measurements.shape[0]) * t_s

    n_states = len(bicyclestate_labels)
    n_eps = len(imu_orient_labels)
    n_biases = len(bias_labels)

    if measurements_imuorient is None:
        measurements_imuorient = np.full((measurements.shape[0],2), np.nan)

    naive_states_gnss = np.c_[measurements[:,0],
                              -measurements[:,1],
                              np.arctan2(-measurements[:,3], measurements[:,2]),
                              np.hypot(-measurements[:,3], measurements[:,2])]
    
    naive_states_can = np.c_[measurements_imuorient[:,0],
                             measurements[:,10],
                             measurements_imuorient[:,1],
                             measurements[:,4],
                             measurements[:,7],
                             measurements[:,6],
                             measurements[:,5]]
    n_gnss = naive_states_gnss.shape[1]

    fig_bike, axes_bike = plt.subplots(n_states, 1, layout='constrained', sharex=True)
    fig_biases, axes_biases = plt.subplots(n_biases+n_eps, 1, layout='constrained', sharex=True)
    axes = np.r_[axes_bike, axes_biases]
    
    for i in range(n_states+n_eps+n_biases):
        minval = np.inf
        maxval = -np.inf

        if i < 4:
            fin_gnss = np.isfinite(naive_states_gnss[:,i])
            if np.any(fin_gnss):
                axes[i].plot(t[fin_gnss], naive_states_gnss[fin_gnss,i], color= color_meas_gnss, label = 'gnss', **plotstyle_meas)
                minval = min(minval, np.min(naive_states_gnss[fin_gnss,i]))
                maxval = max(maxval, np.max(naive_states_gnss[fin_gnss,i]))

        if i >= 2 and i < 9:
            fin_can = np.isfinite(naive_states_can[:,i-2])
            if np.any(fin_can):
                axes[i].plot(t[fin_can], naive_states_can[fin_can,i-2], color= color_meas_can, label = 'can', **plotstyle_meas)
                minval = min(minval, np.min(naive_states_can[fin_can,i-2]))
                maxval = max(maxval, np.max(naive_states_can[fin_can,i-2]))

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
        axes[0].set_title("filtered bicycle states")
    axes[n_states].set_title("IMU/bicycle orientation")
    axes[n_states+n_eps].set_title("biases")

    fig_bike.set_size_inches(10.5, 9)
    fig_biases.set_size_inches(10.5, 4)

    return fig_bike, axes_bike, fig_biases, axes_biases 
        
