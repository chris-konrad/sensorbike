import re
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me

from trajdatamanager.datamanager import Track

class InstrumentedBikeGeometry:
    """ A class that describes the geometry of the instrumented bicycle and 
    provides functions to transform sensor measurements to bicycle coordinates.

    Reference Frames
    ----------------
    B : Attached to the Frame of the bicycle, with B.x pointing forward, 
        B.y pointing to the right and B.z pointing downwards such that 
        B.x is parallel to the ground and B.z is parallel to gravity when
        the bicycle is upright. 
    N : The local reference frame as commonly defined in bicycle dynamics.
        N.x and N.y are fixed to the road surface and N.z points downwards.
    E : The local reference frame as commonly defined in traffic simulation.
        E.x and E.y are fixed to the road surface and E.z points upwards. 
        E.x and N.x are parallel. 
    Sgnss : The reference frame of the GNSS velocity. Sgnss.x equals the
        direction of the GNSS velocity vector derived from x-y coordinates.
        Sgnss.x and Sgnss.y are parallel to the ground and Sgnss.z is parallel
        to E.z. This ignores velocity in E.z direction, coming from the height
        of the GNSS changing when the bicycle tilts.


    Reference Locations
    -------------------

    P_gnss : The position of the antenna of the GNSS device rigidly attached to
        the rack of the bicycle.
    P_rwcp : The rear-wheel contact point moving on the ground.
    P_imu : The position of the bicycle IMU in the control box of the balance
        assist bicycles.
    
    This class was derived from rcid.utils by Christoph M. Konrad.
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

        self._init_can_transformations()
        self._init_gnss2rwcp_transformations()
        self._init_rwcp2gnss_transformations()

    
    def _init_symbols(self):
        
        # Create symbols and reference frames
        E, N, B = sm.symbols('E N B', cls=me.ReferenceFrame)
        Sgnss = sm.symbols('S_gnss', cls=me.ReferenceFrame)
        x, y_E,  v = me.dynamicsymbols('x y_E v')
        psi_E, phi, delta =  me.dynamicsymbols('psi_E phi delta')
        psidot_E = psi_E.diff()
        phidot = phi.diff()
        deltadot = delta.diff()
        a = v.diff()
        
        h_gnss, l_gnss = sm.symbols('h_gnss, l_gnss')
        x_gnss, y_gnss, v_gnss, psi_gnss = sm.symbols('x_gnss y_gnss v_gnss psi_gnss')

        # parameter for the position of the GNSS antenna
        if not np.all([k in self.params.keys() for k in ['l_gnss', 'h_gnss']]):
            msg = ("The calibration must include measurements of the antenna"
                   "height over ground 'h_gnss' and horizontal distance"
                   "to the rear-wheel contanct point 'l_gnss'.")
            raise ValueError(msg)
        self.param_vals = {l_gnss: self.params['l_gnss'], h_gnss: self.params['h_gnss']}
        
        # Set rotations of reference frames
        N.orient_body_fixed(E, (sm.pi, 0, 0), 'XYZ')
        B.orient_body_fixed(N, (-psi_E, phi, 0), 'ZXY')

        Prwcp = me.Point('P_rwcp')
        Pgnss = me.Point('P_gnss')
        Pgnss.set_pos(Prwcp, - l_gnss * B.x - h_gnss * B.z)
        #Prwcp.set_vel(E, v)
        #Pgnss = Prwcp.locatenew('P_gnss', - l_gnss * B.x - h_gnss * B.z)
        r_Prwcp_Pgnss = Prwcp.pos_from(Pgnss).express(E)
        r_Prwcp_Pgnss = r_Prwcp_Pgnss.subs(self.param_vals)

        frames = (E, N, B, Sgnss)
        states_E = (x, y_E, psi_E, v, phi, delta, psidot_E, phidot, deltadot, a)
        gnss_measurements = (x_gnss, y_gnss, psi_gnss, v_gnss)
        points = (Pgnss, Prwcp)

        return frames, states_E, gnss_measurements, points, r_Prwcp_Pgnss


    def _init_gnss2rwcp_transformations(self):
        """ Transform gnss measurement to bicycle states.
        
        Creates a function to derive the position, speed and planar orientation 
        of the rear-wheel contact point given gnss measurements and the bicycles
        roll angle.

        The GNSS measures the position and velocity of the antenna in the E.x/E.y plane.
        """

        frames, states_E, gnss_measurements, points, r_Prwcp_Pgnss = self._init_symbols()
        
        E, N, B, Sgnss = frames 
        x_gnss, y_gnss, psi_gnss, v_gnss = gnss_measurements
        Pgnss, Prwcp = points

        Sgnss.orient_body_fixed(E, (0, 0, psi_gnss), 'XYZ')
        Pgnss.set_vel(E, Sgnss.x * v_gnss)

        vel_rwcp_E = Prwcp.v2pt_theory(Pgnss, E, B).subs(self.param_vals)
        speed_rwcp_E = sm.sqrt(vel_rwcp_E.dot(E.x)**2 + vel_rwcp_E.dot(E.y)**2)
        psi_rwcp_E = sm.atan2(vel_rwcp_E.dot(E.y), vel_rwcp_E.dot(E.x))
        x_rwcp_E = x_gnss + r_Prwcp_Pgnss.dot(E.x)
        y_rwcp_E = y_gnss + r_Prwcp_Pgnss.dot(E.y)

        state_gnss_E = (x_rwcp_E, y_rwcp_E, psi_rwcp_E, speed_rwcp_E)

        self._eval_gnss2rwcp_E = sm.lambdify((gnss_measurements, states_E), state_gnss_E)


    def _init_rwcp2gnss_transformations(self):
        """ Transform the bicycle states to the GNSS location.
        
        Creates a function to calculate what the GNSS would measure given a 
        bicycle state. Used as measurement model in the update step of the UKF.
        """

        frames, states_E, _, points, r_Prwcp_Pgnss = self._init_symbols()
        
        E, N, B, Sgnss = frames 
        x, y_E, psi_E, v, phi, delta, psidot_E, phidot, deltadot, a = states_E
        Pgnss, Prwcp = points

        Prwcp.set_vel(E, v * B.x)

        vel_gnss_E = Pgnss.v2pt_theory(Prwcp, E, B).subs(self.param_vals)
        speed_gnss_E = sm.sqrt(vel_gnss_E.dot(E.x)**2 + vel_gnss_E.dot(E.y)**2)
        psi_gnss_E = sm.atan2(vel_gnss_E.dot(E.y), vel_gnss_E.dot(E.x))
        x_gnss_E = x - r_Prwcp_Pgnss.dot(E.x)
        y_gnss_E = y_E - r_Prwcp_Pgnss.dot(E.y)

        gnss_measurement = (x_gnss_E, y_gnss_E, psi_gnss_E, speed_gnss_E)

        self._eval_rwcp2gnss_E = sm.lambdify(states_E, gnss_measurement)

        
    def _init_can_transformations(self):
        """
        Create functions transforming between the bicycle states in the E frame 
        and the sensor measurements on the can bus. 
        """

        frames, states_E, _,_,_ = self._init_symbols()
        
        E, _, B, _ = frames 
        x, y_E, psi_E, v, phi, delta, psidot_E, phidot, deltadot, a = states_E

        # imu (omit pitch)
        psi_imu, phi_imu, wx_imugyro, wz_imugyro  = me.dynamicsymbols('psi_imu phi_imu  wx_imugyro wz_imugyro ')
        ax_imuaccel = me.dynamicsymbols('ax_imuaccel')

        # steer encoder
        delta_enc = me.dynamicsymbols('delta_enc')
        deltadot_enc = delta_enc.diff()

        # wheelspeed rear
        v_rws = me.dynamicsymbols('v_rws')

        can_measurements = [delta_enc, deltadot_enc, phi_imu, wx_imugyro, psi_imu, wz_imugyro, v_rws, ax_imuaccel]

        # imu2E and vice-versa
        B_w_E_Bz = B.ang_vel_in(E).dot(-B.z)                 #what the gyro measures in E
        B_w_E_Ez = sm.trigsimp(B.ang_vel_in(E).dot(E.z))     #bicycle yaw rate in E=

        B_w_E_Ez_solution = sm.solve(B_w_E_Bz - wz_imugyro, B_w_E_Ez)[0]
        B_w_E_Bz_solution = sm.solve(B_w_E_Ez_solution - psidot_E, wz_imugyro)[0]

        x, y_E, psi_E, v, phi, delta, psidot_E, phidot, deltadot, a
        can_E = ( - psi_imu, v_rws, phi_imu, - delta_enc, B_w_E_Ez_solution, wx_imugyro, - deltadot_enc, ax_imuaccel)
        self._eval_can2E = sm.lambdify((can_measurements, states_E), can_E)
        self._eval_E2can = sm.lambdify(states_E, [delta, deltadot, phi, phidot, psi_E, B_w_E_Bz_solution, v, a])


    def transform_gnss2stateE(self, gnss_measurements, states_E):
        """ Transform a GNSS measurement to a bicycle state in the E frame.

        The transformation depends on the lateral bicycle states. Thus, 
        states_E must be supplied. If no estimate is available, a
        can measurement can be used: states_E = self.transform_can2E(can_measurement)

        Parameters
        ----------
        gnss_measurement : array-like
            A gnss measurement of the position and velocity of
            the antenna in the E-frame, given as [x_gnss, y_gnss, psi_gnss, v_gnss].
        states_E : array-like
            A list of bicycle states in the E frame at the same timestep. Assume to be 
            [x, y, psi, v, phi, delta, psidot, phidot, deltadot, a] (format used by sensorbike.ukf)

        Returns
        -------
        states_GNSS_E:
            The list of bicycle states in the E frame derived from the gnss measurements. States 
            that do not have a corresponding measurement from the GNSS are replaced with None.
            [x, y, psi, v, None, None, None, None, None, None]
        """

        if gnss_measurements.ndim == 1:
            gnss_measurements = gnss_measurements[np.newaxis, :]
        if states_E.ndim == 1:
            states_E = states_E[np.newaxis, :]

        states_gnss_E = np.array(self._eval_gnss2rwcp_E(gnss_measurements.T, states_E.T)).T

        states_gnss_E = np.c_[states_gnss_E, np.full((states_gnss_E.shape[0], 6), np.nan)]

        if states_gnss_E.shape[0] == 1:
            states_gnss_E = states_gnss_E.flatten()
        return states_gnss_E


    def transform_gnss2stateN(self, gnss_measurement, states_N):
        """ Transform a GNSS measurement to a bicycle state in the N frame.

        The transformation depends on the lateral bicycle states. Thus, 
        states_N must be supplied. If no estimate is available, a
        can measurement can be used: states_N = self.transform_can2N(can_measurement)

        Parameters
        ----------
        gnss_measurement : array-like
            A gnss measurement of the position and velocity of
            the antenna in the E frame, given as [x_gnss, y_gnss, psi_gnss, v_gnss].
        states_N : array-like
            A list of bicycle states in the N frame at the same timestep. Assume to be 
            [x, y, psi, v, phi, delta, psidot, phidot, deltadot, a] (format used by sensorbike.ukf)

        Returns
        -------
        states_GNSS_N:
            The list of bicycle states in the N frame derived from the gnss measurements. States 
            that do not have a corresponding measurement from the GNSS are replaced with None.
            [x, y, psi, v, None, None, None, None, None, None]
        """

        states_E = self.transform_statesN2statesE(states_N)

        states_gnss_E = self.transform_gnss2stateE(gnss_measurement, states_E)

        states_gnss_N = self.transform_statesE2statesN(states_gnss_E)

        return states_gnss_N
    

    def transform_stateE2gnss(self, states_E):
        """ Transform a bicycle state in the E-frame to a GNSS measurement.
        Calculates what the GNSS is expected to measure given the current bicycle 
        state. 

        Parameters
        ----------
        states_E : array-like
            A list of bicycle states in the E frame at the same timestep. Assume to be 
            [x, y, psi, v, phi, delta, psidot, phidot, deltadot, a] (format used by sensorbike.ukf)

        Returns
        -------
        gnss_measurement : array-like
            A gnss measurement of the position and velocity of the antenna in the E frame
            corresponding to states_E, given as [x_gnss, y_gnss, psi_gnss, v_gnss].
        """
        return self._eval_rwcp2gnss_E(states_E)


    def transform_statesN2gnss(self, states_N):
        """ Transform a bicycle state in the N-frame to a GNSS measurement.
        Calculates what the GNSS is expected to measure given the current bicycle 
        state. 

        Parameters
        ----------
        states_N : array-like
            A list of bicycle states in the N frame at the same timestep. Assume to be 
            [x, y, psi, v, phi, delta, psidot, phidot, deltadot, a] (format used by sensorbike.ukf)

        Returns
        -------
        gnss_measurement : array-like
            A gnss measurement of the position and velocity of the antenna in the E frame
            corresponding to states_E, given as [x_gnss, y_gnss, psi_gnss, v_gnss].
        """
        states_E = self.transform_statesN2statesE(states_N)
        return self._eval_rwcp2gnss_E(states_E)


    def transform_statesE2statesN(self, states_E):
        """Transform a state vector from the E frame to the N frame.

        Mirrors y, psi, psidot, delta and deltadot.

        Parameters
        ----------
        states_E : array-like
            A list of bicycle states in the E frame at the same timestep. Assume to be 
            [x, y, psi, v, phi, delta, psidot, phidot, deltadot, a] (format used by sensorbike.ukf)

        Returns
        -------
        states_N : array-like
            A list of bicycle states in the N frame at the same timestep. Assume to be 
            [x, y, psi, v, phi, delta, psidot, phidot, deltadot, a] (format used by sensorbike.ukf)
        """

        states_N = states_E.copy()

        # flip y, psi, psidot, delta and deltadot
        if states_N.ndim > 1:
            states_N[:,1] *= -1
            states_N[:,2] *= -1
            states_N[:,5] *= -1
            states_N[:,6] *= -1
            states_N[:,8] *= -1
        else:
            states_N[1] *= -1
            states_N[2] *= -1
            states_N[5] *= -1
            states_N[6] *= -1
            states_N[8] *= -1

        return states_N
    

    def transform_statesN2statesE(self, states_N):
        """Transform a state vector from the N frame to the E frame.

        Mirrors y, psi, psidot, delta and deltadot.

        Parameters
        ----------
        states_N : array-like
            A list of bicycle states in the N frame at the same timestep. Assume to be 
            [x, y, psi, v, phi, delta, psidot, phidot, deltadot, a] (format used by sensorbike.ukf)

        Returns
        -------
        states_E : array-like
            A list of bicycle states in the E frame at the same timestep. Assume to be 
            [x, y, psi, v, phi, delta, psidot, phidot, deltadot, a] (format used by sensorbike.ukf)
        """
        return self.transform_statesE2statesN(states_N)

    
    def transform_can2stateE(self, can_measurements, states_E):
        """Transform the sensor measurements from the CAN bus 
        to bicycle states in the E-frame.

        Predominantly transforms gyro_z -> psidot. At roll=/=0, 
        the gyro does not directly measure yawrate. The transformation
        depends on the current roll angle. Thus, states_E must be given.

        If no state estimate is available, the roll angle from the can
        measurements (i.e., can_measurements[2] may be assigned to states_E[4]).
        All other elements of states_E are disregarded.

        Parameters
        ----------
        can_measurements : array-like
            A list of can measurements at a certain timestep. Assumed to be
            [delta_enc, deltadot_enc, phi_imu, wx_imugyro, psi_imu, wz_imugyro, v_rws, ax_imuaccel].
        states_E : array-like
            A list of bicycle states in the E frame at the same timestep. Assume to be 
            [x, y, psi, v, phi, delta, psidot, phidot, deltadot, a] (format used by sensorbike.ukf)

        Returns
        -------
        states_can_E : array-like
            The list of bicycle states in the E frame derived from the can measurements. States 
            that do not have a corresponding measurement on the CAN bus are replaced with NaN.
            [NaN, NaN, psi, v, phi, delta, psidot, phidot, deltadot, a]
        """
        if can_measurements.ndim == 1:
            can_measurements = can_measurements[np.newaxis, :]
        if states_E.ndim == 1:
            states_E = states_E[np.newaxis, :]

        states_can_E = np.array(self._eval_can2E(can_measurements.T, states_E.T)).T

        states_can_E = np.c_[np.full((states_can_E.shape[0], 2), np.nan), states_can_E]

        if states_can_E.shape[0] == 1:
            states_can_E = states_can_E.flatten()

        return states_can_E
    

    def transform_can2stateN(self, can_measurements, states_N):
        """Transform the sensor measurements from the CAN bus 
        to bicycle states in the N-frame.

        Predominantly transforms gyro_z -> psidot. At roll=/=0, 
        the gyro does not directly measure yawrate. The transformation
        depends on the current roll angle. Thus, states_E must be given.

        If no state estimate is available, the negative roll angle from the can
        measurements (i.e., - can_measurements[2] may be assigned to states_N[4]).
        All other elements of states_N are disregarded.

        Parameters
        ----------
        can_measurements : array-like
            A list of can measurements at a certain timestep. Assumed to be
            [delta_enc, deltadot_enc, phi_imu, wx_imugyro, psi_imu, wz_imugyro, v_rws, ax_imuaccel].
        states_N : array-like
            A list of bicycle states in the N frame at the same timestep. Assume to be 
            [x, y, psi, v, phi, delta, psidot, phidot, deltadot, a] (format used by sensorbike.ukf)

        Returns
        -------
        states_can_N : array-like
            The list of bicycle states in the N frame derived from the can measurements. States 
            that do not have a corresponding measurement on the CAN bus are replaced with None.
            [None, None, psi, v, phi, delta, psidot, phidot, deltadot, a]
        """
        
        states_E = self.transform_statesN2statesE(states_N)

        states_can_E = self.transform_can2stateE(can_measurements, states_E)

        states_can_N = self.transform_statesN2statesE(states_can_E)

        return states_can_N
    
    
    def transform_stateE2can(self, states_E):
        """Transform bicycle states in the E-frame to sensor measurements
        on the can bus. This returns what the CAN sensors would measure 
        if the state was states_E. 

        Predominantly transforms psidot -> gyro_z. At roll=/=0, 
        the gyro does not directly measure yawrate. 

        Parameters
        ----------
        states_E : array-like
            A list of bicycle states in the E frame. Assumed to be 
            [x, y, psi, v, phi, delta, psidot, phidot, deltadot, a] 
            (format used by sensorbike.ukf)

        Returns
        -------
        can_measurements : array-like
            A list of the expected can measurements corresponding to states_E in format
            [delta_enc, deltadot_enc, phi_imu, wx_imugyro, psi_imu, wz_imugyro, v_rws, ax_imuaccel].
        """

        return self._eval_E2can(states_E)
    
    
    def transform_stateN2can(self, states_N):
        """Transform bicycle states in the E-frame to sensor measurements
        on the can bus. This returns what the CAN sensors would measure 
        if the state was states_E. Used as measurement model in UKF.update().

        Predominantly transforms psidot -> gyro_z. At roll=/=0, 
        the gyro does not directly measure yawrate. 

        Parameters
        ----------
        states_N : array-like
            A list of bicycle states in the N frame. Assumed to be 
            [x, y, psi, v, phi, delta, psidot, phidot, deltadot, a] 
            (format used by sensorbike.ukf)

        Returns
        -------
        can_measurements : array-like
            A list of the expected can measurements corresponding to states_N in format
            [delta_enc, deltadot_enc, phi_imu, wx_imugyro, psi_imu, wz_imugyro, v_rws, ax_imuaccel].
        """
        
        states_E = self.transform_statesN2statesE(states_N)

        return self._eval_E2can(states_E)
     
            
    def transform_trk_N2E(self, data_dict):
        """
        Transform a data dictionary or Track object from the N to the E frame in place.

        Flips the trajectories with the keys: 'y', 'psi', 'dpsi', 'delta', 'ddelta'

        Parameters
        ----------
        data_dict : dict or trajdatamanager.Track
            The data object to transform.

        Returns
        -------
        data_dict : dict or trajdatamanager.Track
            The same data object but transformed.
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
    
    def transform_trk_E2N(self, data_dict):
        """
        Transform a data dictionary or Track object from the E to the N frame in place.

        Flips the trajectories with the keys: 'y', 'psi', 'dpsi', 'delta', 'ddelta'

        Parameters
        ----------
        data_dict : dict or trajdatamanager.Track
            The data object to transform.

        Returns
        -------
        data_dict : dict or trajdatamanager.Track
            The same data object but transformed.
        """
        return self.transform_trk_N2E(data_dict)