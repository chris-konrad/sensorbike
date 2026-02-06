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
    
    def __init__(self, geometry_params=None):
        """
        Create an InstrumentedBikeGeometry object. 
        
        geometry_params : dict, optional
            A dictionary describing the position of the GNSS antenna and bicycle IMU:
                h_gnss : distance [m] between GNSS antenna and the rear-wheel contact point in B.z direction (vertical). Default is -1.08 m
                l_gnss : distance [m] between GNSS antenna and the rear-wheel contact point in B.x direction (horizontal). Default is -0.16 m
                h_imu  : distance [m] between the onboard IMU and the rear-wheel contact point in B.z direction (vertical). Default is -0.82 m
                l_imu : distance [m] between the onboard IMU and the rear-wheel contact point in B.x direction (horizontal). Default is 0 m
            The default corresponds to the setup used for the interaction experiment (GNSS with wooden antenna post). 
            Use sensorbike.geometry.get_geometry_params() for different configurations that have been used before.
        """

        if geometry_params is None:
            self.params = get_geometry_params()
        else:
            self.params = geometry_params

        self._init_can_transformations()
        self._init_gnss2rwcp_transformations()
        self._init_rwcp2gnss_transformations()

    
    def _init_symbols(self):
        
        # Create symbols and reference frames
        E, N, B = sm.symbols('E N B', cls=me.ReferenceFrame)
        Sgnss, Simu = sm.symbols('S_gnss S_imu', cls=me.ReferenceFrame)
        x, y,  v = me.dynamicsymbols('x y v')
        psi, phi, delta =  me.dynamicsymbols('psi phi delta')
        epsx, epsy, epsz = sm.symbols("epsx, epsy, epsz") 
        psidot = psi.diff()
        phidot = phi.diff()
        deltadot = delta.diff()
        
        h_gnss, l_gnss, h_imu, l_imu = sm.symbols('h_gnss, l_gnss, h_imu, l_imu')
        x_gnss, y_gnss, vx_gnss, vy_gnss = sm.symbols('x_gnss y_gnss vx_gnss vy_gnss')

        # parameter for the position of the GNSS antenna
        if not np.all([k in self.params.keys() for k in ['l_gnss', 'h_gnss']]):
            msg = ("The calibration must include measurements of the antenna"
                   "height over ground 'h_gnss' and horizontal distance"
                   "to the rear-wheel contanct point 'l_gnss'.")
            raise ValueError(msg)
        self.param_vals = {l_gnss: self.params['l_gnss'], h_gnss: self.params['h_gnss'],
                           l_imu: self.params['l_imu'], h_imu: self.params['h_imu']}
        
        # Set rotations of reference frames
        N.orient_body_fixed(E, (sm.pi, 0, 0), 'XYZ')
        B.orient_body_fixed(N, (psi, phi, 0), 'ZXY')
        Simu.orient_body_fixed(B, (epsx, epsy, epsz), 'XYZ')

        O = me.Point('O')
        Prwcp = me.Point('P_rwcp')
        Prwcp.set_pos(O, x * N.x + y * N.y)
        Pgnss = me.Point('P_gnss')
        Pgnss.set_pos(Prwcp, l_gnss * B.x + h_gnss * B.z)
        Pimu = me.Point('P_imu')
        Pimu.set_pos(Prwcp, l_imu * B.x + h_imu * B.z)

        frames = (E, N, B, Sgnss, Simu)
        states = (x, y, psi, v, phi, delta, psidot, phidot, deltadot)
        gnss_measurements = (x_gnss, y_gnss, vx_gnss, vy_gnss)
        points = (O, Pgnss, Prwcp, Pimu)
        eps = (epsx, epsy, epsz)

        return frames, states, gnss_measurements, points, eps


    def _init_gnss2rwcp_transformations(self):
        """ Transform gnss measurement to bicycle states.
        
        Creates a function to derive the position, speed and planar orientation 
        of the rear-wheel contact point given gnss measurements and the bicycles
        roll angle.

        The GNSS measures the position and velocity of the antenna in the E.x/E.y plane.
        """

        frames, states, gnss_measurements, points, _ = self._init_symbols()
        
        x, y, psi, v, phi, delta, psidot, phidot, deltadot = states
        E, N, B, _, _ = frames 
        x_gnss, y_gnss, vx_gnss, vy_gnss = gnss_measurements
        O, Pgnss, Prwcp, _ = points

        Pgnss.set_vel(E, vx_gnss * E.x + vy_gnss * E.y)
        vel_rwcp_E = Prwcp.v2pt_theory(Pgnss, E, B).subs(self.param_vals)

        speedrwcp_from_gnss = sm.sqrt(vel_rwcp_E.dot(N.x)**2 + vel_rwcp_E.dot(N.y)**2)
        psirwcp_from_gnss = sm.atan2(vel_rwcp_E.dot(N.y), vel_rwcp_E.dot(N.x))

        xrwcp_from_gnss = sm.solve(Pgnss.pos_from(O).dot(E.x) - x_gnss, x)[0]
        yrwcp_from_gnss = sm.solve(Pgnss.pos_from(O).dot(E.y) - y_gnss, y)[0]

        state_from_gnss = (xrwcp_from_gnss, yrwcp_from_gnss, psirwcp_from_gnss, speedrwcp_from_gnss)

        self._eval_gnss2rwcp = sm.lambdify((gnss_measurements, states), state_from_gnss)


    def _init_rwcp2gnss_transformations(self):
        """ Transform the bicycle states to the GNSS location.
        
        Creates a function to calculate what the GNSS would measure given a 
        bicycle state. Used as measurement model in the update step of the UKF.
        """

        frames, states, _, points, _ = self._init_symbols()
        
        E, _, B, _, _ = frames 
        _, _, _, v, _, _, _, _, _ = states
        O, Pgnss, Prwcp, _ = points

        Prwcp.set_vel(E, v * B.x)
        vel_gnss_E = Pgnss.v2pt_theory(Prwcp, E, B).subs(self.param_vals)

        vxgnss_from_states = vel_gnss_E.dot(E.x)
        vygnss_from_states = vel_gnss_E.dot(E.y)

        xgnss_from_states  = Pgnss.pos_from(O).dot(E.x).subs(self.param_vals)
        ygnss_from_states  = Pgnss.pos_from(O).dot(E.y).subs(self.param_vals)

        gnss_measurement = (xgnss_from_states, ygnss_from_states, vxgnss_from_states, vygnss_from_states)

        self._eval_rwcp2gnss = sm.lambdify([states], gnss_measurement)

        
    def _init_can_transformations(self):
        """
        Create functions transforming between the bicycle states in the E frame 
        and the sensor measurements on the can bus. 
        """

        frames, states, _, points, eps = self._init_symbols()
        _, _, Prwcp, Pimu = points
        E, N, B, _, Simu = frames 
        x, y, psi, v, phi, delta, psidot, phidot, deltadot = states
        (epsx, epsy, epsz) = eps

        vdot = v.diff()
        psidotdot = psidot.diff()
        phidotdot = phidot.diff()

        substitions = self.param_vals | {psidotdot: 0, phidotdot:0, vdot: 0}

        # imu (omit pitch)
        psi_imu, phi_imu, wx_imu, wz_imu  = me.dynamicsymbols('psi_imu phi_imu wx_imu wz_imu ')
        ay_imu, az_imu = me.dynamicsymbols('ay_imu, az_imu')

        # steer encoder
        delta_enc = me.dynamicsymbols('delta_enc')
        deltadot_enc = delta_enc.diff()

        # wheelspeed rear
        v_rws = me.dynamicsymbols('v_rws')

        can_measurements = [delta_enc, deltadot_enc, psi_imu, wx_imu, wz_imu, ay_imu, az_imu, v_rws]

        # states to accel
        Prwcp.set_vel(N, v * B.x)

        a_imu = Pimu.a2pt_theory(Prwcp, N, Simu) - 9.81 * N.z
        #a_imu = a_imu.subs(substitions)

        #ax_imu_meas = a_imu.dot(Simu.x)
        ay_imu_meas = a_imu.dot(Simu.y).subs(substitions)
        az_imu_meas = a_imu.dot(Simu.z).subs(substitions)

        # gyroz to states and vice-versa
        wzimu_from_states = B.ang_vel_in(N).dot(Simu.z)
        psidot_from_wzimu = sm.solve(wzimu_from_states - wz_imu, psidot)[0]

        # gyrox to states and vice-versa
        wximu_from_states = B.ang_vel_in(N).dot(Simu.x)
        #psidot_from_wximu = sm.solve(wximu_from_states - wx_imu, psidot)[0]

        states_can = (psi_imu, v_rws, phi_imu, delta_enc, psidot_from_wzimu, wx_imu, deltadot_enc)
        self._eval_can2states = sm.lambdify((can_measurements, states, eps), states_can)
        self._eval_states2can = sm.lambdify([states, eps], [delta, deltadot, wximu_from_states, wzimu_from_states, ay_imu_meas, az_imu_meas, v])


    def transform_gnss2statesE(self, gnss_measurements, states_E):
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


    def transform_states2gnss(self, states):
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
        return np.array(self._eval_rwcp2gnss(states))

    
    def transform_states2can(self, states, eps=None):
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
            [delta_enc, deltadot_enc, phi_imu, wx_imu, psi_imu, wz_imu, v_rws, ax_imu].
        """
        if eps is None:
            if states.ndim ==1:
                eps = np.zeros(3)
            else:
                eps = np.zeros((states.shape[0], 3))

        return self._eval_states2can(states, eps)
    

def get_geometry_params(setup='interaction2024'):
    """Get the geometry parameters for different IMU/GNSS mounting positions.
    Choose between the setup used for different experiments.

    Setups:
    -------
    interaction2024 : GNSS mounted on a small wooden pole on the bicycle rack. 
    Used in https://resolver.tudelft.nl/uuid:092f3b70-2d97-436e-b193-139a593e09c7

    zigzag2024 : GNSS mounted on a small box on the rack of the bicycle.
    Used in https://engrxiv.org/preprint/view/6107/version/7996
    
    Parameters
    ----------
    setup : str
        One of 'interaction2024' or 'zigzag2024'. Default is 'interaction2024'

    Returns
    -------
    geometry_params : dict
        The geometry parameter dict containing the GNSS (h_gnss, l_gnss) and IMU locataion
        (h_imu, l_imu).
    """
    

    gnss_mounting_positions = {
        'interaction2024': dict(h_gnss=-1.08, l_gnss=-0.16),
        'zigzag2024': dict(h_gnss=-0.94, l_gnss=-0.17)}

    imu_mounting_position = dict(h_imu=-0.82, l_imu=-0)

    return gnss_mounting_positions[setup] | imu_mounting_position
