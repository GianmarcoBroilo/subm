## Context
"""
INPUT
Data type: stellar occultation of Io, VLBI Jupiter
A priori cov: uncertainties in the state of Io and Jupiter in RSW
parameters: initial state of Io, initial state of Jupiter
OUTPUT
cov: uncertainty and correlation of estimated parameters for both Io and Jupiter
"""
#%%
# Load standard modules
import math

import numpy as np
from numpy import linalg as lalg
from matplotlib import pyplot as plt

# Load tudatpy modules
from tudatpy.util import result2array
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import estimation, estimation_setup
from tudatpy.kernel.numerical_simulation.estimation_setup import observation
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.numerical_simulation import propagation_setup, propagation
import pandas as pd
from tudatpy.kernel.astro import frame_conversion
import math


# Load spice kernels
spice.load_standard_kernels()

# Set simulation start and end epochs
simulation_start_epoch = 0 #2459215.50
simulation_end_epoch = simulation_start_epoch +  1*constants.JULIAN_YEAR


## Environment setup


# Create default body settings
bodies_to_create = ["Earth", "Io", "Jupiter","Sun"]

time_step = 1500
initial_time = simulation_start_epoch - 5*time_step
final_time = simulation_end_epoch + 5*time_step

# Create default body settings for bodies_to_create, with "Jupiter"/"J2000" as the global frame origin and orientation
global_frame_origin = "SSB"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

original_io_ephemeris_settings = body_settings.get("Io").ephemeris_settings
body_settings.get("Io").ephemeris_settings = environment_setup.ephemeris.tabulated_from_existing(
    original_io_ephemeris_settings,initial_time, final_time, time_step
)
# Create system of bodies
bodies = environment_setup.create_system_of_bodies(body_settings)


"""
Propagation setup
"""

# Define bodies that are propagated
bodies_to_propagate = ["Io","Jupiter"]

# Define central bodies of propagation
central_bodies = []
for body_name in bodies_to_create:
    if body_name == "Io":
        central_bodies.append("Jupiter")
    elif body_name == "Jupiter":
        central_bodies.append("Sun")

### Create the acceleration model

acceleration_settings_io = dict(
    Jupiter = [propagation_setup.acceleration.point_mass_gravity()],
)
acceleration_settings_jup = dict(
    Sun=[propagation_setup.acceleration.point_mass_gravity()],
)
acceleration_settings = {"Io": acceleration_settings_io, "Jupiter": acceleration_settings_jup}

acceleration_models = propagation_setup.create_acceleration_models(
    bodies,acceleration_settings,bodies_to_propagate,central_bodies
)
### Define the initial state
"""
The initial state of Io and Jupiter that will be propagated is now defined. 
"""

# Set the initial state of Io
initial_state = propagation.get_initial_state_of_bodies(
    bodies_to_propagate=bodies_to_propagate,
    central_bodies=central_bodies,
    body_system=bodies,
    initial_time=simulation_start_epoch,
)

# Create termination settings
termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)

# Create propagation settings
propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_state,
    termination_condition
)
"""
Propagate the dynamics of Jupiter and Io and extract state transition and sensitivity matrices
"""
#Setup paramaters settings to propagate the state transition matrix
parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
# Create the parameters that will be estimated
parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies)

# Create numerical integrator settings.
fixed_step_size = 1500.0
integrator_settings = propagation_setup.integrator.runge_kutta_4(
    simulation_start_epoch, fixed_step_size
)
# Create the variational equation solver and propagate the dynamics
variational_equations_solver = numerical_simulation.SingleArcVariationalSimulator(
    bodies, integrator_settings, propagator_settings, parameters_to_estimate, integrate_on_creation=True)

# Extract the resulting state history, state transition matrix history, and sensitivity matrix history
states = variational_equations_solver.state_history
state_transition_matrix = variational_equations_solver.state_transition_matrix_history
sensitivity_matrix = variational_equations_solver.sensitivity_matrix_history

""""
Define the a priori covariance of Io 
"""
#%%
rotation_rsw_to_inertial_dict_io = dict()
for epoch in list(variational_equations_solver.state_history):
    rotation_rsw_to_inertial_dict_io[epoch] = frame_conversion.rsw_to_inertial_rotation_matrix(states[epoch][:6]).reshape(3,3)
uncertainties_rsw_io = np.zeros((3,3))
np.fill_diagonal(uncertainties_rsw_io,[15e3,15e3,15e3])
uncertainties_rsw_velocity_io = np.zeros((3,3))
np.fill_diagonal(uncertainties_rsw_velocity_io,[0.15,1.15,0.75])
covariance_position_initial_io = lalg.multi_dot([rotation_rsw_to_inertial_dict_io[simulation_start_epoch],uncertainties_rsw_io,rotation_rsw_to_inertial_dict_io[simulation_start_epoch].T])
covariance_velocity_initial_io = lalg.multi_dot([rotation_rsw_to_inertial_dict_io[simulation_start_epoch],uncertainties_rsw_velocity_io,rotation_rsw_to_inertial_dict_io[simulation_start_epoch].T])

""""
Define the a priori covariance of Jupiter 
"""
rotation_rsw_to_inertial_dict_jup = dict()
for epoch in list(variational_equations_solver.state_history):
    rotation_rsw_to_inertial_dict_jup[epoch] = frame_conversion.rsw_to_inertial_rotation_matrix(states[epoch][6:12]).reshape(3,3)
uncertainties_rsw_jup = np.zeros((3,3))
np.fill_diagonal(uncertainties_rsw_jup,[1e3,1e3,1e3])
uncertainties_rsw_velocity_jup = np.zeros((3,3))
np.fill_diagonal(uncertainties_rsw_velocity_io,[0.1,0.1,0.1])
covariance_position_initial_jup = lalg.multi_dot([rotation_rsw_to_inertial_dict_jup[simulation_start_epoch],uncertainties_rsw_jup,rotation_rsw_to_inertial_dict_jup[simulation_start_epoch].T])
covariance_velocity_initial_jup = lalg.multi_dot([rotation_rsw_to_inertial_dict_jup[simulation_start_epoch],uncertainties_rsw_velocity_jup,rotation_rsw_to_inertial_dict_jup[simulation_start_epoch].T])

""""
Define global a priori covariance 
"""
covariance_a_priori = np.block([
    [covariance_position_initial_io, np.zeros((3,3)), np.zeros((3,3)),np.zeros((3,3))],
    [np.zeros((3,3)),covariance_velocity_initial_io,np.zeros((3,3)), np.zeros((3,3))],
    [np.zeros((3,3)),np.zeros((3,3)), covariance_position_initial_jup, np.zeros((3,3))],
    [np.zeros((3,3)),np.zeros((3,3)), np.zeros((3,3)), covariance_velocity_initial_jup],
])

""""
Observation Setup
"""
#%%
# Define the position of the ground station on Earth
station_altitude = 0.0
delft_latitude = np.deg2rad(52.00667)
delft_longitude = np.deg2rad(4.35556)

# Add the ground station to the environment
environment_setup.add_ground_station(
    bodies.get_body("Earth"),
    "TrackingStation",
    [station_altitude, delft_latitude, delft_longitude],
    element_conversion.geodetic_position_type)

# Define the uplink link ends types
link_ends = dict()
link_ends[observation.receiver] = ("Earth", "TrackingStation")
link_ends[observation.transmitter] = ("Io", "")
link_ends[observation.transmitter] = ("Jupiter", "")

# Define the observations
