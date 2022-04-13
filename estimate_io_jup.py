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
simulation_start_epoch = 2462502.50
simulation_end_epoch = simulation_start_epoch +  1*constants.JULIAN_YEAR


## Environment setup


# Create default body settings
bodies_to_create = ["Earth", "Io", "Jupiter","Sun","Saturn"]

time_step = 3500
initial_time = simulation_start_epoch - 5*time_step
final_time = simulation_end_epoch + 5*time_step

# Create default body settings for bodies_to_create, with "Jupiter"/"J2000" as the global frame origin and orientation
global_frame_origin = "SSB"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

# Create tabulated settings fo Io and Jupiter
original_io_ephemeris_settings = body_settings.get("Io").ephemeris_settings
body_settings.get("Io").ephemeris_settings = environment_setup.ephemeris.tabulated_from_existing(
    original_io_ephemeris_settings,initial_time, final_time, time_step
)
original_jupiter_ephemeris_settings = body_settings.get("Jupiter").ephemeris_settings
body_settings.get("Jupiter").ephemeris_settings = environment_setup.ephemeris.tabulated_from_existing(
    original_jupiter_ephemeris_settings,initial_time, final_time, time_step
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
    Sun = [propagation_setup.acceleration.point_mass_gravity()]
)
acceleration_settings_jup = dict(
    Sun=[propagation_setup.acceleration.point_mass_gravity()],
    Saturn = [propagation_setup.acceleration.point_mass_gravity()]
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
integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
    simulation_start_epoch, 3500.0, propagation_setup.integrator.rkf_78, 3500.0, 3500.0, 1.0, 1.0
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
#15km RSW position 0.15,1.15,0.75m/s RSW velocity
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
# 1km RSW position 0.1m/s RSW velocity
rotation_rsw_to_inertial_dict_jup = dict()
for epoch in list(variational_equations_solver.state_history):
    rotation_rsw_to_inertial_dict_jup[epoch] = frame_conversion.rsw_to_inertial_rotation_matrix(states[epoch][6:12]).reshape(3,3)
uncertainties_rsw_jup = np.zeros((3,3))
np.fill_diagonal(uncertainties_rsw_jup,[1e3,1e3,1e3])
uncertainties_rsw_velocity_jup = np.zeros((3,3))
np.fill_diagonal(uncertainties_rsw_velocity_jup,[0.1,0.1,0.1])
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
covariance_a_priori_inverse = np.linalg.inv(covariance_a_priori)
""""
Observation Setup
"""
#%%
# Define the position of the ground station on Earth,add ESTRACK Malargue station
station_altitude = 1550.0
estrack_latitude = np.deg2rad(35.776)
estrack_longitude = np.deg2rad(69.398)

# Add the ground station to the environment
environment_setup.add_ground_station(
    bodies.get_body("Earth"),
    "TrackingStation",
    [station_altitude, estrack_latitude, estrack_longitude],
    element_conversion.geodetic_position_type)

# Define the uplink link ends types
link_ends_stellar = dict()
link_ends_stellar[observation.receiver] = ("Earth", "TrackingStation")
link_ends_stellar[observation.transmitter] = ("Io", "")
link_ends_vlbi = dict()
link_ends_vlbi[observation.receiver] = ("Earth", "TrackingStation")
link_ends_vlbi [observation.transmitter] = ("Jupiter", "")

# Create Bias settings 0.5 nrad in both RA and Dec
bias_io = observation.absolute_bias(np.array([0.5e-6,0.5e-6]))
bias_jup = observation.absolute_bias(np.array([0.5e-6,0.5e-6]))

# Create observation settings for each link/observable
observation_settings_list_io = observation.angular_position(link_ends_stellar,bias_settings = bias_io)
observation_settings_list_jup = observation.angular_position(link_ends_vlbi,bias_settings = bias_jup)

# Define the observations for Io
step_io = 80*constants.JULIAN_DAY
observation_times_io =  np.arange(simulation_start_epoch,simulation_end_epoch,step_io)
observation_simulation_settings_io = observation.tabulated_simulation_settings(
    observation.angular_position_type,
    link_ends_stellar,
    observation_times_io
)
# Define the observations for Jupiter, INPUT: flyby period, observation period, integration time
# output is observation_flat containing all the observation times in a single array
step_jup = 53.4*constants.JULIAN_DAY
observation_times_jup = np.arange(simulation_start_epoch,simulation_end_epoch,step_jup)

observation_simulation_settings_jup = observation.tabulated_simulation_settings(
    observation.angular_position_type,
    link_ends_vlbi,
    observation_times_jup
)

# Add noise levels of roughly 05 nrad to Io
noise_level_io = 0.5e-9
observation.add_gaussian_noise_to_settings(
    [observation_simulation_settings_io],
    noise_level_io,
    observation.angular_position_type
)

# Add noise levels of roughly 05 nrad to Jupiter
noise_level_jup = 0.5e-9
observation.add_gaussian_noise_to_settings(
    [observation_simulation_settings_jup],
    noise_level_jup,
    observation.angular_position_type
)

# Add viability settings for VLBI to account for low-elevation calibration data for Earth's troposphere
elevation_setting = observation.elevation_angle_viability(["Earth", "TrackingStation"], np.deg2rad(15))
observation.add_viability_check_to_settings(
    [observation_simulation_settings_jup],
    [elevation_setting]
)
""""
Estimation setup
"""
#%%
observation_settings_list = []
observation_settings_list.append(observation_settings_list_io)
observation_settings_list.append(observation_settings_list_jup)
observation_simulation_settings = []
observation_simulation_settings.append(observation_simulation_settings_io)
observation_simulation_settings.append(observation_simulation_settings_jup)

# Create the estimation object for Io and Jupiter
estimator = numerical_simulation.Estimator(
    bodies,
    parameters_to_estimate,
    observation_settings_list,
    integrator_settings,
    propagator_settings)

# Simulate required observation on Io and Jupiter
simulated_observations = estimation.simulate_observations(
    observation_simulation_settings,
    estimator.observation_simulators,
    bodies)

# Collect all inputs for the inversion in a POD
truth_parameters = parameters_to_estimate.parameter_vector
pod_input = estimation.PodInput(
    simulated_observations, parameters_to_estimate.parameter_set_size, inverse_apriori_covariance=covariance_a_priori_inverse)

pod_input.define_estimation_settings(
    reintegrate_variational_equations=False)

# Setup the weight matrix W
weights_per_observable = \
    {estimation_setup.observation.angular_position_type: noise_level_jup ** -2}
pod_input.set_constant_weight_per_observable(weights_per_observable)

""""
Run the estimation
"""
# Perform estimation (this also prints the residuals and partials)
convergence = estimation.estimation_convergence_checker(1)
pod_output = estimator.perform_estimation(pod_input, convergence_checker=convergence)



""""
Post process the results 
"""
plt.figure(figsize=(9,5))
plt.imshow(np.abs(pod_output.correlations), aspect='auto', interpolation='none')
plt.title("Correlation between the outputs")
plt.colorbar()
plt.tight_layout()
plt.show()



""""
Propagate the covariance matrix for prediction
"""
#%%
covariance_to_propagate = pod_output.covariance
propagated_covariance_dict = dict()
propagated_covariance_rsw_dict_io = dict()
propagated_covariance_rsw_dict_jup = dict()
propagated_formal_errors_dict = dict()
propagated_formal_errors_rsw_dict_io = dict()
propagated_formal_errors_rsw_dict_jup = dict()

for epoch in list(variational_equations_solver.state_history):
    STM = variational_equations_solver.state_transition_matrix_history[epoch]
    full_STM = STM
    # return propagated covariance at epoch
    propagated_covariance_dict[epoch] = lalg.multi_dot([full_STM, covariance_to_propagate, full_STM.transpose()])
    propagated_covariance_rsw_dict_io[epoch] = lalg.multi_dot([rotation_rsw_to_inertial_dict_io[epoch].T,propagated_covariance_dict[epoch][:3,:3],rotation_rsw_to_inertial_dict_io[epoch]])
    propagated_formal_errors_rsw_dict_io[epoch] = np.sqrt(np.diag(propagated_covariance_rsw_dict_io[epoch]))
    propagated_covariance_rsw_dict_jup[epoch] = lalg.multi_dot([rotation_rsw_to_inertial_dict_jup[epoch].T,propagated_covariance_dict[epoch][6:9,6:9],rotation_rsw_to_inertial_dict_jup[epoch]])
    propagated_formal_errors_rsw_dict_jup[epoch] = np.sqrt(np.diag(propagated_covariance_rsw_dict_jup[epoch]))

#%%
""""
Plot the propagated uncertainties  
"""

time_io = np.array(list(propagated_formal_errors_rsw_dict_io))
time_jup = np.array(list(propagated_formal_errors_rsw_dict_jup))
values_io = np.vstack(propagated_formal_errors_rsw_dict_io.values())
values_jup = np.vstack(propagated_formal_errors_rsw_dict_jup.values())
plt.figure(figsize=(9,5))
plt.plot(time_io/86400,values_io[:,0], label = 'R')
plt.plot(time_io/86400,values_io[:,1], label = 'S')
plt.plot(time_io/86400,values_io[:,2], label = 'W')
plt.yscale("log")
plt.ylim(1, 10e3)
plt.grid(True, which="both", ls="-")
plt.title("Propagation of $\sigma$ along radial, along-track and cross-track directions Io")
plt.ylabel('Uncertainty $\sigma$ [m]')
plt.xlabel('Time since 2030 [Days]')
plt.legend()
plt.show()

plt.figure(figsize=(9,5))
plt.plot(time_jup/86400,values_jup[:,0], label = 'R')
plt.plot(time_jup/86400,values_jup[:,1], label = 'S')
plt.plot(time_jup/86400,values_jup[:,2], label = 'W')
plt.yscale("log")
plt.grid(True, which="both", ls="-")
plt.title("Propagation of $\sigma$ along radial, along-track and cross-track directions Jupiter")
plt.ylabel('Uncertainty $\sigma$ [m]')
plt.xlabel('Time since 2030 [Days]')
plt.legend()
plt.show()
state_array = result2array(states)
initial_state = np.savetxt("initial_state2.dat",state_array)


#%%
""""
Plot the propagated uncertainties in terms of RA and Dec in mas
"""
#T is the Jacobian of f1 and f2, where x,y,z are the initial state of Io at t0
T = np.block([
    [-5.04222865062317e-10,-7.28643667446313e-10,2.20819681275570e-09],
    [-2.10820830529310e-09,1.45888433446295e-09,0]
])
propagated_icrf_io = dict()
formal_errors_icrf_io = dict()
for epoch in list(variational_equations_solver.state_history):
    propagated_icrf_io[epoch] = lalg.multi_dot([T,propagated_covariance_dict[epoch][:3,:3],T.T])
    formal_errors_icrf_io[epoch] = np.sqrt(np.diag(propagated_icrf_io[epoch]))

values_icrf = np.vstack(formal_errors_icrf_io.values())
plt.figure(figsize=(9,5))
alpha = values_icrf[:,0]
dec = values_icrf[:,1]
plt.plot(time_io,alpha/4.847309743e-9, label = 'RA')
plt.title("Propagation of $\sigma$ in Right Ascension Io")
plt.ylabel('Uncertainty $\sigma$ [mas]')
plt.xlabel('Time since 2030 [Days]')
plt.legend()
plt.show()

plt.figure(figsize=(9,5))
plt.plot(time_io,dec/4.847309743e-9, label = 'Dec')
plt.title("Propagation of $\sigma$ in Declination Io")
plt.ylabel('Uncertainty $\sigma$ [mas]')
plt.xlabel('Time since 2030 [Days]')
plt.legend()
plt.show()
