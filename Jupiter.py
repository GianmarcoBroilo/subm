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
from tudatpy.kernel.astro import time_conversion
import datetime

# Load spice kernels
spice.load_standard_kernels()

# Set simulation start and end epochs
simulation_start_epoch = 0
simulation_end_epoch = simulation_start_epoch +  1*constants.JULIAN_YEAR


## Environment setup


# Create default body settings
bodies_to_create = ["Earth", "Jupiter","Sun"]

time_step = 1500
initial_time = simulation_start_epoch - 5*time_step
final_time = simulation_end_epoch + 5*time_step

# Create default body settings for bodies_to_create, with "Jupiter"/"J2000" as the global frame origin and orientation
global_frame_origin = "SSB"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

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
bodies_to_propagate = ["Jupiter"]

# Define central bodies of propagation
central_bodies = ["Sun"]


### Create the acceleration model
"""
The acceleration settings that act on `Io` are now to be defined.
* Gravitational acceleration using a Point Mass model from:
    - Jupiter
"""

# Define the accelerations acting on Io
accelerations_settings_jup = dict(
    Sun=[
        propagation_setup.acceleration.point_mass_gravity()
    ]
)

# Create global accelerations dictionary
acceleration_settings = {"Jupiter": accelerations_settings_jup}

# Create acceleration models
acceleration_models = propagation_setup.create_acceleration_models(
    bodies,
    acceleration_settings,
    bodies_to_propagate,
    central_bodies)


### Define the initial state
"""
The initial state of Jupiter that will be propagated is now defined. 
"""

# Set the initial state of Jupiter
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
#Setup paramaters settings to propagate the state transition matrix
parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
# Create the parameters that will be estimated
parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies)

# Create numerical integrator settings.
fixed_step_size = 4000.0
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
Observation Setup
"""

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

# Define the uplink link ends for one-way observable
link_ends = dict()
link_ends[observation.receiver] = ("Earth", "TrackingStation")
link_ends[observation.transmitter] = ("Jupiter", "")

# Create Bias settings 0.5 nrad in both RA and Dec
bias = observation.absolute_bias(np.array([0.5e-9,0.5e-9]))
# Create observation settings for each link/observable
observation_settings_list = [observation.angular_position(link_ends,bias_settings = bias)]

### Define Observation Simulation Settings
# Define observation simulation times for each link:
#observation_times_stellar =  np.arange(simulation_start_epoch+200*constants.JULIAN_DAY,simulation_start_epoch + 210*constants.JULIAN_DAY,60)#2459216.50,2459306.93,2459459.50
observation_times =  [150*constants.JULIAN_DAY]
observation_simulation_settings = observation.tabulated_simulation_settings(
    observation.angular_position_type,
    link_ends,
    observation_times
)

# Add noise levels of roughly 05 nrad and add this as Gaussian noise to the observation
noise_level = 0.5e-9
observation.add_gaussian_noise_to_settings(
    [observation_simulation_settings],
    noise_level,
    observation.angular_position_type
)

""""
A priori covariance
"""
# Define a priori uncertainty 15km on positions, for velocity in RSW: 0.15 m/s 1.15 m/s 0.75 m/s
rotation_rsw_to_inertial_dict = dict()
for epoch in list(variational_equations_solver.state_history):
    rotation_rsw_to_inertial_dict[epoch] = frame_conversion.rsw_to_inertial_rotation_matrix(states[epoch]).reshape(3,3)

uncertainties_rsw = np.zeros((3,3))
np.fill_diagonal(uncertainties_rsw,[1e3,1e3,1e3])
uncertainties_rsw_velocity = np.zeros((3,3))
np.fill_diagonal(uncertainties_rsw_velocity,[0.1,0.1,0.1])
covariance_position_initial = lalg.multi_dot([rotation_rsw_to_inertial_dict[simulation_start_epoch],uncertainties_rsw,rotation_rsw_to_inertial_dict[simulation_start_epoch].T])
covariance_velocity_initial = lalg.multi_dot([rotation_rsw_to_inertial_dict[simulation_start_epoch],uncertainties_rsw_velocity,rotation_rsw_to_inertial_dict[simulation_start_epoch].T])
covariance_a_priori = np.block([
    [covariance_position_initial, np.zeros((3,3))],
    [np.zeros((3,3)), covariance_velocity_initial]
])
covariance_a_priori_inverse = np.linalg.inv(covariance_a_priori)

""""
Estimation setup
"""
#%%
# Create the estimation object
estimator = numerical_simulation.Estimator(
    bodies,
    parameters_to_estimate,
    observation_settings_list,
    integrator_settings,
    propagator_settings)

# Simulate required observation
simulated_observations = estimation.simulate_observations(
    [observation_simulation_settings],
    estimator.observation_simulators,
    bodies)

# Collect all inputs for the inversion in a POD
truth_parameters = parameters_to_estimate.parameter_vector
pod_input = estimation.PodInput(
    simulated_observations, parameters_to_estimate.parameter_set_size, inverse_apriori_covariance=covariance_a_priori_inverse)

pod_input.define_estimation_settings(
    reintegrate_variational_equations=False)

# Setup the weight matrix W (for first case assume W = 1)
weights_per_observable = \
    {estimation_setup.observation.angular_position_type: noise_level ** -2}
pod_input.set_constant_weight_per_observable(weights_per_observable)

""""
Run the estimation
"""
# Perform estimation (this also prints the residuals and partials)
convergence = estimation.estimation_convergence_checker(1)
pod_output = estimator.perform_estimation(pod_input, convergence_checker=convergence)
sti = estimator.state_transition_interface

# Print the estimation error
print(pod_output.formal_errors)
print(truth_parameters - parameters_to_estimate.parameter_vector)


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
# Propagate the covariance matrix
#propagated_dict = dict()
#time = np.arange(simulation_start_epoch, 50*constants.JULIAN_DAY, 86400)
#time = np.ndarray.tolist(time)
#state_transition = sti
#cov_initial = pod_output.covariance
#propagated_dict = estimation.propagate_formal_errors(initial_covariance= cov_initial,state_transition_interface = state_transition,output_times = time)


covariance_to_propagate = pod_output.covariance

propagated_covariance_dict = dict()
propagated_covariance_rsw_dict = dict()
propagated_formal_errors_dict = dict()
propagated_formal_errors_rsw_dict = dict()

for epoch in list(variational_equations_solver.state_history):
    STM = variational_equations_solver.state_transition_matrix_history[epoch]
    full_STM = STM
    # return propagated covariance at epoch
    propagated_covariance_dict[epoch] = lalg.multi_dot([full_STM, covariance_to_propagate, full_STM.transpose()])
    propagated_formal_errors_dict[epoch] = np.sqrt(np.diag(propagated_covariance_dict[epoch]))
    propagated_covariance_rsw_dict[epoch] = lalg.multi_dot([rotation_rsw_to_inertial_dict[epoch].T,propagated_covariance_dict[epoch][:3,:3],rotation_rsw_to_inertial_dict[epoch]])
    propagated_formal_errors_rsw_dict[epoch] = np.sqrt(np.diag(propagated_covariance_rsw_dict[epoch]))

#%%
""""
Plot the propagated uncertainties  
"""

time = np.array(list(propagated_formal_errors_rsw_dict))
values = np.vstack(propagated_formal_errors_rsw_dict.values())

plt.figure(figsize=(9,5))
plt.plot(time/86400,values[:,0], label = 'R')
plt.plot(time/86400,values[:,1], label = 'S')
plt.plot(time/86400,values[:,2], label = 'W')
plt.yscale("log")
plt.scatter(simulated_observations.concatenated_times[1]/86400,1000,marker='x', color='black')
plt.scatter(simulated_observations.concatenated_times[3]/86400,1000,marker='x', color='black')
plt.grid(True, which="both", ls="-")
plt.title("Propagation of $\sigma$ along radial, along-track and cross-track directions")
plt.ylabel('Uncertainty $\sigma$ [m]')
plt.xlabel('Time')
plt.legend()
plt.show()
