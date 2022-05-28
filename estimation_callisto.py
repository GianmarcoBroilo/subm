## Context
"""
This code performs a covariance analysis of the state of Callisto and Jupiter. The covariance matrix will be propagated
to see the behavior of the uncertainties of the two bodies in the RSW reference frame. The simulated observables are for
Callisto a predicted stellar occultation in 2024 (using SORA) and for Jupiter a combination of VLBI and range observables
simulated once every JUNO orbit, that is 53.4 days.
INPUT
Data type: stellar occultation of Callisto, VLBI and RANGE of Jupiter
A priori cov: uncertainties in the state of Callisto and Jupiter in RSW frame
parameters: initial state of Callisto, initial state of Jupiter
OUTPUT
cov: uncertainty and correlation of estimated parameters for both Callisto and Jupiter
"""
#%%


# Load standard modules
import numpy as np
from numpy import linalg as lalg
from matplotlib import pyplot as plt
import datetime
# Load tudatpy modules
#from tudatpy.util import result2array

#import sys
#sys.path.insert(0,'/Users/gianmarcobroilo/Desktop/source-code/cmake-build-debug/tudatpy/tudatpy')
#import kernel
#from kernel.numerical_simulation.estimation import PodInput
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import estimation, estimation_setup
from tudatpy.kernel.numerical_simulation.estimation_setup import observation
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.numerical_simulation import propagation_setup, propagation
from tudatpy.kernel.astro import frame_conversion
from tudatpy.kernel.astro import time_conversion
from sklearn.preprocessing import normalize



# Load spice kernels
spice.load_standard_kernels()

# Set simulation start and end epochs START: 2023-01-01 END: 2027-01-01
calendar_start = datetime.datetime(2023,1,1)
simulation_start_epoch = time_conversion.calendar_date_to_julian_day_since_epoch(calendar_start)*constants.JULIAN_DAY
simulation_end_epoch = simulation_start_epoch +  5*constants.JULIAN_YEAR


## Environment setup


# Create default body settings
bodies_to_create = ["Earth", "Callisto", "Jupiter","Sun","Saturn","Ganymede"]

time_step = 1500
initial_time = simulation_start_epoch - 5*time_step
final_time = simulation_end_epoch + 5*time_step

# Create default body settings for bodies_to_create, with "Jupiter"/"J2000" as the global frame origin and orientation
global_frame_origin = "SSB"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

# Create tabulated settings fo Callisto and Jupiter
original_callisto_ephemeris_settings = body_settings.get("Callisto").ephemeris_settings
body_settings.get("Callisto").ephemeris_settings = environment_setup.ephemeris.tabulated_from_existing(
    original_callisto_ephemeris_settings,initial_time, final_time, time_step)

original_jupiter_ephemeris_settings = body_settings.get("Jupiter").ephemeris_settings
body_settings.get("Jupiter").ephemeris_settings = environment_setup.ephemeris.tabulated_from_existing(
    original_jupiter_ephemeris_settings,initial_time, final_time, time_step)


# Rotation model
body_settings.get("Callisto").rotation_model_settings = environment_setup.rotation_model.synchronous("Jupiter",
                                                                                               global_frame_orientation,"Callisto_Fixed")
# Create system of bodies
bodies = environment_setup.create_system_of_bodies(body_settings)


"""
Propagation setup
"""

# Define bodies that are propagated
bodies_to_propagate = ["Callisto","Jupiter"]

# Define central bodies of propagation
central_bodies = []
for body_name in bodies_to_create:
    if body_name == "Callisto":
        central_bodies.append("Jupiter")
    elif body_name == "Jupiter":
        central_bodies.append("Sun")


# Create the acceleration model
acceleration_settings_cal = dict(
    Jupiter = [propagation_setup.acceleration.mutual_spherical_harmonic_gravity(
        8,0,2,2)],
    Sun = [propagation_setup.acceleration.point_mass_gravity()],
    Saturn = [propagation_setup.acceleration.point_mass_gravity()],
    Ganymede = [propagation_setup.acceleration.mutual_spherical_harmonic_gravity(
        2,2,2,2)]
)
acceleration_settings_jup = dict(
    Sun=[propagation_setup.acceleration.point_mass_gravity()],
    Saturn = [propagation_setup.acceleration.point_mass_gravity()]
)
acceleration_settings = {"Callisto": acceleration_settings_cal, "Jupiter": acceleration_settings_jup}

acceleration_models = propagation_setup.create_acceleration_models(
    bodies,acceleration_settings,bodies_to_propagate,central_bodies
)

# Define the initial state
"""
The initial state of Callisto and Jupiter that will be propagated is now defined 
"""

# Set the initial state of Io and Jupiter
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
Propagate the dynamics of Jupiter and Callisto and extract state transition and sensitivity matrices
"""
#Setup paramaters settings to propagate the state transition matrix
parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
# Create the parameters that will be estimated
parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies)

# Create numerical integrator settings.
integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
    simulation_start_epoch, 1800.0, propagation_setup.integrator.rkf_78, 1800.0, 1800.0, 1.0, 1.0
)
# Create the variational equation solver and propagate the dynamics
variational_equations_solver = numerical_simulation.SingleArcVariationalSimulator(
    bodies, integrator_settings, propagator_settings, parameters_to_estimate, integrate_on_creation=True)

# Extract the resulting state history, state transition matrix history, and sensitivity matrix history
states = variational_equations_solver.state_history
state_transition_matrix = variational_equations_solver.state_transition_matrix_history
sensitivity_matrix = variational_equations_solver.sensitivity_matrix_history

""""
Define the a priori covariance of Callisto 
"""
#%%
#15km RSW position 0.15,1.15,0.75m/s RSW velocity
rotation_rsw_to_inertial_dict_cal = dict()
for epoch in list(variational_equations_solver.state_history):
    rotation_rsw_to_inertial_dict_cal[epoch] = frame_conversion.rsw_to_inertial_rotation_matrix(states[epoch][:6]).reshape(3,3)
uncertainties_rsw_cal = np.zeros((3,3))
np.fill_diagonal(uncertainties_rsw_cal,[15e3,15e3,15e3])
uncertainties_rsw_velocity_cal = np.zeros((3,3))
np.fill_diagonal(uncertainties_rsw_velocity_cal,[0.15,1.15,0.75])
covariance_position_initial_cal = lalg.multi_dot([rotation_rsw_to_inertial_dict_cal[simulation_start_epoch],uncertainties_rsw_cal,rotation_rsw_to_inertial_dict_cal[simulation_start_epoch].T])
covariance_velocity_initial_cal = lalg.multi_dot([rotation_rsw_to_inertial_dict_cal[simulation_start_epoch],uncertainties_rsw_velocity_cal,rotation_rsw_to_inertial_dict_cal[simulation_start_epoch].T])

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
    [covariance_position_initial_cal, np.zeros((3,3)), np.zeros((3,3)),np.zeros((3,3))],
    [np.zeros((3,3)),covariance_velocity_initial_cal, np.zeros((3,3)), np.zeros((3,3))],
    [np.zeros((3,3)),np.zeros((3,3)),covariance_position_initial_jup, np.zeros((3,3))],
    [np.zeros((3,3)),np.zeros((3,3)), np.zeros((3,3)), covariance_velocity_initial_jup],
])
covariance_a_priori_inverse = np.linalg.inv(covariance_a_priori)
""""
Observation Setup
"""
#%%

# Define the uplink/downlink link ends types
link_ends_stellar = dict()
link_ends_stellar[observation.receiver] = ("Earth", "")
link_ends_stellar[observation.transmitter] = ("Callisto", "")
link_ends_vlbi = dict()
link_ends_vlbi[observation.receiver] = ("Earth", "")
link_ends_vlbi [observation.transmitter] = ("Jupiter", "")
link_ends_position = dict()
link_ends_position[observation.observed_body] = ("Callisto","")


# Create observation settings for each link/observable
observation_settings_list_cal = observation.angular_position(link_ends_stellar)
observation_settings_list_jup = observation.angular_position(link_ends_vlbi)
observation_settings_list_position = observation.cartesian_position(link_ends_position)
observation_settings_list_range = observation.one_way_range(link_ends_vlbi)

# Define the observations for Callisto stellar occultation and 3D position
stellar_occ = datetime.datetime(2024,1,15)
stellar_occ = time_conversion.calendar_date_to_julian_day_since_epoch(stellar_occ)*constants.JULIAN_DAY
observation_times_cal = np.array([stellar_occ])

observations_position = np.arange(simulation_start_epoch,simulation_end_epoch, 1*constants.JULIAN_YEAR)


observation_simulation_settings_cal = observation.tabulated_simulation_settings(
    observation.angular_position_type,
    link_ends_stellar,
    observation_times_cal
)

observation_3dposition = observation.tabulated_simulation_settings(
    observation.position_observable_type,
    link_ends_position,
    observations_position,
    reference_link_end_type = observation.observed_body
)

# Define the observations for Jupiter VLBI and RANGE
end = datetime.datetime(2024,2,1)
end = time_conversion.calendar_date_to_julian_day_since_epoch(end)*constants.JULIAN_DAY
end2 = datetime.datetime(2025,10,20)
end2 = time_conversion.calendar_date_to_julian_day_since_epoch(end2)*constants.JULIAN_DAY

observation_times_jup1 = np.arange(simulation_start_epoch,end,53.4*constants.JULIAN_DAY)
observation_times_jup2 = np.arange(end,end2,33*constants.JULIAN_DAY)
observation_times_jup = np.append(observation_times_jup1,observation_times_jup2)

observation_simulation_settings_jup = observation.tabulated_simulation_settings(
    observation.angular_position_type,
    link_ends_vlbi,
    observation_times_jup
)

observation_simulation_settings_range = observation.tabulated_simulation_settings(
    observation.one_way_range_type,
    link_ends_vlbi,
    observation_times_jup
)

# Add noise levels of roughly 1 mas to Callisto 10 mas = 4.8481368E-8 5 mas = 2.4240684E-8 1 mas = 4.8481368E-9
noise_level_cal = 2.4240684e-8
observation.add_gaussian_noise_to_settings(
    [observation_simulation_settings_cal],
    noise_level_cal,
    observation.angular_position_type
)

# Add noise level of 15km to position observable
noise_level_position = 15e3
observation.add_gaussian_noise_to_settings(
    [observation_3dposition],
    noise_level_position,
    observation.position_observable_type
)

# Add noise levels of roughly 0.5 nrad to Jupiter
noise_level_jup = 0.5e-9
observation.add_gaussian_noise_to_settings(
    [observation_simulation_settings_jup],
    noise_level_jup,
    observation.angular_position_type
)

# Add noise level of 10m to Jupiter
noise_level_range = 10
observation.add_gaussian_noise_to_settings(
    [observation_simulation_settings_range],
    noise_level_range,
    observation.one_way_range_type
)

""""
Estimation setup
"""
#%%
# Collect all settings required to simulate the observations
observation_settings_list = []
#observation_settings_list.append(observation_settings_list_cal)
observation_settings_list.append(observation_settings_list_jup)
#observation_settings_list.append(observation_settings_list_position)
#observation_settings_list.append(observation_settings_list_range)

observation_simulation_settings = []
#observation_simulation_settings.append(observation_simulation_settings_cal)
observation_simulation_settings.append(observation_simulation_settings_jup)
#observation_simulation_settings.append(observation_3dposition)
#observation_simulation_settings.append(observation_simulation_settings_range)

# Create the estimation object for Callisto and Jupiter
estimator = numerical_simulation.Estimator(
    bodies,
    parameters_to_estimate,
    observation_settings_list,
    integrator_settings,
    propagator_settings)

# Simulate required observation on Callisto and Jupiter
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

# Setup the weight matrix W with weights for Callisto and weights for Jupiter
weights_vlbi = noise_level_jup ** -2
weights_stellar = noise_level_cal ** -2
weights_position = noise_level_position ** -2
weights_range = noise_level_range ** -2
pod_input.set_constant_weight_for_observable_and_link_ends(observation.angular_position_type,link_ends_vlbi,weights_vlbi)
#pod_input.set_constant_weight_for_observable_and_link_ends(observation.angular_position_type,link_ends_stellar,weights_stellar)
#pod_input.set_constant_weight_for_observable_and_link_ends(observation.position_observable_type,link_ends_position,weights_position)
#pod_input.set_constant_weight_for_observable_and_link_ends(observation.one_way_range_type,link_ends_vlbi,weights_range)

""""
Run the estimation
"""
# Perform estimation (this also prints the residuals and partials)
convergence = estimation.estimation_convergence_checker(1)
pod_output = estimator.perform_estimation(pod_input, convergence_checker=convergence)



""""
Post process the results 
"""
# Plot the correlation between the outputs
plt.figure(figsize=(9,5))
plt.imshow(np.abs(pod_output.correlations), aspect='auto', interpolation='none')
plt.title("Correlation between the outputs")
plt.colorbar()
plt.tight_layout()
plt.show()

#%%
covariance_to_propagate = pod_output.covariance
propagated_covariance_dict = dict()
propagated_covariance_rsw_dict_cal = dict()
propagated_covariance_rsw_dict_jup = dict()
propagated_formal_errors_dict = dict()
propagated_formal_errors_rsw_dict_cal = dict()
propagated_formal_errors_rsw_dict_jup = dict()

for epoch in list(variational_equations_solver.state_history):
    STM = variational_equations_solver.state_transition_matrix_history[epoch]
    full_STM = STM
    # return propagated covariance at epoch
    propagated_covariance_dict[epoch] = lalg.multi_dot([full_STM, covariance_to_propagate, full_STM.transpose()])
    propagated_covariance_rsw_dict_cal[epoch] = lalg.multi_dot([rotation_rsw_to_inertial_dict_cal[epoch].T,propagated_covariance_dict[epoch][:3,:3],rotation_rsw_to_inertial_dict_cal[epoch]])
    propagated_formal_errors_rsw_dict_cal[epoch] = np.sqrt(np.diag(propagated_covariance_rsw_dict_cal[epoch]))
    propagated_covariance_rsw_dict_jup[epoch] = lalg.multi_dot([rotation_rsw_to_inertial_dict_jup[epoch].T,propagated_covariance_dict[epoch][6:9,6:9],rotation_rsw_to_inertial_dict_jup[epoch]])
    propagated_formal_errors_rsw_dict_jup[epoch] = np.sqrt(np.diag(propagated_covariance_rsw_dict_jup[epoch]))


#%%
""""
Propagate the covariance matrix for prediction
"""
state_transition = estimator.state_transition_interface
cov_initial = pod_output.covariance

# Create covariance dictionaries
propagated_times = dict()
propagated_covariance_rsw_dict = dict()
propagated_covariance_rsw_dict_cal = dict()
propagated_covariance_rsw_dict_jup = dict()
propagated_formal_errors_rsw_dict_cal = dict()
propagated_formal_errors_rsw_dict_jup = dict()

time = np.arange(simulation_start_epoch, simulation_end_epoch, 86400)
time = np.ndarray.tolist(time)

propagation = estimation.propagate_covariance_split_output(initial_covariance= cov_initial,state_transition_interface = state_transition,output_times = time)

propagated_times = propagation[0]
propagated_times = [int(num) for num in propagated_times]
propagated_covariance = propagation[1]

propagated_covariance_dict = {propagated_times[i]: propagated_covariance[i] for i in range(len(propagated_times))}

for epoch in list(propagated_covariance_dict):
    propagated_covariance_rsw_dict_cal[epoch] = lalg.multi_dot([rotation_rsw_to_inertial_dict_cal[epoch].T,propagated_covariance_dict[epoch][:3,:3],rotation_rsw_to_inertial_dict_cal[epoch]])
    propagated_formal_errors_rsw_dict_cal[epoch] = np.sqrt(np.diag(propagated_covariance_rsw_dict_cal[epoch]))
    propagated_covariance_rsw_dict_jup[epoch] = lalg.multi_dot([rotation_rsw_to_inertial_dict_jup[epoch].T,propagated_covariance_dict[epoch][6:9,6:9],rotation_rsw_to_inertial_dict_jup[epoch]])
    propagated_formal_errors_rsw_dict_jup[epoch] = np.sqrt(np.diag(propagated_covariance_rsw_dict_jup[epoch]))


# times are equal to epochs in state history
time_cal = np.array(list(propagated_times))
time_jup = np.array(list(propagated_times))
values_cal = np.vstack(propagated_formal_errors_rsw_dict_cal.values())
values_jup = np.vstack(propagated_formal_errors_rsw_dict_jup.values())
tc = time_cal/31536000
tj = time_jup/31536000
#%%
""""
Plot the propagated uncertainties  
"""

plt.figure(figsize=(9,5))
plt.plot(tc,values_cal[:,0], label = 'R', color = 'salmon')
plt.plot(tc,values_cal[:,1], label = 'S', color = 'orange')
plt.plot(tc,values_cal[:,2], label = 'W', color = 'cornflowerblue')
#plt.plot(observation_times_cal/31536000, 100,'o')
plt.yscale("log")
plt.grid(True, which="both", ls="--")
plt.title("Propagation of $\sigma$ along radial, along-track and cross-track directions Callisto")
plt.ylabel('Uncertainty $\sigma$ [m]')
plt.xlabel('Time [years after J2000]')
plt.legend()
plt.show()

plt.figure(figsize=(9,5))
plt.plot(tj,values_jup[:,0], label = 'R', color = 'salmon')
plt.plot(tj,values_jup[:,1], label = 'S', color = 'orange')
plt.plot(tj,values_jup[:,2], label = 'W', color = 'cornflowerblue')
plt.yscale("log")
plt.grid(True, which="both", ls="--")
plt.title("Propagation of $\sigma$ along radial, along-track and cross-track directions Jupiter")
plt.ylabel('Uncertainty $\sigma$ [m]')
plt.xlabel('Time [years after J2000]')
plt.legend()
plt.show()
#%%
""""
Export data  
"""
from tudatpy.util import result2array
state_array = result2array(states)
initial_state = np.savetxt("initial_state.dat",state_array)
uncertainty_cal = np.savetxt("uncertainty_cal.dat",values_cal)
uncertainty_jup = np.savetxt("uncertainty_jup.dat",values_jup)
time_prop = np.savetxt("time_prop.dat",time_cal)
obs = np.savetxt("observations_stellar.dat",observation_times_cal)
obs2 = np.savetxt("observations_vlbi.dat",observation_times_jup)


#%%
""""
Propagate RA and DEC of Jupiter  
"""

T = np.block([
    [-8.34313652508797e-14,-1.73092383777687e-14,1.34742402731453e-12],
    [-2.74811677043994e-13,1.32460556046833e-12,0]
])

propagated_icrf_jup = dict()
formal_errors_icrf_jup = dict()
for epoch in list(propagated_covariance_dict):
    propagated_icrf_jup[epoch] = lalg.multi_dot([T,propagated_covariance_dict[epoch][6:9,6:9],T.T])
    formal_errors_icrf_jup[epoch] = np.sqrt(np.diag(propagated_icrf_jup[epoch]))

values_icrf = np.vstack(formal_errors_icrf_jup.values())
alpha = values_icrf[:,0]
dec = values_icrf[:,1]

alpha_marcsec = alpha*206264806.71915
delta_marcsec = dec*206264806.71915

fig, axs = plt.subplots(2,figsize=(12, 6))
fig.suptitle('Propagated uncertainties in Right Ascension and Declination of Jupiter')


axs[0].plot(tj,alpha_marcsec,'o', color = 'black')
axs[0].set_ylabel('Right Ascension [mas]')

axs[1].plot(time_cal/31536000,delta_marcsec,'o', color = 'black')
axs[1].set_ylabel('Declination [mas]')
axs[1].set_xlabel('Time [years after J2000]')
plt.show()
