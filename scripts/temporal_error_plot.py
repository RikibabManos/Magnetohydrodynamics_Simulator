import matplotlib.pyplot as plt
import numpy as np
import glob
from display_functions import prepare_snapshot_data

def exact_function(x_coordinate, y_coordinate, physical_grid_width, physical_grid_height, current_time):
    """ Calculates the exact By field value at a given coordinate """

    by_amplitude = 1.0 # note that these by_amplitude, resistivity, kx and ky values must be identical to what is defined within the decaying_magnetic_sine_wave_initial_conditions function in the integrator_main.cpp
    resistivity = 10.0
    kx = 2 * np.pi / physical_grid_width  
    ky = 2 * np.pi / physical_grid_height
    return by_amplitude * np.sin(kx * x_coordinate) * np.sin(ky * y_coordinate) * np.exp(- resistivity * (kx * kx + ky * ky) * current_time)


target_sim_time = 0.05 # seconds (simulation time which log-log plot will pick data from)

directories = [
    r"/home/true-sigma/GithubClones/Magnetohydrodynamics_Simulator/testData/decayingMagneticWave_0.01_secondTimeStep/*.dat",
    r"/home/true-sigma/GithubClones/Magnetohydrodynamics_Simulator/testData/decayingMagneticWave_0.005_secondTimeStep/*.dat",
    r"/home/true-sigma/GithubClones/Magnetohydrodynamics_Simulator/testData/decayingMagneticWave_0.0025_secondTimeStep/*.dat",
    r"/home/true-sigma/GithubClones/Magnetohydrodynamics_Simulator/testData/decayingMagneticWave_0.00125_secondTimeStep/*.dat",
    r"/home/true-sigma/GithubClones/Magnetohydrodynamics_Simulator/testData/decayingMagneticWave_0.000625_secondTimeStep/*.dat",
    r"/home/true-sigma/GithubClones/Magnetohydrodynamics_Simulator/testData/decayingMagneticWave_0.0003125_secondTimeStep/*.dat",
    r"/home/true-sigma/GithubClones/Magnetohydrodynamics_Simulator/testData/decayingMagneticWave_0.00015625_secondTimeStep/*.dat"
]

error_data = []
timestep_data = []

for folder in directories:

    current_timestep_snapshot = glob.glob(folder) 
    current_timestep_snapshot.sort()

    if not current_timestep_snapshot:
        print(f"Skipping empty or missing directory: {folder}")
        continue

    # the only data we are interested in for this log log plot is the time step value and the by field

    header_data = np.fromfile(current_timestep_snapshot[0], dtype = np.float64)
    current_frame_rate = int(header_data[3])
    current_timestep = header_data[6]
    x_node_count = int(header_data[0])
    y_node_count = int(header_data[1])
    ghost_cell_count = int(header_data[2])
    x_node_total = x_node_count + 2 * ghost_cell_count
    y_node_total = y_node_count + 2 * ghost_cell_count
    shape_data = [x_node_total, y_node_total, ghost_cell_count]
    cell_width = header_data[4]
    cell_hieght = header_data[5]
    grid_width = x_node_count * cell_width
    grid_height = y_node_count* cell_hieght
    half_cell_width = 0.5 * cell_width
    half_cell_height = 0.5 * cell_hieght

    best_file = None
    best_time_diff = float('inf')
    chosen_actual_time = 0.0

    for idx, snapshot_path in enumerate(current_timestep_snapshot):

        t_frame = idx * current_frame_rate * current_timestep
        difference = abs(t_frame - target_sim_time)

        if difference < best_time_diff:
            best_time_diff = difference
            best_file = snapshot_path
            chosen_actual_time = t_frame

    current_timestep_data = np.fromfile(best_file, dtype = np.float64)
    current_properties_data = prepare_snapshot_data(current_timestep_data, shape_data)
    current_by_data = current_properties_data["by_2D"]

    # generate the exact By field matrix
    x_values = np.linspace(half_cell_width, grid_width - half_cell_width, x_node_count)
    y_values = np.linspace(half_cell_height, grid_height - half_cell_height, y_node_count)
    X, Y = np.meshgrid(x_values, y_values)
    exact_by_matrix = exact_function(X, Y, grid_width, grid_height, target_sim_time)

    current_error = np.sqrt(np.mean((exact_by_matrix - current_by_data) ** 2)) # root mean square error
    spatial_floor = 1.39e-4 # baseline spatial error floor for 200x200 grid, we only want to see the time dependace of the error!
    temporal_error_data = current_error - spatial_floor

    error_data.append(temporal_error_data)
    timestep_data.append(current_timestep)

    print(f"dt: {current_timestep:.7f} s | RMS Error: {temporal_error_data:.8e} | t_actual: {chosen_actual_time:.4f}s")

slope, intercept = np.polyfit(np.log10(timestep_data), np.log10(error_data), 1)

log_dt_min = np.log10(np.min(timestep_data))
log_dt_max = np.log10(np.max(timestep_data))

x_data_log = np.linspace(log_dt_min, log_dt_max, 1000)
y_data_log = slope * x_data_log + intercept

line_bf_x = 10 ** x_data_log
line_bf_y = 10 ** y_data_log

fig = plt.figure(figsize = (10, 6))
ax = fig.add_subplot()
ax.set_title("Log(error) against Log(timestep) for decaying magnetic sine wave")
fit_label = f'Line of Best Fit: $E = {intercept:.2f} \cdot (\Delta t)^{{{slope:.2f}}}$ (Slope $p = {slope:.5f}$)'
plt.loglog(timestep_data, error_data, 'ro', markersize = 4)
plt.loglog(line_bf_x, line_bf_y, 'b--', label = fit_label)
plt.grid(True, which = "both", linestyle = ":", alpha = 0.7)
plt.tight_layout()
plt.legend()
plt.show()