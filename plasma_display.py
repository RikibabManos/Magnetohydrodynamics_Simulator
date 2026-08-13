import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import glob

# make python read binary data file

data_file_list = glob.glob(r"C:\Codes\mhd_sim\snapshot_*.dat")
data_file_list.sort()                                                    # sorts files alphabetically
global_data_initial = np.fromfile(data_file_list[0], dtype = np.float64) # extract data from initial file to find header data and initial state

# extracting headers
x_node_count = int(global_data_initial[0])
y_node_count = int(global_data_initial[1])
ghost_cell_count = int(global_data_initial[2])
x_node_total = x_node_count + 2 * ghost_cell_count
y_node_total = y_node_count + 2 * ghost_cell_count
total_node_count = x_node_total * y_node_total # does NOT account for any ghost cells
frame_rate = int(global_data_initial[3])
cell_width = global_data_initial[4]
cell_height = global_data_initial[5]
time_between_frames = global_data_initial[6]

global_data_initial = global_data_initial[7:] # remove header data
rho_data = global_data_initial[0:total_node_count]

# set up node positions
rho_2D = rho_data.reshape(y_node_total, x_node_total)

rho_max = np.max(rho_data)
rho_min = np.min(rho_data)
    

# setup animation initial frame
fig, ax = plt.subplots(figsize = (10, 8))

rho_heatmap = plt.imshow(
    rho_2D,
    cmap = 'magma',
    origin = 'lower',
    vmin = rho_min,
    vmax = rho_max,
    extent = (0, x_node_total, 0, y_node_total),
    interpolation = 'nearest'
)

fig.colorbar(rho_heatmap, ax = ax, label = 'Density (rho)')

def update(frame):
    """ updates animation """

    current_global_state = np.fromfile(data_file_list[frame], dtype = np.float64)
    current_rho_state = current_global_state[7:total_node_count + 7]
    current_rho_2D = current_rho_state.reshape(y_node_total, x_node_total)
    rho_heatmap.set_array(current_rho_2D)
    current_time = frame * frame_rate * time_between_frames
    ax.set_title(                                                    
        f"MHD SIM (Time: {current_time:.2f}s)"
    )
    
    return (rho_heatmap, )

ani = animation.FuncAnimation(
    fig = fig,
    func = update,
    frames = len(data_file_list),
    interval = 50,
    blit = False,
    repeat = True
)

plt.show()