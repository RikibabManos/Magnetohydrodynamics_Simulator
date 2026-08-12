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

global_data_initial = global_data_initial[6:] # remove header data
rho_data = global_data_initial[0:total_node_count]

# set up node positions
x_positions = np.linspace(0, x_node_total * cell_width, x_node_total)
y_positions = np.linspace(0, y_node_total * cell_height, y_node_total)

X, Y = np.meshgrid(x_positions, y_positions) # flattens 2D grid
x_coords = X.ravel()
y_coords = Y.ravel()

rho_max = np.max(rho_data)
rho_min = np.min(rho_data)

#def __init__():
    

# setup animation initial frame
fig = plt.figure(figsize = (16, 8))
ax = fig.add_subplot()
plot_nodes = ax.scatter(
    x_coords,
    y_coords,
    c = rho_data, # what colour is based off (density for base case)
    marker = 'o',
    cmap = 'magma',
    vmin = rho_min,
    vmax = rho_max,
    s = 50, 
)
fig.colorbar(plot_nodes, ax = ax, label = 'Density (rho)')

def update(frame):
    """ updates animation """

    current_global_state = np.fromfile(data_file_list[frame], dtype = np.float64)
    current_rho_state = current_global_state[6:total_node_count + 6]
    plot_nodes.set_array(current_rho_state)
    current_time = frame / frame_rate
    ax.set_title(                                                    
        f"1D Thermal Conduction Profile (Time: {current_time:.2f}s)"
    )
    
    return (plot_nodes, )



ani = animation.FuncAnimation(
    fig = fig,
    func = update,
    frames = len(data_file_list),
    interval = 5,
    blit = False,
    repeat = False
)

plt.show()