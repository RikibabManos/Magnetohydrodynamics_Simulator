import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import glob

def deal_with_visual_float_point_errors(max, min):
    tolerance = 5e-5

    if (max - min) < tolerance:
        midpoint = (max + min) / 2

        if abs(midpoint) < tolerance:
            max = 0.1
            min = -0.1
        else:
            max = midpoint + 0.1
            min = midpoint - 0.1

    return max, min

# make python read binary data file

data_file_list = glob.glob(r"/home/true-sigma/GithubClones/Magnetohydrodynamics_Simulator/*.dat")
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
root_mean_square_divergence_B = global_data_initial[7]
max_norm_divergence_B = global_data_initial[8]
global_data_initial = global_data_initial[9:] # remove header data

rho_data = global_data_initial[0:total_node_count]
bx_data = global_data_initial[total_node_count * 5: total_node_count * 6]
by_data = global_data_initial[total_node_count * 6: total_node_count * 7]

# define coordinates on 2D grid for streamlines
if x_node_count == y_node_count:

    x = np.linspace(0, x_node_count * cell_width, x_node_count)
    y = np.linspace(0, y_node_count * cell_height, y_node_count)
    X, Y = np.meshgrid(x, y)

else:
    print("X and Y node counts differ")

# set up node positions
rho_2D = rho_data.reshape(y_node_total, x_node_total)
bx_2D = bx_data.reshape(y_node_total, x_node_total)
by_2D = by_data.reshape(y_node_total, x_node_total)

rho_2D = rho_2D[ghost_cell_count: -ghost_cell_count, ghost_cell_count: -ghost_cell_count]
bx_2D = bx_2D[ghost_cell_count: -ghost_cell_count, ghost_cell_count: -ghost_cell_count]
by_2D = by_2D[ghost_cell_count: -ghost_cell_count, ghost_cell_count: -ghost_cell_count]

rho_max = np.max(rho_data)
rho_min = np.min(rho_data)

bx_max = np.max(bx_data)
bx_min = np.min(bx_data)

by_max = np.max(by_data)
by_min = np.min(by_data)

rho_max, rho_min = deal_with_visual_float_point_errors(rho_max, rho_min)
bx_max, bx_min = deal_with_visual_float_point_errors(bx_max, bx_min)
by_max, by_min = deal_with_visual_float_point_errors(by_max, by_min)

# setup animation initial frame
fig = plt.figure(figsize = (18, 9), constrained_layout = True)
gs = gridspec.GridSpec(1, 2, wspace = 0.3, hspace = 0.5)

ax_rho = fig.add_subplot(gs[0, 1])
ax_div_B = fig.add_subplot(gs[0, 0])

div_B_rms = [root_mean_square_divergence_B]
div_B_max = [max_norm_divergence_B]
div_B_rms_line, = ax_div_B.plot([], [], color = 'red', label = 'RMS Divergence of B')
#div_B_max_line, = ax_div_B.plot([], [], color = 'blue', label = 'Max absolute Divergence of B')
time_history = [0]
ax_div_B.legend()

rho_heatmap = ax_rho.imshow(
    rho_2D,
    cmap = 'inferno',
    origin = 'lower',
    vmin = rho_min,
    vmax = rho_max,
    extent = (0, x_node_count * cell_width, 0, y_node_count * cell_height),
    interpolation = 'bilinear'
)

# streamline underlayer
ax_rho.streamplot(
    X, Y, 
    bx_2D, by_2D, 
    color = 'black', 
    density = 1.2, 
    linewidth = 1.6, 
    arrowsize = 0.8
)
# stramline overlayer
ax_rho.streamplot(
    X, Y, 
    bx_2D, by_2D, 
    color = 'white', 
    density = 1.2, 
    linewidth = 0.8, 
    arrowsize = 0.8
)

ax_rho.set_xlim(0, x_node_count * cell_width)
ax_rho.set_ylim(0, y_node_count * cell_height)
 

fig.colorbar(rho_heatmap, ax = ax_rho, label = 'Density')

# Synchronized initial lists
time_history = []
div_B_rms = []
div_B_max = []

def update(frame):
    """ updates animation """
    global time_history, div_B_rms, div_B_max

    # If animation repeats, reset the line histories cleanly
    if frame == 0:
        time_history.clear()
        div_B_rms.clear()
        div_B_max.clear()

    current_global_state = np.fromfile(data_file_list[frame], dtype = np.float64)
    current_time = (frame + 1) * int(frame_rate) * time_between_frames
    
    time_history.append(current_time)
    div_B_rms.append(current_global_state[7])
    
    plt.suptitle(f"2D MHD Simulation Dashboard (Time: {current_time:.3f}s)", fontsize = 16, fontweight = 'bold')

    # Update bottom graph line data and axes limits
    div_B_rms_line.set_data(time_history, div_B_rms)
    current_max = max(div_B_rms)
    ax_div_B.set_ylim(0, max(current_max * 1.1, 1e-15))
    ax_div_B.set_xlim(0, current_time if current_time > 0 else 1e-5)
    
    # Extract and slice density data to remove ghost cells (fixes shape mismatch)
    current_rho_state = current_global_state[9: total_node_count + 9]
    current_rho_2D = current_rho_state.reshape(y_node_total, x_node_total)
    current_rho_2D = current_rho_2D[ghost_cell_count: -ghost_cell_count, ghost_cell_count: -ghost_cell_count]

    current_bx_state = current_global_state[total_node_count * 5: total_node_count * 6]
    current_bx_2D = current_bx_state.reshape(y_node_total, x_node_total)
    current_bx_2D = current_bx_2D[ghost_cell_count: -ghost_cell_count, ghost_cell_count: -ghost_cell_count]#

    current_by_state = current_global_state[total_node_count * 6: total_node_count * 7]
    current_by_2D = current_by_state.reshape(y_node_total, x_node_total)
    current_by_2D = current_by_2D[ghost_cell_count: -ghost_cell_count, ghost_cell_count: -ghost_cell_count]
    
    rho_heatmap.set_array(current_rho_2D)
    ax_rho.set_title("Density with B Field streamlines")

    for i in list(ax_rho.collections) + list(ax_rho.patches):
        i.remove()

    # streamline underlayer
    ax_rho.streamplot(
        X, Y, 
        current_bx_2D, current_by_2D, 
        color = 'black', 
        density = 1.2, 
        linewidth = 1.6, 
        arrowsize = 0.8
    )

    # stramline overlayer
    ax_rho.streamplot(
        X, Y, 
        current_bx_2D, current_by_2D, 
        color = 'white', 
        density = 1.2, 
        linewidth = 0.8, 
        arrowsize = 0.8
    )

    ax_rho.set_xlim(0, x_node_count * cell_width)
    ax_rho.set_ylim(0, y_node_count * cell_height)

    return (div_B_rms_line, rho_heatmap,)

ani = animation.FuncAnimation(
    fig = fig,
    func = update,
    frames = len(data_file_list),
    interval = 50,
    blit = False,
    repeat = True
)

#ani.save(
#    "placeholder1.gif",
#    writer = "pillow",
#    fps  = 1000,
#    dpi = 100
#)

plt.show()