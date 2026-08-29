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
root_mean_square_divergence_B = global_data_initial[7]
max_norm_divergence_B = global_data_initial[8]
global_data_initial = global_data_initial[9:] # remove header data

rho_data = global_data_initial[0:total_node_count]
E_data = global_data_initial[total_node_count: total_node_count * 2]
p_data = global_data_initial[total_node_count * 2: total_node_count * 3]
vx_data = global_data_initial[total_node_count * 3: total_node_count * 4]
vy_data = global_data_initial[total_node_count * 4: total_node_count * 5]
bx_data = global_data_initial[total_node_count * 5: total_node_count * 6]
by_data = global_data_initial[total_node_count * 6: total_node_count * 7]

# set up node positions
rho_2D = rho_data.reshape(y_node_total, x_node_total)
E_2D = E_data.reshape(y_node_total, x_node_total)
p_2D = p_data.reshape(y_node_total, x_node_total)
vx_2D = vx_data.reshape(y_node_total, x_node_total)
vy_2D = vy_data.reshape(y_node_total, x_node_total)
bx_2D = bx_data.reshape(y_node_total, x_node_total)
by_2D = by_data.reshape(y_node_total, x_node_total)

rho_2D = rho_2D[ghost_cell_count: -ghost_cell_count, ghost_cell_count: -ghost_cell_count]
E_2D = E_2D[ghost_cell_count: -ghost_cell_count, ghost_cell_count: -ghost_cell_count]
p_2D = p_2D[ghost_cell_count: -ghost_cell_count, ghost_cell_count: -ghost_cell_count]
vx_2D = vx_2D[ghost_cell_count: -ghost_cell_count, ghost_cell_count: -ghost_cell_count]
vy_2D = vy_2D[ghost_cell_count: -ghost_cell_count, ghost_cell_count: -ghost_cell_count]
bx_2D = bx_2D[ghost_cell_count: -ghost_cell_count, ghost_cell_count: -ghost_cell_count]
by_2D = by_2D[ghost_cell_count: -ghost_cell_count, ghost_cell_count: -ghost_cell_count]

rho_max = np.max(rho_data)
rho_min = np.min(rho_data)

E_max = np.max(E_data)
E_min = np.min(E_data)

p_max = np.max(p_data)
p_min = np.min(p_data)

vx_max = np.max(vx_data)
vx_min = np.min(vx_data)

vy_max = np.max(vy_data)
vy_min = np.min(vy_data)

bx_max = np.max(bx_data)
bx_min = np.min(bx_data)

by_max = np.max(by_data)
by_min = np.min(by_data)

rho_max, rho_min = deal_with_visual_float_point_errors(rho_max, rho_min)
E_max, E_min = deal_with_visual_float_point_errors(E_max, E_min)
p_max, p_min = deal_with_visual_float_point_errors(p_max, p_min)
vx_max, vx_min = deal_with_visual_float_point_errors(vx_max, vx_min)
vy_max, vy_min = deal_with_visual_float_point_errors(vy_max, vy_min)
bx_max, bx_min = deal_with_visual_float_point_errors(bx_max, bx_min)
by_max, by_min = deal_with_visual_float_point_errors(by_max, by_min)

# setup animation initial frame
fig = plt.figure(figsize = (18, 9))
gs = gridspec.GridSpec(2, 4, width_ratios = [1, 1, 1, 1], wspace = 0.5, hspace = 0.5)

ax_rho = fig.add_subplot(gs[0, 0])
ax_div_B = fig.add_subplot(gs[1, 0])
ax_E = fig.add_subplot(gs[0, 1])
ax_p = fig.add_subplot(gs[1, 1])
ax_vx = fig.add_subplot(gs[0, 2])
ax_vy = fig.add_subplot(gs[1, 2])
ax_bx = fig.add_subplot(gs[0, 3])
ax_by = fig.add_subplot(gs[1, 3])

div_B_rms = [root_mean_square_divergence_B]
div_B_max = [max_norm_divergence_B]
div_B_rms_line, = ax_div_B.plot([], [], color = 'red', label = 'RMS Divergence of B')
#div_B_max_line, = ax_div_B.plot([], [], color = 'blue', label = 'Max absolute Divergence of B')
time_history = [0]
ax_div_B.legend()

rho_heatmap = ax_rho.imshow(
    rho_2D,
    cmap = 'magma',
    origin = 'lower',
    vmin = rho_min,
    vmax = rho_max,
    extent = (0, x_node_total, 0, y_node_total),
    interpolation = 'nearest'
)

E_heatmap = ax_E.imshow(
    E_2D,
    cmap = 'cividis',
    origin = 'lower',
    vmin = E_min,
    vmax = E_max,
    extent = (0, x_node_total, 0, y_node_total),
    interpolation = 'nearest'
)

p_heatmap = ax_p.imshow(
    p_2D,
    cmap = 'plasma',
    origin = 'lower',
    vmin = p_min,
    vmax = p_max,
    extent = (0, x_node_total, 0, y_node_total),
    interpolation = 'nearest'
)

vx_heatmap = ax_vx.imshow(
    vx_2D,
    cmap = 'magma',
    origin = 'lower',
    vmin = vx_min,
    vmax = vx_max,
    extent = (0, x_node_total, 0, y_node_total),
    interpolation = 'nearest'
)

vy_heatmap = ax_vy.imshow(
    vy_2D,
    cmap = 'magma',
    origin = 'lower',
    vmin = vy_min,
    vmax = vy_max,
    extent = (0, x_node_total, 0, y_node_total),
    interpolation = 'nearest'
)

bx_heatmap = ax_bx.imshow(
    bx_2D,
    cmap = 'RdBu_r',
    origin = 'lower',
    vmin = bx_min,
    vmax = bx_max,
    extent = (0, x_node_total, 0, y_node_total),
    interpolation = 'nearest'
)

by_heatmap = ax_by.imshow(
    by_2D,
    cmap = 'coolwarm',
    origin = 'lower',
    vmin = by_min,
    vmax = by_max,
    extent = (0, x_node_total, 0, y_node_total),
    interpolation = 'nearest'
)

fig.colorbar(rho_heatmap, ax = ax_rho, label = 'Density')
fig.colorbar(E_heatmap, ax = ax_E, label = 'Energy')
fig.colorbar(p_heatmap, ax = ax_p, label = 'Thermal Pressure')
fig.colorbar(vx_heatmap, ax = ax_vx, label = 'Velocity (x)')
fig.colorbar(vy_heatmap, ax = ax_vy, label = 'Velocity (y)')
fig.colorbar(bx_heatmap, ax = ax_bx, label = 'Magnetic Flux Density (x)')
fig.colorbar(by_heatmap, ax = ax_by, label = 'Magnetic Flux Density (y)')

def update(frame):
    """ updates animation """

    global time_history, div_B_rms, div_B_max

    # if animation repeated, reset the line histories
    if frame == 0:
        time_history.clear()
        div_B_rms.clear()
        div_B_max.clear()

    current_global_state = np.fromfile(data_file_list[frame], dtype = np.float64)
    current_time = (frame + 1) * int(frame_rate) * time_between_frames
    time_history.append(current_time)
    plt.suptitle(f"2D MHD Simulation Dashboard (Time: {current_time:.3f}s)", fontsize = 16, fontweight = 'bold')

    div_B_rms.append(current_global_state[7])
    #div_B_max.append(current_global_state[8])
    div_B_rms_line.set_data(time_history, div_B_rms)
    #div_B_max_line.set_data(time_history, div_B_max)
    current_max = max(div_B_rms)
    ax_div_B.set_ylim(0, max(current_max * 1.1, 1e-15))
    ax_div_B.set_xlim(0, current_time if current_time > 0 else 1e-5)
    
    current_rho_state = current_global_state[9: total_node_count + 9]
    current_rho_2D = current_rho_state.reshape(y_node_total, x_node_total)
    rho_heatmap.set_array(current_rho_2D)
    ax_rho.set_title(                                                    
        "Density"
    )

    current_E_state = current_global_state[9 + total_node_count: 2 *total_node_count + 9]
    current_E_2D = current_E_state.reshape(y_node_total, x_node_total)
    E_heatmap.set_array(current_E_2D)
    ax_E.set_title(                                                    
        "Energy"
    )

    current_p_state = current_global_state[9 + total_node_count * 2: 3 * total_node_count + 9]
    current_p_2D = current_p_state.reshape(y_node_total, x_node_total)
    p_heatmap.set_array(current_p_2D)
    ax_p.set_title(                                                    
        "Thermal Pressure"
    )

    current_vx_state = current_global_state[9 + total_node_count * 3: 4 * total_node_count + 9]
    current_vx_2D = current_vx_state.reshape(y_node_total, x_node_total)
    vx_heatmap.set_array(current_vx_2D)
    ax_vx.set_title(                                                    
        "Velocity (x)"
    )

    current_vy_state = current_global_state[9 + total_node_count * 4: 5 * total_node_count + 9]
    current_vy_2D = current_vy_state.reshape(y_node_total, x_node_total)
    vy_heatmap.set_array(current_vy_2D)
    ax_vy.set_title(                                                    
        "Velocity (y)"
    )

    current_bx_state = current_global_state[9 + total_node_count * 5: 6 * total_node_count + 9]
    current_bx_2D = current_bx_state.reshape(y_node_total, x_node_total)
    bx_heatmap.set_array(current_bx_2D)
    ax_bx.set_title(                                                    
        "B Field (x)"
    )

    current_by_state = current_global_state[9 + total_node_count * 6: 7 * total_node_count + 9]
    current_by_2D = current_by_state.reshape(y_node_total, x_node_total)
    by_heatmap.set_array(current_by_2D)
    ax_by.set_title(                                                    
        "B Field (y)"
    )
    
    return (div_B_rms_line, rho_heatmap, E_heatmap, p_heatmap, vx_heatmap, vy_heatmap, bx_heatmap, by_heatmap, )

ani = animation.FuncAnimation(
    fig = fig,
    func = update,
    frames = len(data_file_list),
    interval = 50,
    blit = False,
    repeat = True
)

plt.tight_layout()
plt.show()