import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import glob
from display_functions import prepare_snapshot_data, deal_with_visual_float_point_errors

# make python read binary data file

data_file_list = glob.glob(r"C:\Codes\Magnetohydrodynamics_Simulator-RikibabManos-near-final-product\Magnetohydrodynamics_Simulator-RikibabManos-near-final-product/*.dat")
data_file_list.sort()                                                    # sorts files alphabetically
global_data_initial = np.fromfile(data_file_list[0], dtype = np.float64) # extract data from initial file to find header data and initial state

# extracting headers
x_node_count = int(global_data_initial[0])
y_node_count = int(global_data_initial[1])
ghost_cell_count = int(global_data_initial[2])
x_node_total = x_node_count + 2 * ghost_cell_count
y_node_total = y_node_count + 2 * ghost_cell_count
shape_data = [x_node_total, y_node_total, ghost_cell_count]
total_node_count = x_node_total * y_node_total
frame_rate = int(global_data_initial[3])
cell_width = global_data_initial[4]
cell_height = global_data_initial[5]
time_between_frames = global_data_initial[6]
root_mean_square_divergence_B = global_data_initial[7]
max_norm_divergence_B = global_data_initial[8]

## --- testing for core compression of sausage instability ---
#p_frame0 = np.fromfile("snapshot_00000.dat", dtype=np.float64)[9 + total_node_count * 2: 9 + total_node_count * 3].reshape((y_node_total, x_node_total))
#p_frameT = np.fromfile("snapshot_00030.dat", dtype=np.float64)[9 + total_node_count * 2: 9 + total_node_count * 3].reshape((y_node_total, x_node_total))
#
#i_mid = x_node_total // 2
#
#p_line_0 = p_frame0[:, i_mid]
#p_line_t = p_frameT[:, i_mid]
#
## finding the maximum compression point index
#max_compression = np.argmax(p_line_t) 
#
##print(f"p_neck(0): {p_line_0[max_compression]:.5f}")
##print(f"p_neck(t): {p_line_t[max_compression]:.5f}")
#compression_time = 30 * int(frame_rate) * time_between_frames
#print(f"Compression Ratio: {p_line_t[max_compression] / p_line_0[max_compression]:.2f}x, at t = {compression_time}s")

mhd_properties = prepare_snapshot_data(global_data_initial, shape_data)

rho_max = np.max(mhd_properties["rho_2D"])
rho_min = np.min(mhd_properties["rho_2D"])

E_max = np.max(mhd_properties["E_2D"])
E_min = np.min(mhd_properties["E_2D"])

p_max = np.max(mhd_properties["p_2D"])
p_min = np.min(mhd_properties["p_2D"])

vx_max = np.max(mhd_properties["vx_2D"])
vx_min = np.min(mhd_properties["vx_2D"])

vy_max = np.max(mhd_properties["vy_2D"])
vy_min = np.min(mhd_properties["vy_2D"])

bx_max = np.max(mhd_properties["bx_2D"])
bx_min = np.min(mhd_properties["bx_2D"])

by_max = np.max(mhd_properties["by_2D"])
by_min = np.min(mhd_properties["by_2D"])

rho_max, rho_min = deal_with_visual_float_point_errors(rho_max, rho_min)
E_max, E_min = deal_with_visual_float_point_errors(E_max, E_min)
p_max, p_min = deal_with_visual_float_point_errors(p_max, p_min)
vx_max, vx_min = deal_with_visual_float_point_errors(vx_max, vx_min)
vy_max, vy_min = deal_with_visual_float_point_errors(vy_max, vy_min)
bx_max, bx_min = deal_with_visual_float_point_errors(bx_max, bx_min)
by_max, by_min = deal_with_visual_float_point_errors(by_max, by_min)

# setup animation initial frame
fig = plt.figure(figsize = (18, 9), constrained_layout = True)
gs = gridspec.GridSpec(2, 4, width_ratios = [1, 1, 1, 1], wspace = 0.5, hspace = 0.5)

ax_rho = fig.add_subplot(gs[0, 0])
ax_div_B = fig.add_subplot(gs[1, 0])
#ax_p_max = fig.add_subplot(gs[1, 0])
ax_E = fig.add_subplot(gs[0, 1])
ax_p = fig.add_subplot(gs[1, 1])
ax_vx = fig.add_subplot(gs[0, 2])
ax_vy = fig.add_subplot(gs[1, 2])
ax_bx = fig.add_subplot(gs[0, 3])
ax_by = fig.add_subplot(gs[1, 3])

div_B_rms = [root_mean_square_divergence_B]
div_B_max = [max_norm_divergence_B]
div_B_rms_line, = ax_div_B.plot([], [], color = 'red', label = 'RMS Divergence of B')
div_B_max_line, = ax_div_B.plot([], [], color = 'blue', label = 'Max absolute Divergence of B')
p_max_history = [p_max]
#p_max_line, = ax_p_max.plot([], [], color = 'purple', label = 'Max pressure')
time_history = [0]
ax_div_B.legend()
#ax_p_max.legend()



rho_heatmap = ax_rho.imshow(
    mhd_properties["rho_2D"],
    cmap = 'magma',
    origin = 'lower',
    vmin = rho_min,
    vmax = rho_max,
    extent = (0, x_node_count, 0, y_node_count),
    interpolation = 'nearest'
)

E_heatmap = ax_E.imshow(
    mhd_properties["E_2D"],
    cmap = 'cividis',
    origin = 'lower',
    vmin = E_min,
    vmax = E_max,
    extent = (0, x_node_count, 0, y_node_count),
    interpolation = 'nearest'
)

p_heatmap = ax_p.imshow(
    mhd_properties["p_2D"],
    cmap = 'plasma',
    origin = 'lower',
    vmin = p_min,
    vmax = p_max,
    extent = (0, x_node_count, 0, y_node_count),
    interpolation = 'nearest'
)

vx_heatmap = ax_vx.imshow(
    mhd_properties["vx_2D"],
    cmap = 'magma',
    origin = 'lower',
    vmin = vx_min,
    vmax = vx_max,
    extent = (0, x_node_count, 0, y_node_count),
    interpolation = 'nearest'
)

vy_heatmap = ax_vy.imshow(
    mhd_properties["vy_2D"],
    cmap = 'magma',
    origin = 'lower', 
    vmin = vy_min,
    vmax = vy_max,
    extent = (0, x_node_count, 0, y_node_count),
    interpolation = 'nearest'
)

bx_heatmap = ax_bx.imshow(
    mhd_properties["bx_2D"],
    cmap = 'RdBu_r',
    origin = 'lower',
    vmin = bx_min,
    vmax = bx_max,
    extent = (0, x_node_count, 0, y_node_count),
    interpolation = 'nearest'
)

by_heatmap = ax_by.imshow(
    mhd_properties["by_2D"],
    cmap = 'coolwarm',
    origin = 'lower',
    vmin = by_min,
    vmax = by_max,
    extent = (0, x_node_count, 0, y_node_count),
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

    global time_history, div_B_rms, div_B_max, p_max_history

    # if animation repeated, reset the line histories
    if frame == 0:
        time_history.clear()
        div_B_rms.clear()
        div_B_max.clear()
        p_max_history.clear()

    current_global_state = np.fromfile(data_file_list[frame], dtype = np.float64)
    current_time = frame * int(frame_rate) * time_between_frames
    time_history.append(current_time)
    plt.suptitle(f"2D MHD Simulation Dashboard (Time: {current_time:.3f}s)", fontsize = 16, fontweight = 'bold')

    div_B_rms.append(current_global_state[7])
    div_B_max.append(current_global_state[8])
    div_B_rms_line.set_data(time_history, div_B_rms)
    div_B_max_line.set_data(time_history, div_B_max)
    current_max = max(div_B_max)
    ax_div_B.set_ylim(0, max(current_max * 1.1, 1e-15))
    ax_div_B.set_xlim(0, current_time if current_time > 0 else 1e-5)

    update_mhd_properties = prepare_snapshot_data(current_global_state, shape_data)

    #new_p = np.max(update_mhd_properties[p_2D))
    #p_max_history.append(new_p)
    #current_max_p = max(p_max_history)
    #p_max_line.set_data(time_history, p_max_history)
    #ax_p_max.set_ylim(0, max(current_max_p * 1.1, 1e-15))
    #ax_p_max.set_xlim(0, current_time if current_time > 0 else 1e-5)

    rho_heatmap.set_array(update_mhd_properties["rho_2D"])
    ax_rho.set_title(                                                    
        "Density"
    )

    E_heatmap.set_array(update_mhd_properties["E_2D"])
    ax_E.set_title(                                                    
        "Energy"
    )

    p_heatmap.set_array(update_mhd_properties["p_2D"])
    ax_p.set_title(                                                    
        "Thermal Pressure"
    )

    vx_heatmap.set_array(update_mhd_properties["vx_2D"])
    ax_vx.set_title(                                                    
        "Velocity (x)"
    )

    vy_heatmap.set_array(update_mhd_properties["vy_2D"])
    ax_vy.set_title(                                                    
        "Velocity (y)"
    )

    bx_heatmap.set_array(update_mhd_properties["bx_2D"])
    ax_bx.set_title(                                                    
        "B Field (x)"
    )

    by_heatmap.set_array(update_mhd_properties["by_2D"])
    ax_by.set_title(                                                    
        "B Field (y)"
    )

    ax_rho.set_xlim(0, x_node_count)
    ax_rho.set_ylim(0, y_node_count)
    ax_E.set_xlim(0, x_node_count)
    ax_E.set_ylim(0, y_node_count)
    ax_p.set_xlim(0, x_node_count)
    ax_p.set_ylim(0, y_node_count)
    ax_vx.set_xlim(0, x_node_count)
    ax_vy.set_ylim(0, y_node_count)
    ax_bx.set_xlim(0, x_node_count)
    ax_by.set_ylim(0, y_node_count)
    
    return (div_B_max_line, div_B_rms_line, rho_heatmap, E_heatmap, p_heatmap, vx_heatmap, vy_heatmap, bx_heatmap, by_heatmap, )

ani = animation.FuncAnimation(
    fig = fig,
    func = update,
    frames = len(data_file_list),
    interval = 50,
    blit = False,
    repeat = True
)

#ani.save(
#    "placeholder.gif",
#    writer = "pillow",
#    fps  = 1000,
#    dpi = 100
#)

plt.show()