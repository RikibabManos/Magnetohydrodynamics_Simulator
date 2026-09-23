import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import glob
from pathlib import Path
from display_functions import prepare_snapshot_data, get_global_extrema

script_dir = Path(__file__).parent
output_dir = script_dir.parent / "build/output" 
data_file_list = glob.glob(str(output_dir / "snapshot_*.dat"))
data_file_list.sort()
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
# note that a compression ratio > 1 is unlikely outside of low beta regime
#p_frame0 = np.fromfile("/home/true-sigma/GithubClones/Magnetohydrodynamics_Simulator/testData/sausageInstability_512x512_grid_lowBetaRegime/snapshot_00000.dat", dtype=np.float64)[9 + total_node_count * 2: 9 + total_node_count * 3].reshape((y_node_total, x_node_total))
#p_frameT = np.fromfile("/home/true-sigma/GithubClones/Magnetohydrodynamics_Simulator/testData/sausageInstability_512x512_grid_lowBetaRegime/snapshot_00040.dat", dtype=np.float64)[9 + total_node_count * 2: 9 + total_node_count * 3].reshape((y_node_total, x_node_total))
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
#compression_time = int(frame_rate) * time_between_frames * 40 # the integer number must be manually changed to mirror the second frame measured
#print(f"Compression Ratio: {p_line_t[max_compression] / p_line_0[max_compression]:.2f}x, at t = {compression_time}s")

mhd_properties = prepare_snapshot_data(global_data_initial, shape_data)

# setup animation initial frame
fig = plt.figure(figsize = (18, 9))
gs = gridspec.GridSpec(3, 4, height_ratios = [0.1, 1, 1], wspace = 0.5, hspace = 0.2)
plt.subplots_adjust(top = 0.9, bottom = 0.08, left = 0.05, right = 0.95)

ax_title = fig.add_subplot(gs[0, :]) # seperate title axis so figure title can be attributed here, easier with blit = True
ax_title.axis('off')

ax_rho = fig.add_subplot(gs[1, 0])
ax_div_B = fig.add_subplot(gs[2, 0])
ax_E = fig.add_subplot(gs[1, 1])
ax_p = fig.add_subplot(gs[2, 1])
ax_vx = fig.add_subplot(gs[1, 2])
ax_vy = fig.add_subplot(gs[2, 2])
ax_bx = fig.add_subplot(gs[1, 3])
ax_by = fig.add_subplot(gs[2, 3])

global_extrema = get_global_extrema(data_file_list, shape_data)

div_B_rms = [root_mean_square_divergence_B]
div_B_max = [max_norm_divergence_B]
div_B_rms_line, = ax_div_B.plot([], [], color = 'red', label = 'RMS Divergence of B')
div_B_max_line, = ax_div_B.plot([], [], color = 'blue', label = 'Max absolute Divergence of B')
p_max_history = [global_extrema["p"][0]]
time_history = [0]
ax_div_B.legend()

rho_heatmap = ax_rho.imshow(
    mhd_properties["rho_2D"],
    cmap = 'magma',
    origin = 'lower',
    vmin = global_extrema["rho"][1],
    vmax = global_extrema["rho"][0],
    extent = (0, x_node_count, 0, y_node_count),
    interpolation = 'nearest'
)

E_heatmap = ax_E.imshow(
    mhd_properties["E_2D"],
    cmap = 'cividis',
    origin = 'lower',
    vmin = global_extrema["E"][1],
    vmax = global_extrema["E"][0],
    extent = (0, x_node_count, 0, y_node_count),
    interpolation = 'nearest'
)

p_heatmap = ax_p.imshow(
    mhd_properties["p_2D"],
    cmap = 'plasma',
    origin = 'lower',
    vmin = global_extrema["p"][1],
    vmax = global_extrema["p"][0],
    extent = (0, x_node_count, 0, y_node_count),
    interpolation = 'nearest'
)

vx_heatmap = ax_vx.imshow(
    mhd_properties["vx_2D"],
    cmap = 'magma',
    origin = 'lower',
    vmin = global_extrema["vx"][1],
    vmax = global_extrema["vy"][0],
    extent = (0, x_node_count, 0, y_node_count),
    interpolation = 'nearest'
)

vy_heatmap = ax_vy.imshow(
    mhd_properties["vy_2D"],
    cmap = 'magma',
    origin = 'lower', 
    vmin = global_extrema["vy"][1],
    vmax = global_extrema["vy"][0],
    extent = (0, x_node_count, 0, y_node_count),
    interpolation = 'nearest'
)

bx_heatmap = ax_bx.imshow(
    mhd_properties["bx_2D"],
    cmap = 'RdBu_r',
    origin = 'lower',
    vmin = global_extrema["bx"][1],
    vmax = global_extrema["bx"][0],
    extent = (0, x_node_count, 0, y_node_count),
    interpolation = 'nearest'
)

by_heatmap = ax_by.imshow(
    mhd_properties["by_2D"],
    cmap = 'coolwarm',
    origin = 'lower',
    vmin = global_extrema["by"][1],
    vmax = global_extrema["by"][0],
    extent = (0, x_node_count, 0, y_node_count),
    interpolation = 'nearest'
)

ax_rho.set_xticks([]) 
ax_rho.set_yticks([]) 
ax_E.set_xticks([])
ax_E.set_yticks([]) 
ax_p.set_xticks([])
ax_p.set_yticks([]) 
ax_vx.set_xticks([])
ax_vx.set_yticks([]) 
ax_vy.set_xticks([]) 
ax_vy.set_yticks([]) 
ax_bx.set_xticks([]) 
ax_bx.set_yticks([]) 
ax_by.set_xticks([]) 
ax_by.set_yticks([]) 

rho_colourbar = fig.colorbar(rho_heatmap, ax = ax_rho, label = 'Density')
E_colourbar = fig.colorbar(E_heatmap, ax = ax_E, label = 'Energy')
p_colourbar = fig.colorbar(p_heatmap, ax = ax_p, label = 'Thermal Pressure')
vx_colourbar = fig.colorbar(vx_heatmap, ax = ax_vx, label = 'Velocity (x)')
vy_colourbar = fig.colorbar(vy_heatmap, ax = ax_vy, label = 'Velocity (y)')
bx_colourbar = fig.colorbar(bx_heatmap, ax = ax_bx, label = 'Magnetic Flux Density (x)')
by_colourbar = fig.colorbar(by_heatmap, ax = ax_by, label = 'Magnetic Flux Density (y)')

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


ax_rho.set_title("Density")
ax_E.set_title("Energy")
ax_p.set_title("Thermal Pressure")
ax_vx.set_title("Velocity (x)")
ax_vy.set_title("Velocity (y)")
ax_bx.set_title("B Field (x)")
ax_by.set_title("B Field (y)")

main_title = ax_title.text(
    0.5, 
    0.5, 
    '', 
    ha = 'center',
    va = 'center',
    fontsize = 16, 
    fontweight = 'bold',
    transform = ax_title.transAxes
)

total_sim_time = len(data_file_list) * int(frame_rate) * time_between_frames
ax_div_B.set_xlim(0, total_sim_time)
greatest_div_B = np.fromfile(data_file_list[-1], dtype = np.float64)[8] * 1.2
ax_div_B.set_ylim(0, max(greatest_div_B, 1e-15))

div_B_y_label = r'$\nabla \cdot B$ / $\text{grid_length}^{-1}$'
ax_div_B.set_ylabel(div_B_y_label)
ax_div_B.set_xlabel("Time Elapsed / s")

def init():

    global total_initial_energy, mhd_properties

    rho_heatmap.set_array(mhd_properties["rho_2D"])
    p_heatmap.set_array(mhd_properties["p_2D"])
    E_heatmap.set_array(mhd_properties["E_2D"])
    vx_heatmap.set_array(mhd_properties["vx_2D"])
    vy_heatmap.set_array(mhd_properties["vy_2D"])
    bx_heatmap.set_array(mhd_properties["bx_2D"])
    by_heatmap.set_array(mhd_properties["by_2D"])

    div_B_max = []
    div_B_rms = []
    div_B_rms_line.set_data([], [])
    div_B_max_line.set_data([], [])

    main_title.set_text('2D MHD Simulation Dashboard (Time: 0.000s)')
    
    total_initial_energy = np.sum(mhd_properties["E_2D"])
    print(f"Total initial energy: {total_initial_energy}")
    
    return (main_title, div_B_max_line, div_B_rms_line, rho_heatmap, E_heatmap, p_heatmap, vx_heatmap, vy_heatmap, bx_heatmap, by_heatmap, )

def update(frame):

    global time_history, div_B_rms, div_B_max, p_max_history, total_initial_energy

    # reset the line histories to accommodate for animation repeats
    if frame == 0:

        time_history.clear()
        div_B_rms.clear()
        div_B_max.clear()
        p_max_history.clear()
    
    current_global_state = np.fromfile(data_file_list[frame], dtype = np.float64)
    current_timestep = current_global_state[6]
    current_time = frame * int(frame_rate) * current_time_step
    time_history.append(current_time)
    main_title.set_text(f"2D MHD Simulation Dashboard (Time: {current_time:.3f}s)")

    div_B_rms.append(current_global_state[7])
    div_B_max.append(current_global_state[8])
    div_B_rms_line.set_data(time_history, div_B_rms)
    div_B_max_line.set_data(time_history, div_B_max)

    update_mhd_properties = prepare_snapshot_data(current_global_state, shape_data)
    rho_heatmap.set_array(update_mhd_properties["rho_2D"])
    E_heatmap.set_array(update_mhd_properties["E_2D"])
    p_heatmap.set_array(update_mhd_properties["p_2D"])
    vx_heatmap.set_array(update_mhd_properties["vx_2D"])
    vy_heatmap.set_array(update_mhd_properties["vy_2D"])
    bx_heatmap.set_array(update_mhd_properties["bx_2D"])
    by_heatmap.set_array(update_mhd_properties["by_2D"])

    if frame % 10 == 0: # output energy conservation, not too often to overload terminal

        total_current_energy = np.sum(update_mhd_properties["E_2D"])
        energy_tolerance = 1e-8
        
        if abs(total_current_energy - total_initial_energy) < energy_tolerance:        
            print(f"Total current energy: {total_current_energy}. Energy is currently conserved")
        else:
            print(f"Total current energy: {total_current_energy}. Energy is currently NOT conserved")
    
    return (main_title, div_B_max_line, div_B_rms_line, rho_heatmap, E_heatmap, p_heatmap, vx_heatmap, vy_heatmap, bx_heatmap, by_heatmap, )

ani = animation.FuncAnimation(
    fig = fig,
    func = update,
    init_func = init,
    frames = len(data_file_list),
    interval = 50,
    blit = True,
    repeat = True
)

#ani.save(
#    "placeholder.gif",
#    writer = "pillow",
#    fps  = 60,
#    dpi = 100
#)

plt.show()
