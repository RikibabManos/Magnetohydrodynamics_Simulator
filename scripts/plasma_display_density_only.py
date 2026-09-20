import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import glob
from display_functions import prepare_snapshot_data

data_file_list = glob.glob(r"/home/true-sigma/GithubClones/Magnetohydrodynamics_Simulator/testData/OrszagTang_512x512_grid/*.dat")
data_file_list.sort()                                                    # sorts files alphabetically
global_data_initial = np.fromfile(data_file_list[0], dtype = np.float64) # extract data from initial file to find header data and initial state

# extracting headers
x_node_count = int(global_data_initial[0])
y_node_count = int(global_data_initial[1])
ghost_cell_count = int(global_data_initial[2])
x_node_total = x_node_count + 2 * ghost_cell_count
y_node_total = y_node_count + 2 * ghost_cell_count
shape_data = [x_node_total, y_node_total, ghost_cell_count]
total_node_count = x_node_total * y_node_total # does NOT account for any ghost cells
frame_rate = int(global_data_initial[3])
cell_width = global_data_initial[4]
cell_height = global_data_initial[5]
time_between_frames = global_data_initial[6]
root_mean_square_divergence_B = global_data_initial[7]
max_norm_divergence_B = global_data_initial[8]

mhd_properties, initial_extrema = prepare_snapshot_data(global_data_initial, shape_data)

# define coordinates on 2D grid for streamlines
if x_node_count == y_node_count:

    x = np.linspace(0, x_node_count * cell_width, x_node_count)
    y = np.linspace(0, y_node_count * cell_height, y_node_count)
    X, Y = np.meshgrid(x, y)

else:
    print("ERROR: X and Y node counts differ")

# setup animation initial frame
fig = plt.figure(figsize = (12, 6))
gs = gridspec.GridSpec(2, 2, height_ratios = [0.1, 1], wspace = 0.3, hspace = 0.5)
plt.subplots_adjust(top = 0.9, bottom = 0.08, left = 0.05, right = 0.95)

ax_title = fig.add_subplot(gs[0, :])
ax_title.axis('off')

ax_rho = fig.add_subplot(gs[1, 1])
ax_div_B = fig.add_subplot(gs[1, 0])

div_B_rms = [root_mean_square_divergence_B]
div_B_max = [max_norm_divergence_B]
div_B_rms_line, = ax_div_B.plot([], [], color = 'red', label = 'RMS Divergence of B')
div_B_max_line, = ax_div_B.plot([], [], color = 'blue', label = 'Max absolute Divergence of B')
ax_div_B.legend()

rho_heatmap = ax_rho.imshow(
    mhd_properties["rho_2D"],
    cmap = 'inferno',
    origin = 'lower',
    vmin = initial_extrema["rho"][1],
    vmax = initial_extrema["rho"][0],
    extent = (0, x_node_count * cell_width, 0, y_node_count * cell_height),
    interpolation = 'bilinear'
)

ax_rho.set_xlim(0, x_node_count * cell_width)
ax_rho.set_ylim(0, y_node_count * cell_height)
 
fig.colorbar(rho_heatmap, ax = ax_rho, label = 'Density')

# synchronized initial lists
time_history = []
div_B_rms = []
div_B_max = []

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

# if you wish to use full streamlines, uncomment the code below, and the corresponding commented code in the update func. Note that the performance is significantly worse than using quivers
## streamline underlayer
#ax_rho.streamplot(
#    X, Y, 
#    mhd_properties["bx_2D"], mhd_properties["by_2D"], 
#    color = 'black', 
#    density = 1.2, 
#    linewidth = 1.6, 
#    arrowsize = 0.8
#)
#
## stramline overlayer
#ax_rho.streamplot(
#    X, Y, 
#    mhd_properties["bx_2D"], mhd_properties["by_2D"], 
#    color = 'white', 
#    density = 1.2, 
#    linewidth = 0.8, 
#    arrowsize = 0.8
#)

stride_size = 32 # plots a quiver every 'stride' nodes
offset = cell_width / 2 # offset required so that over and underlayer quivers do not overlap on same coordinate, otherwise they will blend together only showing black

magnetic_streamlines_underlayer = ax_rho.quiver(
    X[::stride_size, ::stride_size] + offset,
    Y[::stride_size, ::stride_size],
    mhd_properties["bx_2D"][::stride_size, ::stride_size],
    mhd_properties["by_2D"][::stride_size, ::stride_size],
    color = 'black',
    scale = None,
    pivot = 'middle',
    width = 0.009,
    zorder = 2
)

magnetic_streamlines_overlayer = ax_rho.quiver(
    X[::stride_size, ::stride_size], 
    Y[::stride_size, ::stride_size],
    mhd_properties["bx_2D"][::stride_size, ::stride_size],
    mhd_properties["by_2D"][::stride_size, ::stride_size],
    color = 'white',
    scale = None,
    pivot = 'middle',
    width = 0.004,
    zorder = 3
)

ax_rho.set_xlim(0, x_node_count * cell_width)
ax_rho.set_ylim(0, y_node_count * cell_height)

def init():

    global total_initial_energy, mhd_properties

    rho_heatmap.set_array(mhd_properties["rho_2D"])
    main_title.set_text('2D MHD Simulation Dashboard (Time: 0.000s)')

    div_B_max = []
    div_B_rms = []
    div_B_rms_line.set_data([], [])
    div_B_max_line.set_data([], [])
    
    return (div_B_max_line, div_B_rms_line, rho_heatmap, )

def update(frame):

    global time_history, div_B_rms, div_B_max

    # reset the line histories to accomadate for animation repeats
    if frame == 0:

        time_history.clear()
        div_B_rms.clear()
        div_B_max.clear()

    current_global_state = np.fromfile(data_file_list[frame], dtype = np.float64)
    current_time = (frame + 1) * int(frame_rate) * time_between_frames
    main_title.set_text(f"2D MHD Simulation Dashboard (Time: {current_time:.3f}s)")
    
    time_history.append(current_time)
    div_B_rms.append(current_global_state[7])
    div_B_max.append(current_global_state[8])
    
    div_B_rms_line.set_data(time_history, div_B_rms)
    div_B_max_line.set_data(time_history, div_B_max)
    
    update_mhd_properties, _ = prepare_snapshot_data(current_global_state, shape_data)
    
    rho_heatmap.set_array(update_mhd_properties["rho_2D"])
    ax_rho.set_title("Density with B Field streamlines")

    magnetic_streamlines_underlayer.set_UVC(
        update_mhd_properties["bx_2D"][::stride_size, ::stride_size],
        update_mhd_properties["by_2D"][::stride_size, ::stride_size]
    )

    magnetic_streamlines_overlayer.set_UVC(
        update_mhd_properties["bx_2D"][::stride_size, ::stride_size],
        update_mhd_properties["by_2D"][::stride_size, ::stride_size]
    )

    # CORRESPONDING CODE FOR FULL STREAMLINES
    #for i in list(ax_rho.collections) + list(ax_rho.patches): 
    #    i.remove()
#
    ## streamline underlayer
    #ax_rho.streamplot(
    #    X, Y, 
    #    update_mhd_properties["bx_2D"], update_mhd_properties["by_2D"], 
    #    color = 'black', 
    #    density = 1.2, 
    #    linewidth = 1.6, 
    #    arrowsize = 0.8
    #)
#
    ## stramline overlayer
    #ax_rho.streamplot(
    #    X, Y, 
    #    update_mhd_properties["bx_2D"], update_mhd_properties["by_2D"], 
    #    color = 'white', 
    #    density = 1.2, 
    #    linewidth = 0.8, 
    #    arrowsize = 0.8
    #)

    return (div_B_rms_line, div_B_max_line, rho_heatmap, magnetic_streamlines_overlayer, magnetic_streamlines_underlayer, )

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
#    "placeholder1.gif",
#    writer = "pillow",
#    fps  = 1000,
#    dpi = 100
#)

plt.show()