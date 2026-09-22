import numpy as np

def prepare_snapshot_data(data_file, shape_info):
    """ Function to extract all header data from binary snapshot and prepare induvidual fields """

    nxt, nyt, ng = shape_info
    total_node_count = nxt * nyt
    data_file = data_file[9:] # remove header data

    data_file = data_file.reshape(-1, nyt, nxt)

    properties = {
        "rho_2D": data_file[0, ng: -ng, ng: -ng],
        "E_2D": data_file[1, ng: -ng, ng: -ng],
        "p_2D": data_file[2, ng: -ng, ng: -ng],
        "vx_2D": data_file[3, ng: -ng, ng: -ng],
        "vy_2D": data_file[4, ng: -ng, ng: -ng],
        "bx_2D": data_file[5, ng: -ng, ng: -ng],
        "by_2D": data_file[6, ng: -ng, ng: -ng]
    }

    return properties

def deal_with_visual_float_point_errors(max, min):
    tolerance = 5e-5

    if (max - min) < tolerance:
        midpoint = (max + min) / 2

        if abs(midpoint) < tolerance:
            max = 1e-4
            min = -1e-4
        else:
            max = midpoint + 1e-4
            min = midpoint - 1e-4

    return max, min

def get_global_extrema(whole_data_file_list, shape_info):

    rho_max_data = []
    rho_min_data = []
    E_max_data = []
    E_min_data = []
    p_max_data = []
    p_min_data = []
    vx_max_data = []
    vx_min_data = []
    vy_max_data = []
    vy_min_data = []
    bx_max_data = []
    bx_min_data = []
    by_max_data = []
    by_min_data = []
    
    for file in whole_data_file_list:
    
        current_data = np.fromfile(file, dtype = np.float64)
        current_properties = prepare_snapshot_data(current_data, shape_info)
    
        rho_max_data.append(np.max(current_properties["rho_2D"]))
        rho_min_data.append(np.min(current_properties["rho_2D"]))
        E_max_data.append(np.max(current_properties["E_2D"]))
        E_min_data.append(np.min(current_properties["E_2D"]))    
        p_max_data.append(np.max(current_properties["p_2D"]))
        p_min_data.append(np.min(current_properties["p_2D"]))
        vx_max_data.append(np.max(current_properties["vx_2D"]))
        vx_min_data.append(np.min(current_properties["vx_2D"]))
        vy_max_data.append(np.max(current_properties["vy_2D"]))
        vy_min_data.append(np.min(current_properties["vy_2D"]))
        bx_max_data.append(np.max(current_properties["bx_2D"]))
        bx_min_data.append(np.min(current_properties["bx_2D"]))
        by_max_data.append(np.max(current_properties["by_2D"]))
        by_min_data.append(np.min(current_properties["by_2D"]))
    
    global_rho_max = max(rho_max_data)
    global_rho_min = min(rho_min_data)
    global_E_max = max(E_max_data)
    global_E_min = min(E_min_data)
    global_p_max = max(p_max_data)
    global_p_min = min(p_min_data)
    global_vx_max = max(vx_max_data)
    global_vx_min = min(vx_min_data)
    global_vy_max = max(vy_max_data)
    global_vy_min = min(vy_min_data)
    global_bx_max = max(bx_max_data)
    global_bx_min = min(bx_min_data)
    global_by_max = max(by_max_data)
    global_by_min = min(by_min_data)
    
    global_rho_max, global_rho_min = deal_with_visual_float_point_errors(global_rho_max, global_rho_min)
    global_E_max, global_E_min = deal_with_visual_float_point_errors(global_E_max, global_E_min)
    global_p_max, global_p_min = deal_with_visual_float_point_errors(global_p_max, global_p_min)
    global_vx_max, global_vx_min = deal_with_visual_float_point_errors(global_vx_max, global_vx_min)
    global_vy_max, global_vy_min = deal_with_visual_float_point_errors(global_vy_max, global_vy_min)
    global_bx_max, global_bx_min = deal_with_visual_float_point_errors(global_bx_max, global_bx_min)
    global_by_max, global_by_min = deal_with_visual_float_point_errors(global_by_max, global_by_min)

    extrema = {
        "rho": [global_rho_max, global_rho_min],
        "E": [global_E_max, global_E_min],
        "p": [global_p_max, global_p_min],
        "vx": [global_vx_max, global_vx_min],
        "vy": [global_vy_max, global_vy_min],
        "bx": [global_bx_max, global_bx_min],
        "by": [global_by_max, global_by_min],
    }

    return extrema