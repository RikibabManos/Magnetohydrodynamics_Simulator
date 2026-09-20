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

    rho_max = np.max(properties["rho_2D"])
    rho_min = np.min(properties["rho_2D"])

    E_max = np.max(properties["E_2D"])
    E_min = np.min(properties["E_2D"])

    p_max = np.max(properties["p_2D"])
    p_min = np.min(properties["p_2D"])

    vx_max = np.max(properties["vx_2D"])
    vx_min = np.min(properties["vx_2D"])

    vy_max = np.max(properties["vy_2D"])
    vy_min = np.min(properties["vy_2D"])

    bx_max = np.max(properties["bx_2D"])
    bx_min = np.min(properties["bx_2D"])

    by_max = np.max(properties["by_2D"])
    by_min = np.min(properties["by_2D"])

    rho_max, rho_min = deal_with_visual_float_point_errors(rho_max, rho_min)
    E_max, E_min = deal_with_visual_float_point_errors(E_max, E_min)
    p_max, p_min = deal_with_visual_float_point_errors(p_max, p_min)
    vx_max, vx_min = deal_with_visual_float_point_errors(vx_max, vx_min)
    vy_max, vy_min = deal_with_visual_float_point_errors(vy_max, vy_min)
    bx_max, bx_min = deal_with_visual_float_point_errors(bx_max, bx_min)
    by_max, by_min = deal_with_visual_float_point_errors(by_max, by_min)

    extreme_values = {
        "rho": [rho_max, rho_min],
        "E": [E_max, E_min],
        "p": [p_max, p_min],
        "vx": [vx_max, vx_min],
        "vy": [vy_max, vy_min],
        "bx": [bx_max, bx_min],
        "by": [by_max, by_min]
    }

    return properties, extreme_values

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
