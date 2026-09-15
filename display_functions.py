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
        "by_2D": data_file[6, ng: -ng, ng: -ng],
    }

    return properties

def deal_with_visual_float_point_errors(max, min):
    tolerance = 5e-5

    if (max - min) < tolerance:
        midpoint = (max + min) / 2

        if abs(midpoint) < tolerance:
            max = 0.01
            min = -0.01
        else:
            max = midpoint + 0.01
            min = midpoint - 0.01

    return max, min
