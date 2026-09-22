#pragma once
#include <iostream>
#include <fstream>
#include <iomanip>
#include "Grid.hpp"


struct divergence_checker_values {

    double root_mean_square_div;
    double max_norm_divergence;

};

divergence_checker_values calculate_current_div_B(const Grid& grid_test) { // function to calculate current max and norm divergence of B field, used to test if constrained transport method if working as intended

    double max_divergence_norm = 0.0;
    double root_mean_square_norm_divergence = 0.0;
    const int j_final = grid_test.ny;
    const int i_final = grid_test.nx;

    for (int j = 0; j < j_final; j++) {
        for (int i = 0; i < i_final; i++) {

            int current_physical_index = grid_test.indexC(grid_test.gi(i), grid_test.gj(j));
            int right_physical_index = grid_test.indexC(grid_test.gi(i + 1), grid_test.gj(j));
            int top_physical_index = grid_test.indexC(grid_test.gi(i), grid_test.gj(j + 1));
            
            double current_x_divergence = (grid_test.bxf[right_physical_index] - grid_test.bxf[current_physical_index]) / grid_test.cell_width;
            double current_y_divergence = (grid_test.byf[top_physical_index] - grid_test.byf[current_physical_index]) / grid_test.cell_height;
            double total_current_divergence = current_x_divergence + current_y_divergence;
            root_mean_square_norm_divergence += total_current_divergence * total_current_divergence;

            if (std::abs(total_current_divergence) > max_divergence_norm) {max_divergence_norm = std::abs(total_current_divergence);}

        }
    }
    
    root_mean_square_norm_divergence = sqrt(root_mean_square_norm_divergence / (i_final * j_final));
    return {root_mean_square_norm_divergence, max_divergence_norm};

}

double CFL_condition_checker(const Grid& grid_checker) {

    const double dx = grid_checker.cell_width;
    const double dy = grid_checker.cell_height;
    const double adiabatic_index = 5.0 / 3.0;
    const int total_y_nodes = grid_checker.ny;
    const int total_x_nodes = grid_checker.nx;
    const double resistivity_value = grid_checker.resistivity;
    double maximum_dt_resistive = INFINITY; // default if resisitivity == 0.0, large so never triggers if resistivity is 0.0

    if (resistivity_value != 0.0) {
        maximum_dt_resistive = std::min(dx, dy) * std::min(dx, dy) / (4.0 * resistivity_value); // for resistive CFL limit
    }

    double max_fluid_speed = 1e-8; // floor value to prevent time steps being too large or div by zero errors

    for (int j = 0; j < total_y_nodes; j++) {
        for (int i = 0; i < total_x_nodes; i++) {

            int current_idx = grid_checker.indexC(grid_checker.gi(i), grid_checker.gj(j)); 
            double current_density = std::max(grid_checker.rho[current_idx], 1e-8); // the density/ pressure floors prevent invalid timesteps due to pressure/ density calulations (specifically in regions of low density or vacuums)
            double current_pressure = std::max(grid_checker.p[current_idx], 1e-8);
            double current_vx = grid_checker.vx[current_idx];
            double current_vy = grid_checker.vy[current_idx];
            double current_bx = grid_checker.bxc[current_idx];
            double current_by = grid_checker.byc[current_idx];

            double speed_of_sound = adiabatic_index * current_pressure / current_density;
            double alfven_speed = (current_bx * current_bx + current_by * current_by) / current_density;
            double magnetosonic_speed = std::sqrt(speed_of_sound + alfven_speed);
            double fluid_speed = std::sqrt(current_vx * current_vx + current_vy * current_vy);
            double current_fastest_speed = magnetosonic_speed + fluid_speed;

            if (current_fastest_speed > max_fluid_speed) {max_fluid_speed = current_fastest_speed;}
        
        }
    }

    double maximum_dt_hyperbolic = 0.5 * std::min(dx, dy) / max_fluid_speed; // for advective CFL limit
    double time_interval_limit = std::min(maximum_dt_hyperbolic, maximum_dt_resistive);
    double current_optimal_time_step = time_interval_limit * 0.9; // give some leeway under limit

    return current_optimal_time_step;

}

void write_data_to_binary_file(Grid& grid_bin, int current_frame_index, int snapshot_rate, double time_step) {

    std::fstream binary_fout; 
    std::ostringstream ss;
    ss << "output/snapshot_" << std::setw(6) << std::setfill('0') << current_frame_index << ".dat";
    std::string filename = ss.str();
    binary_fout.open(filename, std::ios::out | std::ios::binary);

    if (!binary_fout.is_open()) {
        std::cout << "Binary file not accessed error" << '\n';
        return;
    }

    divergence_checker_values div_B_values = calculate_current_div_B(grid_bin);

    // adding header information to file
    double header[9] = {
        static_cast<double>(grid_bin.nx),
        static_cast<double>(grid_bin.ny),
        static_cast<double>(grid_bin.ng),
        static_cast<double>(snapshot_rate),
        grid_bin.cell_width,
        grid_bin.cell_height,
        time_step,
        div_B_values.root_mean_square_div,
        div_B_values.max_norm_divergence
    };

    binary_fout.write(reinterpret_cast<char*>(header), sizeof(header));

    // all data in global state written into a binary file sequentially, after 9 header values, INCLUDING ghost cells!
    binary_fout.write(reinterpret_cast<char*>(grid_bin.rho.data()), (grid_bin.rho.size() * sizeof(grid_bin.rho[0])));
    binary_fout.write(reinterpret_cast<char*>(grid_bin.E.data()),   (grid_bin.E.size()   * sizeof(grid_bin.E[0])));
    binary_fout.write(reinterpret_cast<char*>(grid_bin.p.data()),   (grid_bin.p.size()   * sizeof(grid_bin.p[0])));
    binary_fout.write(reinterpret_cast<char*>(grid_bin.vx.data()),  (grid_bin.vx.size()  * sizeof(grid_bin.vx[0])));
    binary_fout.write(reinterpret_cast<char*>(grid_bin.vy.data()),  (grid_bin.vy.size()  * sizeof(grid_bin.vy[0])));
    binary_fout.write(reinterpret_cast<char*>(grid_bin.bxc.data()), (grid_bin.bxc.size() * sizeof(grid_bin.bxc[0])));
    binary_fout.write(reinterpret_cast<char*>(grid_bin.byc.data()), (grid_bin.byc.size() * sizeof(grid_bin.byc[0])));

    binary_fout.close(); 
    
}
