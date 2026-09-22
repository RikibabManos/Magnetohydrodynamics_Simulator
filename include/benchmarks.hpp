#pragma once
#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include "Grid.hpp"
#include "MUSCL.hpp"
#include "HLL.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>

void orszag_tang_initial_conditions(Grid& grid_ot, integrator_reserved_memory& mag_potential_storage) { // displays the Orszag-Tang vortex benchmark test

    const int grid_x_node_num = grid_ot.nxt;
    const int grid_y_node_num = grid_ot.nyt;
    const int ghost_cell_num = grid_ot.ng;
    const double width_of_cell = grid_ot.cell_width;
    const double height_of_cell = grid_ot.cell_height;
    const int grid_node_total = grid_ot.nxt * grid_ot.nyt;

    std::vector<double>& Az = mag_potential_storage.magnetic_potential;

    // constant values
    const double rho =  25.0 / 9.0; // adiabatic index squared
    const double p   =  5.0 / 3.0;  // adiabatic index (monotomic gas)

    // the for loops below MUST be inclusive so that one extra node is populated with a vector potential value, this is so that in the following loop bx and by can be populated by reaching for an extra magnetic potential value, instead of using uninitialised data
    for (int j = 0; j <= grid_y_node_num; ++j) {
        for (int i = 0; i <= grid_x_node_num; ++i) {

            int index = grid_ot.indexC(i, j);

            // corner coordinates (bottom-left of cell i,j)
            double x_corner = (i - ghost_cell_num) * width_of_cell;
            double y_corner = (j - ghost_cell_num) * height_of_cell;

            // Orszag-Tang vector potential
            Az[index] = 0.5 * cos(2 * x_corner) + cos(y_corner);

        }
    }

    for (int j = 0; j < grid_y_node_num; ++j) {
        for (int i = 0; i < grid_x_node_num; ++i) {
            
            int idx = grid_ot.indexC(i, j);
            int top = grid_ot.indexC(i, j + 1);
            int right = grid_ot.indexC(i + 1, j);
        
            double x_coordinate_of_current_node = (i - ghost_cell_num + 0.5) * width_of_cell;
            double y_coordinate_of_current_node = (j - ghost_cell_num + 0.5) * height_of_cell;

            double vx = -sin(y_coordinate_of_current_node);
            double vy =  sin(x_coordinate_of_current_node);
            double bx =  (Az[top] - Az[idx]) / height_of_cell;
            double by = -(Az[right] - Az[idx]) / width_of_cell;
        
            PrimitiveState w{rho, vx, vy, p, bx, by};
            ConservedState u = ConservedState::fromPrimitive(w);
        
            grid_ot.rho[idx] = u.rho;
            grid_ot.mx[idx] = u.mx;
            grid_ot.my[idx] = u.my;
            grid_ot.E[idx] = u.E;
            grid_ot.bxf[idx] = u.bx;
            grid_ot.byf[idx] = u.by;
            grid_ot.p[idx] = p;
            grid_ot.vx[idx] = vx;
            grid_ot.vy[idx] = vy;
        
        }
    }
}

void alfven_wave_test_initial_values(Grid& grid_mw) { // initialises a pure Alfven wave propagating to the right (only by plot should render a moving animation under stock conditions)

    const int grid_x_node_num = grid_mw.nxt;
    const int grid_y_node_num = grid_mw.nyt;
    const int physical_x_node_num = grid_mw.nx;
    const int physical_y_node_num = grid_mw.ny;
    const int ghost_cell_num = grid_mw.ng;
    const double width_of_cell = grid_mw.cell_width;
    const double height_of_cell = grid_mw.cell_height;
    const int grid_node_total = grid_mw.nxt * grid_mw.nyt;
    constexpr double pi = 3.14159265359;

    const double amplitude_bx = 0.0;
    const double amplitude_by = 1e-3;
    const double total_physical_width = physical_x_node_num * width_of_cell;
    const double total_physical_height = physical_y_node_num * height_of_cell;
    const double kx = 2.0 * pi / total_physical_width;
    const double ky = 2.0 * pi / total_physical_height;
    const double rho = 1.0;
    const double vx = 0.0;
    const double p  = 1.0;
    const double bx = 1.0; 

    for (int j = 0; j < grid_y_node_num; ++j) {
        for (int i = 0; i < grid_x_node_num; ++i) {
            
            int idx = grid_mw.indexC(i, j);
            double x_coordinate_of_current_node = (i - ghost_cell_num + 0.5) * width_of_cell;
            double y_coordinate_of_current_node = (j - ghost_cell_num + 0.5) * height_of_cell;

            double by = amplitude_by * sin(kx * x_coordinate_of_current_node);
            double vy = - by / std::sqrt(rho);

            PrimitiveState w{rho, vx, vy, p, bx, by};
            ConservedState u = ConservedState::fromPrimitive(w);
        
            grid_mw.rho[idx] = u.rho;
            grid_mw.mx[idx] = u.mx;
            grid_mw.my[idx] = u.my;
            grid_mw.E[idx] = u.E;
            grid_mw.bxf[idx] = u.bx;
            grid_mw.byf[idx] = u.by;
            grid_mw.p[idx] = p;
            grid_mw.vx[idx] = vx;
            grid_mw.vy[idx] = vy;

        }
    }
}

void gaussian_density_pulse_test_initial_values(Grid& grid_gauss) { // displays a gausian pulse centred at the origin that travels diagonally upwards, looping round due to periodic boundry conditions. If resistivity set to be non zero, pulse will decay/ diffuse radially outwards over time.

    const int physical_x_node_num = grid_gauss.nx;
    const int physical_y_node_num = grid_gauss.ny;
    const int grid_x_node_num = grid_gauss.nxt;
    const int grid_y_node_num = grid_gauss.nyt;
    const int ghost_cell_num = grid_gauss.ng;
    const double width_of_cell = grid_gauss.cell_width;
    const double height_of_cell = grid_gauss.cell_height;
    const int grid_node_total = grid_gauss.nxt * grid_gauss.nyt;

    // setting up gaussian pulse size and location
    const double x_mid = 0.5 * physical_x_node_num * grid_gauss.cell_width;
    const double y_mid = 0.5 * physical_x_node_num * grid_gauss.cell_height;
    const double pulse_radius = 2.0;
    const double radius_sq = pulse_radius * pulse_radius;

    const double vx = 1.0;
    const double vy = 1.0;
    const double p  = 1.0;
    const double bx = 0.0;
    const double by = 0.0;
    
    for (int j = 0; j < grid_y_node_num; ++j) {
        for (int i = 0; i < grid_x_node_num; ++i) {
            
            int idx = grid_gauss.indexC(i, j);
        
            double x_relative_to_centre = (i - ghost_cell_num + 0.5) * width_of_cell;
            double y_relative_to_centre = (j - ghost_cell_num + 0.5) * height_of_cell;
            double dist_sq = (x_relative_to_centre - x_mid) * (x_relative_to_centre - x_mid) + (y_relative_to_centre - y_mid) * (y_relative_to_centre - y_mid);

            double rho = 1.0 + exp(- 5 * dist_sq / radius_sq);
        
            PrimitiveState w{rho, vx, vy, p, bx, by};
            ConservedState u = ConservedState::fromPrimitive(w);

            grid_gauss.rho[idx] = u.rho;
            grid_gauss.mx[idx] = u.mx;
            grid_gauss.my[idx] = u.my;
            grid_gauss.E[idx] = u.E;
            grid_gauss.bxf[idx] = u.bx;
            grid_gauss.byf[idx] = u.by;
            grid_gauss.p[idx] = p;
            grid_gauss.vx[idx] = vx;
            grid_gauss.vy[idx] = vy;
        }
    }
}

void acoustic_wave_test_initial_values(Grid& grid_ac) { // displays an acoustic 1D progressive wave travelling in the right direction 
    
    const int physical_x_node_num = grid_ac.nx;
    const int physical_y_node_num = grid_ac.ny;
    const int grid_x_node_num = grid_ac.nxt;
    const int grid_y_node_num = grid_ac.nyt;
    const int ghost_cell_num = grid_ac.ng;
    const int grid_node_total = grid_ac.nxt * grid_ac.nyt;
    const double width_of_cell = grid_ac.cell_width;
    const double height_of_cell = grid_ac.cell_height;
    const double total_physical_grid_width = width_of_cell * physical_x_node_num;
    const double total_physical_grid_height = height_of_cell * physical_y_node_num;
    constexpr double pi = 3.14159265359;
    constexpr double adiabatic_index = 5.0 / 3.0;

    const double amplitude_x = 1e-4;
    const double amplitude_y = 1e-3;
    const double rho_background = 1.0;
    const double p_background = 1.0 / adiabatic_index;
    const double bx = 0.0;
    const double by = 0.0;
    const double speed_of_sound = std::sqrt(adiabatic_index * p_background / rho_background);

    for (int j = 0; j < grid_y_node_num; ++j) {
        for (int i = 0; i < grid_x_node_num; ++i) {
            
            int idx = grid_ac.indexC(i, j);

            double x_position = (i - ghost_cell_num) * width_of_cell;
            double y_position = (j - ghost_cell_num) * height_of_cell;
            double x_sin = amplitude_x * sin(4 * pi * x_position / total_physical_grid_width);
            double y_sin = 0.0; 

            double rho = rho_background + x_sin + y_sin;
            double vx = (speed_of_sound / rho_background) * x_sin;
            double vy = (speed_of_sound / rho_background) * y_sin;
            double p = p_background + (speed_of_sound * speed_of_sound) * (x_sin + y_sin);

            PrimitiveState w{rho, vx, vy, p, bx, by};
            ConservedState u = ConservedState::fromPrimitive(w);
        
            grid_ac.rho[idx] = u.rho;
            grid_ac.mx[idx] = u.mx;
            grid_ac.my[idx] = u.my;
            grid_ac.E[idx] = u.E;
            grid_ac.bxf[idx] = u.bx;
            grid_ac.byf[idx] = u.by;
            grid_ac.p[idx] = p;
            grid_ac.vx[idx] = vx;
            grid_ac.vy[idx] = vy;

        }
    }
}

void sausage_instability_initial_conditions(Grid& grid_si) {

    const int grid_x_node_num = grid_si.nxt;
    const int grid_y_node_num = grid_si.nyt;
    const int physical_grid_height = grid_si.ny;
    const int ghost_cell_num = grid_si.ng;
    const double cell_width = grid_si.cell_width;
    const double cell_height = grid_si.cell_height;
    const double adiabatic_index = 5.0 / 3.0; 
    const double background_B = 5.0;              
    const double background_rho = 1.0;            
    const double background_p = 0.1;              
    constexpr double pi = 3.14159265359;
    const double domain_height = physical_grid_height * cell_height;

    const double pertubation_amplitude = 0.15;    
    const double pertubation_wavenumber = 2.0 * pi / domain_height;    
    const double radius_sq = 0.04; // plasma pinch channel radius squared

    const double x_mid = 0.5 * grid_x_node_num * cell_width;

    for (int j = 0; j < grid_y_node_num; ++j) {
        for (int i = 0; i < grid_x_node_num; ++i) {
            
            int idx = grid_si.indexC(i, j);

            // cell-centered spatial positions
            double x_position = (i - ghost_cell_num + 0.5) * cell_width;
            double y_position = (j - ghost_cell_num + 0.5) * cell_height;

            double r_dist = x_position - x_mid;

            // m = 0 perturbation modulation along the channel axis (y-direction)
            double perturbation = 1.0 - pertubation_amplitude * std::cos(pertubation_wavenumber * y_position);

            // background floor of 0.001 to prevent 0/0 division
            double rho = background_rho * std::exp(-(r_dist * r_dist) / (radius_sq * perturbation)) + 0.1;
            double p   = background_p   * std::exp(-(r_dist * r_dist) / (radius_sq * perturbation)) + 0.1;
            rho= std::max(rho, 1e-4);
            p = std::max(p, 1e-4);
            double vx  = 0.0;
            double vy  = 0.0;
            double bx  = 0.0;
            double by  = background_B * (r_dist / std::sqrt(radius_sq)) * std::exp(-(r_dist * r_dist) / radius_sq);

            PrimitiveState w{rho, vx, vy, p, bx, by};
            ConservedState u = ConservedState::fromPrimitive(w);

            grid_si.rho[idx] = u.rho;
            grid_si.mx[idx] = u.mx;
            grid_si.my[idx] = u.my;
            grid_si.E[idx] = u.E;
            grid_si.bxf[idx] = bx;
            grid_si.byf[idx] = by;
            grid_si.p[idx] = p;
            grid_si.vx[idx] = vx;
            grid_si.vy[idx] = vy;

        }
    }
}

void kink_instability_initial_conditions(Grid& grid_ki) {

    const int grid_x_node_num = grid_ki.nxt;
    const int grid_y_node_num = grid_ki.nyt;
    const int physical_grid_height = grid_ki.ny;
    const int ghost_cell_num = grid_ki.ng;
    const double cell_width = grid_ki.cell_width;
    const double cell_height = grid_ki.cell_height;
    const double background_B   = 1.0;        
    const double background_rho = 1.0;      
    const double background_p   = 1.0;     
    constexpr double pi = 3.14159265359;   
    
    const double velocity_pertubation_amplitude = 0.05; // 5%  
    const double sinusoidal_wavenumber = 2.0 * pi; 
    const double radius_sq = 0.04; // channel width

    const double x_mid = 0.5 * grid_x_node_num * cell_width;

    for (int j = 0; j < grid_y_node_num; ++j) {
        for (int i = 0; i < grid_x_node_num; ++i) {
            
            int idx = grid_ki.indexC(i, j);

            // cell-centered spatial positions
            double x_position = (i - ghost_cell_num + 0.5) * cell_width;
            double y_position = (j - ghost_cell_num + 0.5) * cell_height;

            double r_dist = x_position - x_mid;

            double envelope = std::exp(-(r_dist * r_dist) / radius_sq);
            double rho = background_rho * envelope + 0.1; 
            double p = background_p   * envelope + 0.1;

            // m = 1 transverse velocity seed to trigger column displacement/buckling
            double vx = velocity_pertubation_amplitude * std::sin(sinusoidal_wavenumber * y_position) * envelope;
            double vy = 0.0;
            double bx = 0.0;
            double by = background_B * envelope;

            PrimitiveState w{rho, vx, vy, p, bx, by};
            ConservedState u = ConservedState::fromPrimitive(w);

            grid_ki.rho[idx] = u.rho;
            grid_ki.mx[idx] = u.mx;
            grid_ki.my[idx] = u.my;
            grid_ki.E[idx] = u.E;
            grid_ki.bxf[idx] = bx;
            grid_ki.byf[idx] = by;
            grid_ki.p[idx] = p;
            grid_ki.vx[idx] = vx;
            grid_ki.vy[idx] = vy;

        }
    }
}

void decaying_magnetic_sine_wave_initial_conditions(Grid& grid_dmw, integrator_reserved_memory& vector_potential_storage) {

    const double cell_width = grid_dmw.cell_width;
    const double cell_height = grid_dmw.cell_height;
    const int grid_x_node_num = grid_dmw.nxt; 
    const int grid_y_node_num = grid_dmw.nyt; 
    const int ghost_node_num = grid_dmw.ng;
    const int physicial_x_node_num = grid_dmw.nx;
    const int physicial_y_node_num = grid_dmw.ny;
    const double physical_grid_width = physicial_x_node_num * cell_width;
    const double physical_grid_height = physicial_y_node_num * cell_height;
    constexpr double pi = 3.14159265359; 

    const double kx = 2 * pi / physical_grid_width;
    const double ky = 2 * pi / physical_grid_height;
    
    const double rho = 1.0;
    const double p = 1.0;
    const double amplitude_bx = 1.0;
    const double amplitude_by = 1.0;
    const double vx = 0.0;
    const double vy = 0.0;

    // derive face B fields from magnetic vector potential
    std::vector<double>& Az = vector_potential_storage.magnetic_potential;

    for (int j = 0; j <= grid_y_node_num; j++) {
        for (int i = 0; i <= grid_x_node_num; i++) {

            int index = grid_dmw.indexC(i, j);

            // coordinates for node corners
            double x_left   = (i - ghost_node_num) * cell_width;
            double y_bottom = (j - ghost_node_num) * cell_height;

            Az[index] = (1.0 / kx) * cos(kx * x_left) * sin(ky * y_bottom);

        }
    }

    for (int j = 0; j < grid_y_node_num; j++) {
        for (int i = 0; i < grid_x_node_num; i++) {

            int bottom_left_corner = grid_dmw.indexC(i, j);
            int bottom_right_corner = grid_dmw.indexC(i + 1, j);
            int top_left_corner = grid_dmw.indexC(i, j + 1);

            double bx = (Az[top_left_corner] - Az[bottom_left_corner]) / cell_height;
            double by = - (Az[bottom_right_corner] - Az[bottom_left_corner]) / cell_width;

            PrimitiveState w{rho, vx, vy, p, bx, by};
            ConservedState u = ConservedState::fromPrimitive(w);

            grid_dmw.rho[bottom_left_corner] = u.rho;
            grid_dmw.mx[bottom_left_corner]  = u.mx;
            grid_dmw.my[bottom_left_corner]  = u.my;
            grid_dmw.E[bottom_left_corner]   = u.E;
            grid_dmw.bxf[bottom_left_corner] = bx;
            grid_dmw.byf[bottom_left_corner] = by;
            grid_dmw.p[bottom_left_corner]   = p;
            grid_dmw.vx[bottom_left_corner]  = vx;
            grid_dmw.vy[bottom_left_corner]  = vy;

        }
    }
}

void test_selection(Grid& grid, integrator_reserved_memory& memory) {

    std::cout << "======================================= "<< "\n";
    std::cout << "       MHD SIMULATOR TEST SUITE         "<< "\n";
    std::cout << "======================================= "<< "\n";
    std::cout << "1. Orszag-Tang Vortex Benchmark" << "\n";
    std::cout << "2. Sausage Instability (m=0)" << "\n";
    std::cout << "3. Kink Instability (m=1)" << "\n";
    std::cout << "4. Decaying Magnetic Sine Wave" << "\n";
    std::cout << "5. Alfvén Wave Test" << "\n";
    std::cout << "6. Gaussian Density Pulse" << "\n";
    std::cout << "7. Acoustic Wave Test" << "\n";
    std::cout << "---------------------------------------" << "\n";
    std::cout << "Enter test choice (1-7): ";

    int choice;
    std::cin >> choice;

    switch (choice) {
        
        case 1: orszag_tang_initial_conditions(grid, memory); break;
        case 2: sausage_instability_initial_conditions(grid); break;
        case 3: kink_instability_initial_conditions(grid); break;
        case 4: decaying_magnetic_sine_wave_initial_conditions(grid, memory); break;
        case 5: alfven_wave_test_initial_values(grid); break;
        case 6: gaussian_density_pulse_test_initial_values(grid); break;
        case 7: acoustic_wave_test_initial_values(grid); break;
        default:
            std::cout << "Invalid selection! Running Orszag-Tang by default.\n";
            orszag_tang_initial_conditions(grid, memory);
            break;

    }
}
