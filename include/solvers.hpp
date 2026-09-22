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

// file contains helper functions and usful structs for integrator_main.cpp

struct global_state_dot { // structure that holds addresses of all the derivative values of important variables, specificaly used to transfer arrays from one reserve to another 

    std::vector<double>* rho_dot = nullptr; // density derivative
    std::vector<double>* m_dot_x = nullptr; // momentum density derivative in x direction
    std::vector<double>* m_dot_y = nullptr; // momentum density derivative in y direction
    std::vector<double>* b_dot_x = nullptr; // magnetic field strength derivative in x direction
    std::vector<double>* b_dot_y = nullptr; // magnetic field strength derivative in y direction
    std::vector<double>* E_dot = nullptr;   // total energy density derivative

    global_state_dot() = default;

    // constructor allocate addresses to pointers
    global_state_dot(std::vector<double>& rho, std::vector<double>& m_x, std::vector<double>& m_y, std::vector<double>& b_x, std::vector<double>& b_y, std::vector<double>& E)
    :rho_dot(&rho),
     m_dot_x(&m_x),
     m_dot_y(&m_y),
     b_dot_x(&b_x),
     b_dot_y(&b_y),
     E_dot(&E)
    {}
};

struct IMEX_Butcher_Tableus{ // struct that holds the Butcher Tableu coefficients for use in the IMEX integrator, tableus based off the ARS(2,2,2) scheme

    const double gamma = (2.0 - sqrt(2.0)) / 2.0; 
    const double delta = 1.0 - (1.0 / (2.0 * gamma));

    std::vector<double> A_ex = { // explicit Butcher Table (note in collapsed 1D form)

        0.0, 0.0, 0.0,
        gamma, 0.0, 0.0,
        delta, 1.0 - delta, 0.0

    };

    std::vector<double> c_ex = {0.0, gamma, 1.0}; // time steps for explicit part
    std::vector<double> b_ex = {delta, 1.0 - delta, 0.0}; // final weight of each explicit step

    std::vector<double> A_im = { // implicit Butcher Table (note in collapsed 1D form)

        0.0, 0.0, 0.0,
        0.0, gamma, 0.0,
        0.0, 1.0 - gamma, gamma

    };

    std::vector<double> c_im = {0.0, gamma, 1.0}; // time steps for implicit part
    std::vector<double> b_im = {0.0, 1.0 - gamma, gamma}; // final weight of each implicit step

};

struct integrator_reserved_memory{ // 'scratchpad', pre-allocates memory on stack once for IMEX which can be written on/ cleared each cycle 

    // f1, f2 hold the non-'stiff' derivative rates for the respective mini step for the explicit integrator part (no f3 needed as no 3rd column dependance; see explicit Butcher tableu)
    // g1, g2, g3 hold the 'stiff' derivative rates for each implicit mini step
    // 1 --> 3 beacuse ARS(2,2,2) has 3 steps
    // rhs holds final weighted sum of all g and f lists, to be used in the implicit integrator calculation

    // mass density
    std::vector<double> f1_rho;
    std::vector<double> f2_rho;    
    std::vector<double> g1_rho;
    std::vector<double> g2_rho;
    std::vector<double> g3_rho;

    // momentum density in x direction
    std::vector<double> f1_m_x;
    std::vector<double> f2_m_x;
    std::vector<double> g1_m_x;
    std::vector<double> g2_m_x;
    std::vector<double> g3_m_x;
 
    // momentum density in y direction
    std::vector<double> f1_m_y;
    std::vector<double> f2_m_y;
    std::vector<double> g1_m_y;
    std::vector<double> g2_m_y;    
    std::vector<double> g3_m_y;    

    // total energy density 
    std::vector<double> f1_E;
    std::vector<double> f2_E;
    std::vector<double> g1_E;   
    std::vector<double> g2_E;
    std::vector<double> g3_E;

    // face centred B field in x direction
    std::vector<double> f1_b_x_f;
    std::vector<double> f2_b_x_f;
    std::vector<double> g1_b_x_f;
    std::vector<double> g2_b_x_f;    
    std::vector<double> g3_b_x_f;

    // face centred B field in y direction
    std::vector<double> f1_b_y_f;
    std::vector<double> f2_b_y_f;
    std::vector<double> g1_b_y_f;
    std::vector<double> g2_b_y_f;
    std::vector<double> g3_b_y_f;

    // --- flux values for each direction ---

    // mass density fluxs
    std::vector<double> rho_flux_horizontal;
    std::vector<double> rho_flux_vertical;

    // momentum density flux in x direction
    std::vector<double> m_x_flux_horizontal;     
    std::vector<double> m_x_flux_vertical;

    // momentum density flux in y direction
    std::vector<double> m_y_flux_horizontal;
    std::vector<double> m_y_flux_vertical;

    // total Energy flux
    std::vector<double> E_flux_horizontal;    
    std::vector<double> E_flux_vertical;  
    
    // B field flux in x direction
    std::vector<double> b_x_flux_horizontal;
    std::vector<double> b_x_flux_vertical;

    // B field flux in y direction
    std::vector<double> b_y_flux_horizontal;
    std::vector<double> b_y_flux_vertical;

    // stiff values
    std::vector<double> b_x_laplacian;
    std::vector<double> b_y_laplacian;

    // --- concerning implicit terms/ solving ---

    // matrix storage (Ax = b)
    Eigen::SparseMatrix<double> A_matrix;

    // no ghost padding vectors
    std::vector<double> b_x_f_solved_1;
    std::vector<double> b_y_f_solved_1;
    std::vector<double> b_x_f_solved_2;
    std::vector<double> b_y_f_solved_2;

    // for electric field
    std::vector<double> electric_field_at_corners;
    std::vector<double> Ez_corner_no_padding;

    std::vector<double> magnetic_potential;

    integrator_reserved_memory(int cell_total, int cell_total_no_ghost, int magnetic_potential_size) // constructor which allocates required memory on heap once based off number of nodes
    
    :
     f1_rho(cell_total, 0.0), f2_rho(cell_total, 0.0), g1_rho(cell_total, 0.0), g2_rho(cell_total, 0.0), g3_rho(cell_total, 0.0),
     f1_m_x(cell_total, 0.0), f2_m_x(cell_total, 0.0), g1_m_x(cell_total, 0.0), g2_m_x(cell_total, 0.0), g3_m_x(cell_total, 0.0),  
     f1_m_y(cell_total, 0.0), f2_m_y(cell_total, 0.0), g1_m_y(cell_total, 0.0), g2_m_y(cell_total, 0.0), g3_m_y(cell_total, 0.0),
     f1_E(cell_total, 0.0), f2_E(cell_total, 0.0), g1_E(cell_total, 0.0), g2_E(cell_total, 0.0), g3_E(cell_total, 0.0),
     f1_b_x_f(cell_total, 0.0), f2_b_x_f(cell_total, 0.0), g1_b_x_f(cell_total, 0.0), g2_b_x_f(cell_total, 0.0), g3_b_x_f(cell_total, 0.0),
     f1_b_y_f(cell_total, 0.0), f2_b_y_f(cell_total, 0.0), g1_b_y_f(cell_total, 0.0), g2_b_y_f(cell_total, 0.0), g3_b_y_f(cell_total, 0.0),
     
     rho_flux_horizontal(cell_total, 0.0), rho_flux_vertical(cell_total, 0.0), 
     m_x_flux_horizontal(cell_total, 0.0), m_x_flux_vertical(cell_total, 0.0),
     m_y_flux_horizontal(cell_total, 0.0), m_y_flux_vertical(cell_total, 0.0),
     E_flux_horizontal(cell_total, 0.0), E_flux_vertical(cell_total, 0.0),
     b_x_flux_horizontal(cell_total, 0.0), b_x_flux_vertical(cell_total, 0.0),
     b_y_flux_horizontal(cell_total, 0.0), b_y_flux_vertical(cell_total, 0.0),

     b_x_laplacian(cell_total, 0.0), b_y_laplacian(cell_total, 0.0),

     b_x_f_solved_1(cell_total, 0.0), b_y_f_solved_1(cell_total, 0.0),
     b_x_f_solved_2(cell_total, 0.0), b_y_f_solved_2(cell_total, 0.0),

     A_matrix(cell_total_no_ghost, cell_total_no_ghost),

     electric_field_at_corners(cell_total, 0.0), Ez_corner_no_padding(cell_total_no_ghost, 0.0),

     magnetic_potential(magnetic_potential_size, 0.0)
     
    {}
};

inline double get_partial_derivative(double right_top_quantity, double left_bottom_quantity, double length) { // helper function to calculate the partial derivative of a quantity based off input values
    // right_top_quantity refers to either the right or top faces of a cell, left_bottom_quantity refers to left or bottom
    
    return (right_top_quantity - left_bottom_quantity) / length;

}

inline void get_intermediary_grid(Grid& grid_functions, std::vector<double>& grid_property, const std::vector<double>& property_change, double scalar){ // adds the change in a property (multiplied by a scalar) to the current property value (property of the simulator held in the grid struct e.g. rho)
    // lists must be equal length (accounted for in integrator_reserve_memory struct)
    int list_length = grid_property.size();

    for (int j = 0; j < grid_functions.ny; j++){
        for (int i = 0; i < grid_functions.nx; i++){

            int current_physical_index = grid_functions.indexC(grid_functions.gi(i), grid_functions.gj(j));
            grid_property[current_physical_index] += property_change[current_physical_index] * scalar;

        }
    }
}

inline void generate_cell_properties(PrimitiveState& left_face, PrimitiveState& right_face, PrimitiveState& top_face, PrimitiveState& bottom_face, Grid& grid_cell_properties, ReconstructedValues& reconstructed_val, int i_value, int j_value) { // assigns PrimitiveState structs with properties at each face

            int current_cell_index = grid_cell_properties.indexC(grid_cell_properties.gi(i_value), grid_cell_properties.gj(j_value));
            int right_cell_index = grid_cell_properties.indexC(grid_cell_properties.gi(i_value + 1), grid_cell_properties.gj(j_value));
            int left_cell_index = grid_cell_properties.indexC(grid_cell_properties.gi(i_value - 1), grid_cell_properties.gj(j_value));
            int top_cell_index = grid_cell_properties.indexC(grid_cell_properties.gi(i_value), grid_cell_properties.gj(j_value + 1));
            int bottom_cell_index = grid_cell_properties.indexC(grid_cell_properties.gi(i_value), grid_cell_properties.gj(j_value - 1));

            left_face.rho = reconstructed_val.rhoXFaceR[current_cell_index];
            left_face.vx = reconstructed_val.vxXFaceR[current_cell_index];
            left_face.vy = reconstructed_val.vyXFaceR[current_cell_index];
            left_face.p = reconstructed_val.pXFaceR[current_cell_index];
            left_face.bx = grid_cell_properties.bxf[right_cell_index];
            left_face.by = reconstructed_val.byXFaceR[current_cell_index];

            right_face.rho = reconstructed_val.rhoXFaceL[right_cell_index];
            right_face.vx = reconstructed_val.vxXFaceL[right_cell_index];
            right_face.vy = reconstructed_val.vyXFaceL[right_cell_index];
            right_face.p = reconstructed_val.pXFaceL[right_cell_index];
            right_face.bx = grid_cell_properties.bxf[right_cell_index];
            right_face.by = reconstructed_val.byXFaceL[right_cell_index];
            
            top_face.rho = reconstructed_val.rhoYFaceL[top_cell_index];
            top_face.vx = reconstructed_val.vxYFaceL[top_cell_index];
            top_face.vy = reconstructed_val.vyYFaceL[top_cell_index];
            top_face.p = reconstructed_val.pYFaceL[top_cell_index];
            top_face.bx = reconstructed_val.bxYFaceL[top_cell_index];
            top_face.by = grid_cell_properties.byf[top_cell_index];

            bottom_face.rho = reconstructed_val.rhoYFaceR[current_cell_index];
            bottom_face.vx = reconstructed_val.vxYFaceR[current_cell_index];
            bottom_face.vy = reconstructed_val.vyYFaceR[current_cell_index];
            bottom_face.p = reconstructed_val.pYFaceR[current_cell_index];
            bottom_face.bx = reconstructed_val.bxYFaceR[current_cell_index];
            bottom_face.by = grid_cell_properties.byf[top_cell_index];
}

void time_derivatives(const integrator_reserved_memory& memory,  const Grid& grid_div, global_state_dot& dU_explicit_rhs_memory) { // calculates the time derivatives of each quantity 
    
    std::vector<double>& rho_derivative = (*dU_explicit_rhs_memory.rho_dot);
    std::vector<double>& mx_derivative = (*dU_explicit_rhs_memory.m_dot_x);
    std::vector<double>& my_derivative = (*dU_explicit_rhs_memory.m_dot_y);
    std::vector<double>& E_derivative = (*dU_explicit_rhs_memory.E_dot);
    std::vector<double>& bxf_derivative = (*dU_explicit_rhs_memory.b_dot_x);
    std::vector<double>& byf_derivative = (*dU_explicit_rhs_memory.b_dot_y);

    const int total_x_nodes = grid_div.nx;
    const int total_y_nodes = grid_div.ny;
    const double cell_width = grid_div.cell_width;
    const double cell_height = grid_div.cell_height; 

    for (int j = 0; j < total_y_nodes; j++) {
        for (int i = 0; i < total_x_nodes; i++) {
            
            int current_node_index = grid_div.indexC(grid_div.gi(i), grid_div.gj(j));
            int left_node_index = grid_div.indexC(grid_div.gi(i - 1), grid_div.gj(j));
            int bottom_node_index = grid_div.indexC(grid_div.gi(i), grid_div.gj(j - 1));
            int right_node_index = grid_div.indexC(grid_div.gi(i + 1), grid_div.gj(j));
            int top_node_index = grid_div.indexC(grid_div.gi(i), grid_div.gj(j + 1));

            rho_derivative[current_node_index] = -( get_partial_derivative(memory.rho_flux_horizontal[current_node_index], memory.rho_flux_horizontal[left_node_index], cell_width) + get_partial_derivative(memory.rho_flux_vertical[current_node_index], memory.rho_flux_vertical[bottom_node_index], cell_height) );
            E_derivative[current_node_index] = -( get_partial_derivative(memory.E_flux_horizontal[current_node_index], memory.E_flux_horizontal[left_node_index], cell_width) + get_partial_derivative(memory.E_flux_vertical[current_node_index], memory.E_flux_vertical[bottom_node_index], cell_height) );
            mx_derivative[current_node_index] = -( get_partial_derivative(memory.m_x_flux_horizontal[current_node_index], memory.m_x_flux_horizontal[left_node_index], cell_width) + get_partial_derivative(memory.m_x_flux_vertical[current_node_index], memory.m_x_flux_vertical[bottom_node_index], cell_height) );
            my_derivative[current_node_index] = -( get_partial_derivative(memory.m_y_flux_horizontal[current_node_index], memory.m_y_flux_horizontal[left_node_index], cell_width) + get_partial_derivative(memory.m_y_flux_vertical[current_node_index], memory.m_y_flux_vertical[bottom_node_index], cell_height) );

            // force exact periodic wrapping to bypass uninitialized ghost cell memory
            int i_right = (i + 1) % total_x_nodes;
            int j_top = (j + 1) % total_y_nodes;
            int right_node_index2 = grid_div.indexC(grid_div.gi(i_right), grid_div.gj(j));
            int top_node_index2 = grid_div.indexC(grid_div.gi(i), grid_div.gj(j_top));

            bxf_derivative[current_node_index] = -(grid_div.Ez[top_node_index2] - grid_div.Ez[current_node_index]) / cell_height;
            byf_derivative[current_node_index] =  (grid_div.Ez[right_node_index2] - grid_div.Ez[current_node_index]) / cell_width;

        }
    }
}

void cell_corner_electric_fields(integrator_reserved_memory& B_memory, Grid& grid_B) { // function that calculates the explicit rhs terms of the time derivative of both B field components

    const int total_x_nodes = grid_B.nx;
    const int total_y_nodes = grid_B.ny;
    const int j_index_start = -1;
    const int j_index_end = total_y_nodes + 1;
    const int i_index_start = -1;
    const int i_index_end = total_x_nodes + 1;

    for (int j = j_index_start; j < j_index_end; j++){    
        for (int i = i_index_start; i < i_index_end; i++){
            
            int current_cell_index = grid_B.indexC(grid_B.gi(i), grid_B.gj(j));
            int left_cell_index = grid_B.indexC(grid_B.gi(i - 1), grid_B.gj(j));
            int bottom_cell_index = grid_B.indexC(grid_B.gi(i), grid_B.gj(j - 1)); 
            int bottom_left_cell_index = grid_B.indexC(grid_B.gi(i - 1), grid_B.gj(j - 1));

            // averaging the face electric fields (z-components) to give cell corner values, note that average is taken using the flux along each of the four faces adjacent to a corner
            // setup using these specific faces (below) as HLL defines a node as owning right and top face (against usual convention)

            double electric_field_north = -B_memory.b_y_flux_horizontal[left_cell_index];        
            double electric_field_south = -B_memory.b_y_flux_horizontal[bottom_left_cell_index];
            double electric_field_east = B_memory.b_x_flux_vertical[bottom_cell_index];          
            double electric_field_west = B_memory.b_x_flux_vertical[bottom_left_cell_index];     

            // direction of velocity is required for Upwind Constrained Transport Method

            double vx_corner = 0.25 * (grid_B.vx[current_cell_index] + grid_B.vx[left_cell_index] + grid_B.vx[bottom_cell_index] + grid_B.vx[bottom_left_cell_index]);
            double vy_corner = 0.25 * (grid_B.vy[current_cell_index] + grid_B.vy[left_cell_index] + grid_B.vy[bottom_cell_index] + grid_B.vy[bottom_left_cell_index]);
            double Ez_y;
            double Ez_x;

            if (abs(vy_corner) < 1e-12) {
                Ez_y = 0.5 * (electric_field_south + electric_field_north);
            }
            else {
                Ez_y = (vy_corner > 0.0) ? electric_field_south : electric_field_north; // taking North direction as positive
            }

            if (abs(vx_corner) < 1e-12) {
                Ez_x = 0.5 * (electric_field_east + electric_field_west);
            }
            else {
                Ez_x = (vx_corner > 0.0) ? electric_field_west : electric_field_east;   // taking East direction as positive
            }

            double electric_field = 0.5 * (Ez_x + Ez_y);
            B_memory.electric_field_at_corners[current_cell_index] = electric_field;
            grid_B.Ez[current_cell_index] = electric_field;
            
        }
    }
}

void flux_population(Grid& grid_flux, ReconstructedValues& RV, integrator_reserved_memory& reserve_flux) { // populates each node with its flux values and stores in designated integrator_memory_reserve vector
    
    // define PrimitiveState structs as FluxVector reuired those as inputs
    PrimitiveState left_face{};
    PrimitiveState right_face{};
    PrimitiveState top_face{};
    PrimitiveState bottom_face{};

    const int total_x_nodes = grid_flux.nx;
    const int total_y_nodes = grid_flux.ny;
    const int j_index_start = -1;
    const int j_index_end = total_y_nodes + 1;
    const int i_index_start = -1;
    const int i_index_end = total_x_nodes + 1;

    for (int j = j_index_start; j < j_index_end; j++){    
        for (int i = i_index_start; i < i_index_end; i++){

            int current_cell_index = grid_flux.indexC(grid_flux.gi(i), grid_flux.gj(j));
            int left_cell_index = grid_flux.indexC(grid_flux.gi(i - 1), grid_flux.gj(j));
            int bottom_cell_index = grid_flux.indexC(grid_flux.gi(i), grid_flux.gj(j - 1));     

            generate_cell_properties(left_face, right_face, top_face, bottom_face, grid_flux, RV, i, j);
            
            FluxVector x_direction_flux = computeHLLFluxX(left_face, right_face);
            FluxVector y_direction_flux = computeHLLFluxY(bottom_face, top_face);
            
            reserve_flux.rho_flux_horizontal[current_cell_index] = x_direction_flux.rho; 
            reserve_flux.m_x_flux_horizontal[current_cell_index] = x_direction_flux.mx; 
            reserve_flux.m_y_flux_horizontal[current_cell_index] = x_direction_flux.my; 
            reserve_flux.E_flux_horizontal[current_cell_index] = x_direction_flux.E; 
            reserve_flux.b_x_flux_horizontal[current_cell_index] = x_direction_flux.bx; 
            reserve_flux.b_y_flux_horizontal[current_cell_index] = x_direction_flux.by; 
            
            reserve_flux.rho_flux_vertical[current_cell_index] = y_direction_flux.rho; 
            reserve_flux.m_x_flux_vertical[current_cell_index] = y_direction_flux.mx; 
            reserve_flux.m_y_flux_vertical[current_cell_index] = y_direction_flux.my; 
            reserve_flux.E_flux_vertical[current_cell_index] = y_direction_flux.E; 
            reserve_flux.b_x_flux_vertical[current_cell_index] = y_direction_flux.bx; 
            reserve_flux.b_y_flux_vertical[current_cell_index] = y_direction_flux.by; 
            
        }
    }
}

void explicit_rhs(Grid& grid_ex, global_state_dot& dU_explicit, ReconstructedValues& MUSCL_output, integrator_reserved_memory& reserve) { // explicit half of IMEX integrator, returns a partial global_state_dot struct of only non-'stiff' variables

    grid_ex.fillGhostPeriodic();                               
    grid_ex.computePrimitives();                            
    MUSCLReconstructAll(grid_ex, MUSCL_output);           
    flux_population(grid_ex, MUSCL_output, reserve);      
    cell_corner_electric_fields(reserve, grid_ex); 
    time_derivatives(reserve, grid_ex, dU_explicit);       

}

void generate_A_matrix(const Grid& grid_solve, integrator_reserved_memory& matrix_store, const double alpha) { // builds a matrix representing the discretised Laplacian operator in 2D, using a 5-point finite-difference stencil

    const int x_size = grid_solve.nx;
    const int y_size = grid_solve.ny;
    const int total_grid_size = x_size * y_size;
    const double x_coeff = alpha / (grid_solve.cell_width * grid_solve.cell_width);
    const double y_coeff = alpha / (grid_solve.cell_height * grid_solve.cell_height);

    std::vector<Eigen::Triplet<double>> matrix_values;
    matrix_values.reserve(total_grid_size * 5); // maximum of 5 non-zero values per row, 

    for (int j = 0; j < y_size; j++) {
        for (int i = 0; i < x_size; i++) {

            int current_index = j * x_size + i;

            if (j == 0) {matrix_values.emplace_back(current_index, (y_size - 1) * x_size + i, - y_coeff);} // bottom node
            else {matrix_values.emplace_back(current_index, current_index - x_size, - y_coeff);}

            if (j == y_size - 1) {matrix_values.emplace_back(current_index, i, - y_coeff);} // top node
            else {matrix_values.emplace_back(current_index, current_index + x_size, - y_coeff);}

            if (i == 0) {matrix_values.emplace_back(current_index, current_index + x_size - 1, - x_coeff);} // left node
            else {matrix_values.emplace_back(current_index, current_index - 1, - x_coeff);}

            if (i == x_size - 1) {matrix_values.emplace_back(current_index, current_index - x_size + 1, - x_coeff);} // right node
            else {matrix_values.emplace_back(current_index, current_index + 1, - x_coeff);}

            matrix_values.emplace_back(current_index, current_index, 1 + 2 * (x_coeff + y_coeff) ); // diagonal node
            
        }
    }

    Eigen::SparseMatrix<double> next_time_step_matrix(total_grid_size, total_grid_size);
    next_time_step_matrix.setFromTriplets(matrix_values.begin(), matrix_values.end());
    next_time_step_matrix.makeCompressed();

    matrix_store.A_matrix = next_time_step_matrix;

}

void resistive_B_term(const Grid& grid_state, std::vector<double>& bx_rate, std::vector<double>& by_rate, Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper>& solver, integrator_reserved_memory& workspace, const bool solve_implicitly) { // this function deals with the sole 'stiff' resistive magnetic term, with the option to calculate its value both explicitly and implicitly 
    
    const int physical_y_node_count = grid_state.ny;
    const int physical_x_node_count = grid_state.nx;
    
    for (int j = 0; j < physical_y_node_count; j++) {
        for (int i = 0; i < physical_x_node_count; i++) {

            int current_index = grid_state.indexC(grid_state.gi(i), grid_state.gj(j));
            int left_index = grid_state.indexC(grid_state.gi(i - 1), grid_state.gj(j));
            int bottom_index = grid_state.indexC(grid_state.gi(i), grid_state.gj(j - 1));

            double dBy_dx = (grid_state.byf[current_index] - grid_state.byf[left_index]) / grid_state.cell_width;
            double dBx_dy = (grid_state.bxf[current_index] - grid_state.bxf[bottom_index]) / grid_state.cell_height;
            
            workspace.Ez_corner_no_padding[j * physical_x_node_count + i] = grid_state.resistivity * (dBy_dx - dBx_dy);

        }
    }

    Eigen::Map<const Eigen::VectorXd> Ez_rhs(workspace.Ez_corner_no_padding.data(), workspace.Ez_corner_no_padding.size());
    Eigen::VectorXd Ez_field;

    // solve implicitly or explicitly
    if (solve_implicitly) {Ez_field = solver.solve(Ez_rhs);} 
    else {Ez_field = Ez_rhs;}

    for (int j = 0; j < physical_y_node_count; j++) {
        for (int i = 0; i < physical_x_node_count; i++) {
            
            int i_right = (i + 1) % physical_x_node_count;
            int j_top = (j + 1) % physical_y_node_count;

            double Ez_current = Ez_field[j * physical_x_node_count + i];
            double Ez_top = Ez_field[j_top * physical_x_node_count + i];
            double Ez_right = Ez_field[j * physical_x_node_count + i_right];

            int grid_index = grid_state.indexC(grid_state.gi(i), grid_state.gj(j));

            bx_rate[grid_index] = - (Ez_top - Ez_current) / grid_state.cell_height;
            by_rate[grid_index] =   (Ez_right - Ez_current) / grid_state.cell_width;

        }
    }
}

void IMEX(Grid& grid_imex, Grid& grid_intermediary, Grid& grid_whole_step, integrator_reserved_memory& workspace, ReconstructedValues& MUSCL_values, IMEX_Butcher_Tableus& coefficients, Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper>& solver, double dt) { // this is the main integrator function, all time stepping will be brought together within this function so once this is run, the next state will be populated

    Grid grid_initial = grid_imex;  
    
    global_state_dot f1_storage(workspace.f1_rho, workspace.f1_m_x, workspace.f1_m_y, workspace.f1_b_x_f, workspace.f1_b_y_f, workspace.f1_E);
    explicit_rhs(grid_imex, f1_storage, MUSCL_values, workspace); 
    
    // explicit evaluation of initial magnetic resistive terms
    resistive_B_term(grid_imex, workspace.g1_b_x_f, workspace.g1_b_y_f, solver, workspace, false); 
    
    grid_intermediary = grid_initial;
    grid_whole_step = grid_initial;
    const double diag_coeff = dt * coefficients.A_im[4]; 

    // ------------------------------------
    //          Intermediary Step
    // ------------------------------------

    const double total_explicit_coeff_1 = dt * coefficients.A_ex[3];
    const double total_implicit_coeff_1 = dt * coefficients.A_im[3];

    get_intermediary_grid(grid_imex, grid_intermediary.rho, workspace.f1_rho, total_explicit_coeff_1);
    get_intermediary_grid(grid_imex, grid_intermediary.mx, workspace.f1_m_x, total_explicit_coeff_1);
    get_intermediary_grid(grid_imex, grid_intermediary.my, workspace.f1_m_y, total_explicit_coeff_1);
    get_intermediary_grid(grid_imex, grid_intermediary.E, workspace.f1_E, total_explicit_coeff_1);
    get_intermediary_grid(grid_imex, grid_intermediary.bxf, workspace.f1_b_x_f, total_explicit_coeff_1);
    get_intermediary_grid(grid_imex, grid_intermediary.byf, workspace.f1_b_y_f, total_explicit_coeff_1);

    get_intermediary_grid(grid_imex, grid_intermediary.bxf, workspace.g1_b_x_f, total_implicit_coeff_1);
    get_intermediary_grid(grid_imex, grid_intermediary.byf, workspace.g1_b_y_f, total_implicit_coeff_1);

    grid_intermediary.fillGhostPeriodic();
    grid_intermediary.computePrimitives();

    // implicit evaluation of magnetic resistive terms
    resistive_B_term(grid_intermediary, workspace.b_x_f_solved_1, workspace.b_y_f_solved_1, solver, workspace, true);

    get_intermediary_grid(grid_imex, grid_intermediary.bxf, workspace.b_x_f_solved_1, diag_coeff);
    get_intermediary_grid(grid_imex, grid_intermediary.byf, workspace.b_y_f_solved_1, diag_coeff);

    grid_intermediary.fillGhostPeriodic();
    grid_intermediary.computePrimitives();
    
    global_state_dot f2_storage(workspace.f2_rho, workspace.f2_m_x, workspace.f2_m_y, workspace.f2_b_x_f, workspace.f2_b_y_f, workspace.f2_E);
    explicit_rhs(grid_intermediary, f2_storage, MUSCL_values, workspace);
    
    resistive_B_term(grid_intermediary, workspace.g2_b_x_f, workspace.g2_b_y_f, solver, workspace, false);

    // ------------------------------------
    //             Whole Step 
    // ------------------------------------

    const double total_explicit_coeff_2 = dt * coefficients.A_ex[6];
    const double total_explicit_coeff_3 = dt * coefficients.A_ex[7];
    const double total_implicit_coeff_2 = dt * coefficients.A_im[6];
    const double total_implicit_coeff_3 = dt * coefficients.A_im[7];

    get_intermediary_grid(grid_imex, grid_whole_step.rho, workspace.f1_rho, total_explicit_coeff_2);
    get_intermediary_grid(grid_imex, grid_whole_step.mx, workspace.f1_m_x, total_explicit_coeff_2);
    get_intermediary_grid(grid_imex, grid_whole_step.my, workspace.f1_m_y, total_explicit_coeff_2);
    get_intermediary_grid(grid_imex, grid_whole_step.E, workspace.f1_E, total_explicit_coeff_2);
    get_intermediary_grid(grid_imex, grid_whole_step.bxf, workspace.f1_b_x_f, total_explicit_coeff_2);
    get_intermediary_grid(grid_imex, grid_whole_step.byf, workspace.f1_b_y_f, total_explicit_coeff_2);

    get_intermediary_grid(grid_imex, grid_whole_step.rho, workspace.f2_rho, total_explicit_coeff_3);
    get_intermediary_grid(grid_imex, grid_whole_step.mx, workspace.f2_m_x, total_explicit_coeff_3);
    get_intermediary_grid(grid_imex, grid_whole_step.my, workspace.f2_m_y, total_explicit_coeff_3);
    get_intermediary_grid(grid_imex, grid_whole_step.E, workspace.f2_E, total_explicit_coeff_3);
    get_intermediary_grid(grid_imex, grid_whole_step.bxf, workspace.f2_b_x_f, total_explicit_coeff_3);
    get_intermediary_grid(grid_imex, grid_whole_step.byf, workspace.f2_b_y_f, total_explicit_coeff_3);

    get_intermediary_grid(grid_imex, grid_whole_step.bxf, workspace.g1_b_x_f, total_implicit_coeff_2);
    get_intermediary_grid(grid_imex, grid_whole_step.byf, workspace.g1_b_y_f, total_implicit_coeff_2);

    get_intermediary_grid(grid_imex, grid_whole_step.bxf, workspace.g2_b_x_f, total_implicit_coeff_3);
    get_intermediary_grid(grid_imex, grid_whole_step.byf, workspace.g2_b_y_f, total_implicit_coeff_3);

    grid_whole_step.fillGhostPeriodic();
    grid_whole_step.computePrimitives();

    resistive_B_term(grid_whole_step, workspace.b_x_f_solved_2, workspace.b_y_f_solved_2, solver, workspace, true);

    get_intermediary_grid(grid_imex, grid_whole_step.bxf, workspace.b_x_f_solved_2, diag_coeff);
    get_intermediary_grid(grid_imex, grid_whole_step.byf, workspace.b_y_f_solved_2, diag_coeff);

    grid_whole_step.fillGhostPeriodic();
    grid_whole_step.computePrimitives();
    
    resistive_B_term(grid_whole_step, workspace.g3_b_x_f, workspace.g3_b_y_f, solver, workspace, false);

    // ------------------------------------
    //          FINAL COMBINATION 
    // ------------------------------------

    const double f1_coeff = dt * coefficients.b_ex[0];
    const double f2_coeff = dt * coefficients.b_ex[1];
    const double g2_coeff = dt * coefficients.b_im[1];
    const double g3_coeff = dt * coefficients.b_im[2];

    get_intermediary_grid(grid_imex, grid_imex.rho, workspace.f1_rho, f1_coeff);
    get_intermediary_grid(grid_imex, grid_imex.mx, workspace.f1_m_x, f1_coeff);
    get_intermediary_grid(grid_imex, grid_imex.my, workspace.f1_m_y, f1_coeff);
    get_intermediary_grid(grid_imex, grid_imex.E, workspace.f1_E, f1_coeff);
    get_intermediary_grid(grid_imex, grid_imex.bxf, workspace.f1_b_x_f, f1_coeff);
    get_intermediary_grid(grid_imex, grid_imex.byf, workspace.f1_b_y_f, f1_coeff);

    get_intermediary_grid(grid_imex, grid_imex.rho, workspace.f2_rho, f2_coeff);
    get_intermediary_grid(grid_imex, grid_imex.mx, workspace.f2_m_x, f2_coeff);
    get_intermediary_grid(grid_imex, grid_imex.my, workspace.f2_m_y, f2_coeff);
    get_intermediary_grid(grid_imex, grid_imex.E, workspace.f2_E, f2_coeff);
    get_intermediary_grid(grid_imex, grid_imex.bxf, workspace.f2_b_x_f, f2_coeff);
    get_intermediary_grid(grid_imex, grid_imex.byf, workspace.f2_b_y_f, f2_coeff);

    get_intermediary_grid(grid_imex, grid_imex.bxf, workspace.g2_b_x_f, g2_coeff);
    get_intermediary_grid(grid_imex, grid_imex.byf, workspace.g2_b_y_f, g2_coeff);

    get_intermediary_grid(grid_imex, grid_imex.bxf, workspace.g3_b_x_f, g3_coeff);
    get_intermediary_grid(grid_imex, grid_imex.byf, workspace.g3_b_y_f, g3_coeff);

    grid_imex.fillGhostPeriodic();
    grid_imex.computePrimitives();
    
}