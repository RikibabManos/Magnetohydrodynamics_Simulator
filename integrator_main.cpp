#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <iomanip>
#include <chrono>
#include "Grid.hpp"
#include "MUSCL.hpp"
#include "HLL.hpp"
#include "integrator_help.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace std;

void get_divergences(integrator_reserved_memory& memory,  Grid& grid_div, global_state_dot& dU_explicit_rhs_memory) { // helper function to calculate the divergence of momentum density, mass density, B field and total energy density for all nodes, at the centre of the cell
    
    vector<double>& rho_derivative = (*dU_explicit_rhs_memory.rho_dot);
    vector<double>& mx_derivative = (*dU_explicit_rhs_memory.m_dot_x);
    vector<double>& my_derivative = (*dU_explicit_rhs_memory.m_dot_y);
    vector<double>& E_derivative = (*dU_explicit_rhs_memory.E_dot);
    vector<double>& bx_derivative = (*dU_explicit_rhs_memory.b_dot_x);
    vector<double>& by_derivative = (*dU_explicit_rhs_memory.b_dot_y);

    int total_x_nodes = grid_div.nx;
    int total_y_nodes = grid_div.ny;

    int j_index_initial = 0;
    int j_index_final = total_y_nodes;
    int i_index_initial = 0;
    int i_index_final = total_x_nodes;
    
    double cell_width = grid_div.cell_width;
    double cell_height = grid_div.cell_height; 

    for (int j = j_index_initial; j < j_index_final; j++){
        for (int i = i_index_initial; i < i_index_final; i++){
            
        int current_node_index = grid_div.indexC(grid_div.gi(i), grid_div.gj(j));
        int left_node_index = grid_div.indexC(grid_div.gi(i - 1), grid_div.gj(j));
        int bottom_node_index = grid_div.indexC(grid_div.gi(i), grid_div.gj(j - 1));
            
            rho_derivative[current_node_index] = -( get_partial_derivative(memory.rho_flux_horizontal[current_node_index], memory.rho_flux_horizontal[left_node_index], cell_width) + get_partial_derivative(memory.rho_flux_vertical[current_node_index], memory.rho_flux_vertical[bottom_node_index], cell_height) );
            E_derivative[current_node_index] = -( get_partial_derivative(memory.E_flux_horizontal[current_node_index], memory.E_flux_horizontal[left_node_index], cell_width) + get_partial_derivative(memory.E_flux_vertical[current_node_index], memory.E_flux_vertical[bottom_node_index], cell_height) );
            mx_derivative[current_node_index] = -( get_partial_derivative(memory.m_x_flux_horizontal[current_node_index], memory.m_x_flux_horizontal[left_node_index], cell_width) + get_partial_derivative(memory.m_x_flux_vertical[current_node_index], memory.m_x_flux_vertical[bottom_node_index], cell_height) );
            my_derivative[current_node_index] = -( get_partial_derivative(memory.m_y_flux_horizontal[current_node_index], memory.m_y_flux_horizontal[left_node_index], cell_width) + get_partial_derivative(memory.m_y_flux_vertical[current_node_index], memory.m_y_flux_vertical[bottom_node_index], cell_height) );
            bx_derivative[current_node_index] = -( get_partial_derivative(memory.b_x_flux_horizontal[current_node_index], memory.b_x_flux_horizontal[left_node_index], cell_width) + get_partial_derivative(memory.b_x_flux_vertical[current_node_index], memory.b_x_flux_vertical[bottom_node_index], cell_height) );
            by_derivative[current_node_index] = -( get_partial_derivative(memory.b_y_flux_horizontal[current_node_index], memory.b_y_flux_horizontal[left_node_index], cell_width) + get_partial_derivative(memory.b_y_flux_vertical[current_node_index], memory.b_y_flux_vertical[bottom_node_index], cell_height) );

        }
    }
    

}

void flux_population(Grid& grid_flux, ReconstructedValues& RV, integrator_reserved_memory& reserve_flux){ 
    // --- NOTE TO SELF, CHANGE TO USE POINTERS IN FUTURE FOR SPEED --- 
    // function that populates each node with its flux values and stores in designated integrator_memory_reserve vector
    
    // define structs for single node value storage
    PrimitiveState left_face{};
    PrimitiveState right_face{};
    PrimitiveState top_face{};
    PrimitiveState bottom_face{};

    int total_x_nodes = grid_flux.nx;
    int total_y_nodes = grid_flux.ny;

    int j_index_start = -1;
    int j_index_end = total_y_nodes;
    int i_index_start = -1;
    int i_index_end = total_x_nodes;

    for (int j = j_index_start; j < j_index_end; j++){    
        for (int i = i_index_start; i < i_index_end; i++){

            int current_cell_index = grid_flux.indexC(grid_flux.gi(i), grid_flux.gj(j));
            int left_cell_index = grid_flux.indexC(grid_flux.gi(i - 1), grid_flux.gj(j));
            int bottom_cell_index = grid_flux.indexC(grid_flux.gi(i), grid_flux.gj(j - 1));     

            // populate PrimitiveSate struct so we can use as inputs to the computeHLLFlux functions
            generate_cell_properties(left_face, right_face, top_face, bottom_face, grid_flux, RV, i, j);
            
            // calculate current node's flux values
            FluxVector x_direction_flux = computeHLLFluxX(left_face, right_face);
            FluxVector y_direction_flux = computeHLLFluxY(bottom_face, top_face);
            
            // store values in scrapboard memory
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
};

void explicit_rhs(Grid& grid_ex, global_state_dot& dU_explicit, ReconstructedValues& MUSCL_output, integrator_reserved_memory& reserve){ // explicit half of IMEX integrator, returns a partial global state derivative array of only non-'stiff' variables

    grid_ex.fillGhostPeriodic();                     // sets ghost cell values to those of opposing sides, mimics periodic boundry coditions
    grid_ex.computePrimitives();                     // calculates primitive variables at nodes from previously calculated conserved variables 
    MUSCLReconstructAll(grid_ex, MUSCL_output);      // calculates primitive values at cell faces 
    flux_population(grid_ex, MUSCL_output, reserve); // populate reserved memory with flux values
    get_divergences(reserve, grid_ex, dU_explicit);  // calculates the current step's explicit rhs terms

}

inline void implicit_rhs(Grid& grid_laplacian, global_state_dot& dU_implicit_rhs_memory){

    //vector<double>& mx_stiff_terms = (*dU_implicit_rhs_memory.m_dot_x);
    //vector<double>& my_stiff_terms = (*dU_implicit_rhs_memory.m_dot_y);
    //vector<double>& E_stiff_terms = (*dU_implicit_rhs_memory.E_dot);
    vector<double>& bx_stiff_terms = (*dU_implicit_rhs_memory.b_dot_x);
    vector<double>& by_stiff_terms = (*dU_implicit_rhs_memory.b_dot_y);

    int j_index_start = 0;
    int j_index_final = grid_laplacian.ny;
    int i_index_start = 0;
    int i_index_final = grid_laplacian.nx;

    double node_x_seperation_sq = grid_laplacian.cell_width * grid_laplacian.cell_width;
    double node_y_seperation_sq = grid_laplacian.cell_height * grid_laplacian.cell_height;

    for (int j = 0; j < j_index_final; j++){
        for (int i = 0; i < i_index_final; i++){

            int physical_i = grid_laplacian.gi(i);
            int physical_j = grid_laplacian.gj(j);
            int physical_right = grid_laplacian.gi(i + 1);
            int physical_left = grid_laplacian.gi(i - 1);
            int physical_top = grid_laplacian.gj(j + 1);
            int physical_bottom = grid_laplacian.gj(j - 1);

            int current_cell_index = grid_laplacian.indexC(physical_i, physical_j);
            int right_cell_index = grid_laplacian.indexC(physical_right, physical_j);
            int left_cell_index = grid_laplacian.indexC(physical_left, physical_j);
            int top_cell_index = grid_laplacian.indexC(physical_i, physical_top);
            int bottom_cell_index = grid_laplacian.indexC(physical_i, physical_bottom);

            double current_b_x_laplacian = ( grid_laplacian.bxc[right_cell_index] - 2 * grid_laplacian.bxc[current_cell_index] + grid_laplacian.bxc[left_cell_index] ) / node_x_seperation_sq + ( grid_laplacian.bxc[top_cell_index] - 2 * grid_laplacian.bxc[current_cell_index] + grid_laplacian.bxc[bottom_cell_index] ) / node_y_seperation_sq;
            double current_b_y_laplacian = ( grid_laplacian.byc[right_cell_index] - 2 * grid_laplacian.byc[current_cell_index] + grid_laplacian.byc[left_cell_index] ) / node_x_seperation_sq + ( grid_laplacian.byc[top_cell_index] - 2 * grid_laplacian.byc[current_cell_index] + grid_laplacian.byc[bottom_cell_index] ) / node_y_seperation_sq;

            bx_stiff_terms[current_cell_index] = current_b_x_laplacian * grid_laplacian.resistivity;
            by_stiff_terms[current_cell_index] = current_b_y_laplacian * grid_laplacian.resistivity;
        }
    }

}

void generate_A_matrix(Grid& grid_solve, integrator_reserved_memory& matrix_store, double alpha){

    int x_size = grid_solve.nx;
    int y_size = grid_solve.ny;
    int total_grid_size = x_size * y_size;
    double x_coeff = alpha / (grid_solve.cell_width * grid_solve.cell_width);
    double y_coeff = alpha / (grid_solve.cell_height * grid_solve.cell_height);

    std::vector<Eigen::Triplet<double>> matrix_values;
    matrix_values.reserve(total_grid_size * 5); // maximum of 5 non-zero values per row

    int j_start = 0;
    int j_end = y_size;
    int i_start = 0;
    int i_end = x_size;

    // assign matrix non-zero values
    for (int j = j_start; j < j_end; j++) {
        for (int i = i_start; i < i_end; i++) {

            int current_index = j * x_size + i;

            if (j == 0) {matrix_values.emplace_back(current_index, (y_size - 1) * x_size + i, - y_coeff);} // bottom node
            else {matrix_values.emplace_back(current_index, current_index - x_size, - y_coeff);}

            if (j == j_end - 1) {matrix_values.emplace_back(current_index, i, - y_coeff);} // top node
            else {matrix_values.emplace_back(current_index, current_index + x_size, - y_coeff);}

            if (i == 0) {matrix_values.emplace_back(current_index, current_index + x_size - 1, - x_coeff);} // left node
            else {matrix_values.emplace_back(current_index, current_index - 1, - x_coeff);}

            if (i == i_end - 1) {matrix_values.emplace_back(current_index, current_index - x_size + 1, - x_coeff);} // right node
            else {matrix_values.emplace_back(current_index, current_index + 1, - x_coeff);}

            matrix_values.emplace_back(current_index, current_index, 1 + 2 * (x_coeff + y_coeff) ); // diagonal node
            
        }
    }

    Eigen::SparseMatrix<double> next_time_step_matrix(total_grid_size, total_grid_size);
    next_time_step_matrix.setFromTriplets(matrix_values.begin(), matrix_values.end());
    next_time_step_matrix.makeCompressed();

    matrix_store.A_matrix = next_time_step_matrix;

}

void remove_ghost_padding(Grid& grid_preparation, integrator_reserved_memory& removal_memory) { // extracts only physical nodes

    for (int j = 0; j < grid_preparation.ny; j++) {
        for (int i = 0; i < grid_preparation.nx; i++) {

            int true_index = j * grid_preparation.nx + i;
            int padded_vector_index = grid_preparation.indexC(grid_preparation.gi(i), grid_preparation.gj(j));
            removal_memory.b_x_c_no_padding[true_index] = grid_preparation.bxc[padded_vector_index];
            removal_memory.b_y_c_no_padding[true_index] = grid_preparation.byc[padded_vector_index];

        }
    }
}

void add_ghost_padding(Grid& grid_sub_step_complete, integrator_reserved_memory& added_memory) {

    for (int j = 0; j < grid_sub_step_complete.ny; j++) {
        for (int i = 0; i < grid_sub_step_complete.nx; i++) { 
            
            int physical_index = j * grid_sub_step_complete.nx + i;
            int grid_index = grid_sub_step_complete.indexC(grid_sub_step_complete.gi(i), grid_sub_step_complete.gj(j));

            grid_sub_step_complete.bxc[grid_index] = added_memory.b_x_c_no_padding[physical_index];
            grid_sub_step_complete.byc[grid_index] = added_memory.b_y_c_no_padding[physical_index];

        }
    }
}

void sync_B_center_to_face_values(Grid& grid) { // interpolates the updated cell-centered B fields back to the staggered faces
    
    for (int j = grid.gj(-1); j <= grid.gj(grid.ny); j++) {
        for (int i = grid.gi(-1); i <= grid.gi(grid.nx); i++) {
            
            int current = grid.indexC(i, j);
            int left_neighbor = grid.indexC(i - 1, j);
            int bottom_neighbor = grid.indexC(i, j - 1);

            // left face is average of current center and left center
            grid.bxf[current] = 0.5 * (grid.bxc[current] + grid.bxc[left_neighbor]);
            
            // bottom face is average of current center and bottom center
            grid.byf[current] = 0.5 * (grid.byc[current] + grid.byc[bottom_neighbor]);
        }
    }
}

void IMEX(Grid& grid_imex, Grid& grid_intermediary, Grid& grid_whole_step, integrator_reserved_memory& workspace, ReconstructedValues& MUSCL_values, IMEX_Butcher_Tableus& coefficients, Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper>& solver, double dt){ // function that calculates the new global state of the system using a combination of implicit and explicit methods

    Grid grid_initial = grid_imex;  

    global_state_dot f1_storage(workspace.f1_rho, workspace.f1_m_x, workspace.f1_m_y, workspace.f1_b_x_c, workspace.f1_b_y_c, workspace.f1_E);
    explicit_rhs(grid_imex, f1_storage, MUSCL_values, workspace); // calculate the explicit derivative terms at the initial state
    
    // reset intermediary grids
    grid_intermediary = grid_initial;
    grid_whole_step = grid_initial;

    // --- INTERMEDIARY STEP ---

    // matrix setup 1
    double matrix_coeff = dt * grid_imex.resistivity;
    generate_A_matrix(grid_imex, workspace, matrix_coeff * coefficients.A_im[4]);
    solver.compute(workspace.A_matrix);

    // calculate psuedo grid state due to explicit rhs only

    double total_explicit_coeff_1 = dt * coefficients.A_ex[3];
    get_intermediary_grid(grid_intermediary.rho, workspace.f1_rho, total_explicit_coeff_1);
    get_intermediary_grid(grid_intermediary.mx, workspace.f1_m_x, total_explicit_coeff_1);
    get_intermediary_grid(grid_intermediary.my, workspace.f1_m_y, total_explicit_coeff_1);
    get_intermediary_grid(grid_intermediary.E, workspace.f1_E, total_explicit_coeff_1);
    get_intermediary_grid(grid_intermediary.bxc, workspace.f1_b_x_c, total_explicit_coeff_1);
    get_intermediary_grid(grid_intermediary.byc, workspace.f1_b_y_c, total_explicit_coeff_1);

    global_state_dot g1_storage(workspace.g1_rho, workspace.g1_m_x, workspace.g1_m_y, workspace.g1_b_x_c, workspace.g1_b_y_c, workspace.g1_E);
    implicit_rhs(grid_imex, g1_storage);
    
    // calculate known rhs 

    double total_implicit_coeff_1 = dt * coefficients.A_im[3];
    //get_intermediary_grid(grid_intermediary.rho, workspace.g1_rho, total_implicit_coeff_1);
    //get_intermediary_grid(grid_intermediary.mx, workspace.g1_m_x, total_implicit_coeff_1);
    //get_intermediary_grid(grid_intermediary.my, workspace.g1_m_y, total_implicit_coeff_1);
    //get_intermediary_grid(grid_intermediary.E, workspace.g1_E, total_implicit_coeff_1);
    get_intermediary_grid(grid_intermediary.bxc, workspace.g1_b_x_c, total_implicit_coeff_1);
    get_intermediary_grid(grid_intermediary.byc, workspace.g1_b_y_c, total_implicit_coeff_1);

    // use known rhs to implicityly solve for terms with stiff components

    remove_ghost_padding(grid_intermediary, workspace);

    Eigen::Map<const Eigen::VectorXd> bxc_rhs_1 (
        workspace.b_x_c_no_padding.data(),
        workspace.b_x_c_no_padding.size()
    );

    Eigen::Map<const Eigen::VectorXd> byc_rhs_1 (
        workspace.b_y_c_no_padding.data(),
        workspace.b_y_c_no_padding.size()
    );

    Eigen::VectorXd bxc_1 = solver.solve(bxc_rhs_1);
    Eigen::VectorXd byc_1 = solver.solve(byc_rhs_1);

    // convert result into std::vector so that we can more easily append to grid_intermediary
    workspace.b_x_c_no_padding.assign(bxc_1.data(), bxc_1.data() + bxc_1.size());
    workspace.b_y_c_no_padding.assign(byc_1.data(), byc_1.data() + byc_1.size());

    add_ghost_padding(grid_intermediary, workspace);
    sync_B_center_to_face_values(grid_intermediary);
    grid_intermediary.fillGhostPeriodic();
    grid_intermediary.computePrimitives();

    // --- WHOLE STEP ---

    // matrix setup 2
    generate_A_matrix(grid_imex, workspace, matrix_coeff * coefficients.A_im[8]);
    solver.compute(workspace.A_matrix);

    // generate intermediary step explicit derivative terms

    global_state_dot f2_storage(workspace.f2_rho, workspace.f2_m_x, workspace.f2_m_y, workspace.f2_b_x_c, workspace.f2_b_y_c, workspace.f2_E);
    explicit_rhs(grid_intermediary, f2_storage, MUSCL_values, workspace);

    // adding weighted f1 contribution
    double total_explicit_coeff_2 = dt * coefficients.A_ex[6];
    get_intermediary_grid(grid_whole_step.rho, workspace.f1_rho, total_explicit_coeff_2);
    get_intermediary_grid(grid_whole_step.mx, workspace.f1_m_x, total_explicit_coeff_2);
    get_intermediary_grid(grid_whole_step.my, workspace.f1_m_y, total_explicit_coeff_2);
    get_intermediary_grid(grid_whole_step.E, workspace.f1_E, total_explicit_coeff_2);
    get_intermediary_grid(grid_whole_step.bxc, workspace.f1_b_x_c, total_explicit_coeff_2);
    get_intermediary_grid(grid_whole_step.byc, workspace.f1_b_y_c, total_explicit_coeff_2);

    // adding weighted f2 contribution
    double total_explicit_coeff_3 = dt * coefficients.A_ex[7];
    get_intermediary_grid(grid_whole_step.rho, workspace.f2_rho, total_explicit_coeff_3);
    get_intermediary_grid(grid_whole_step.mx, workspace.f2_m_x, total_explicit_coeff_3);
    get_intermediary_grid(grid_whole_step.my, workspace.f2_m_y, total_explicit_coeff_3);
    get_intermediary_grid(grid_whole_step.E, workspace.f2_E, total_explicit_coeff_3);
    get_intermediary_grid(grid_whole_step.bxc, workspace.f2_b_x_c, total_explicit_coeff_3);
    get_intermediary_grid(grid_whole_step.byc, workspace.f2_b_y_c, total_explicit_coeff_3);

    sync_B_center_to_face_values(grid_whole_step);
    grid_whole_step.fillGhostPeriodic();
    grid_whole_step.computePrimitives();

    global_state_dot g2_storage(workspace.g2_rho, workspace.g2_m_x, workspace.g2_m_y, workspace.g2_b_x_c, workspace.g2_b_y_c, workspace.g2_E);
    implicit_rhs(grid_intermediary, g2_storage);

    // calculate final grid state due to explicit and implicit rhs contributions

    double total_implicit_coeff_2 = dt * coefficients.A_im[6];
    //get_intermediary_grid(grid_whole_step.rho, workspace.g1_rho, total_implicit_coeff_2);
    //get_intermediary_grid(grid_whole_step.mx, workspace.g1_m_x, total_implicit_coeff_2);
    //get_intermediary_grid(grid_whole_step.my, workspace.g1_m_y, total_implicit_coeff_2);
    //get_intermediary_grid(grid_whole_step.E, workspace.g1_E, total_implicit_coeff_2);
    get_intermediary_grid(grid_whole_step.bxc, workspace.g1_b_x_c, total_implicit_coeff_2);
    get_intermediary_grid(grid_whole_step.byc, workspace.g1_b_y_c, total_implicit_coeff_2);

    double total_implicit_coeff_3 = dt * coefficients.A_im[7];
    //get_intermediary_grid(grid_whole_step.rho, workspace.g2_rho, total_implicit_coeff_3);
    //get_intermediary_grid(grid_whole_step.mx, workspace.g2_m_x, total_implicit_coeff_3);
    //get_intermediary_grid(grid_whole_step.my, workspace.g2_m_y, total_implicit_coeff_3);
    //get_intermediary_grid(grid_whole_step.E, workspace.g2_E, total_implicit_coeff_3);
    get_intermediary_grid(grid_whole_step.bxc, workspace.g2_b_x_c, total_implicit_coeff_3);
    get_intermediary_grid(grid_whole_step.byc, workspace.g2_b_y_c, total_implicit_coeff_3);

    sync_B_center_to_face_values(grid_whole_step);
    grid_whole_step.fillGhostPeriodic();
    grid_whole_step.computePrimitives();

    // use known rhs to implicityly solve for terms with stiff components

    remove_ghost_padding(grid_whole_step, workspace);

    Eigen::Map<const Eigen::VectorXd> bxc_rhs_2 (
        workspace.b_x_c_no_padding.data(),
        workspace.b_x_c_no_padding.size()
    );

    Eigen::Map<const Eigen::VectorXd> byc_rhs_2 (
        workspace.b_y_c_no_padding.data(),
        workspace.b_y_c_no_padding.size()
    );

    Eigen::VectorXd bxc_2 = solver.solve(bxc_rhs_2);
    Eigen::VectorXd byc_2 = solver.solve(byc_rhs_2);

    // convert result into std::vector so that we can more easily append to grid_whole_step
    workspace.b_x_c_no_padding.assign(bxc_2.data(), bxc_2.data() + bxc_2.size());
    workspace.b_y_c_no_padding.assign(byc_2.data(), byc_2.data() + byc_2.size());

    add_ghost_padding(grid_whole_step, workspace);
    sync_B_center_to_face_values(grid_whole_step);
    grid_whole_step.fillGhostPeriodic();
    grid_whole_step.computePrimitives();

    // --- FINAL DERIVATIVE ---

    global_state_dot g3_storage(workspace.g3_rho, workspace.g3_m_x, workspace.g3_m_y, workspace.g3_b_x_c, workspace.g3_b_y_c, workspace.g3_E);
    implicit_rhs(grid_whole_step, g3_storage);

    // now compute the new state as a weighted sum of the initial and intermediary state contributions

    double f1_coeff = coefficients.b_ex[0];
    double f2_coeff = coefficients.b_ex[1];
    double g2_coeff = coefficients.b_im[1];
    double g3_coeff = coefficients.b_im[2];

    get_intermediary_grid(grid_imex.rho, workspace.f1_rho, dt * f1_coeff);
    get_intermediary_grid(grid_imex.mx, workspace.f1_m_x, dt * f1_coeff);
    get_intermediary_grid(grid_imex.my, workspace.f1_m_y, dt * f1_coeff);
    get_intermediary_grid(grid_imex.E, workspace.f1_E, dt * f1_coeff);
    get_intermediary_grid(grid_imex.bxc, workspace.f1_b_x_c, dt * f1_coeff);
    get_intermediary_grid(grid_imex.byc, workspace.f1_b_y_c, dt * f1_coeff);

    get_intermediary_grid(grid_imex.rho, workspace.f2_rho, dt * f2_coeff);
    get_intermediary_grid(grid_imex.mx, workspace.f2_m_x, dt * f2_coeff);
    get_intermediary_grid(grid_imex.my, workspace.f2_m_y, dt * f2_coeff);
    get_intermediary_grid(grid_imex.E, workspace.f2_E, dt * f2_coeff);
    get_intermediary_grid(grid_imex.bxc, workspace.f2_b_x_c, dt * f2_coeff);
    get_intermediary_grid(grid_imex.byc, workspace.f2_b_y_c, dt * f2_coeff);

    //get_intermediary_grid(grid_imex.rho, workspace.g2_rho, dt * g2_coeff);
    //get_intermediary_grid(grid_imex.mx, workspace.g2_m_x, dt * g2_coeff);
    //get_intermediary_grid(grid_imex.my, workspace.g2_m_y, dt * g2_coeff);
    //get_intermediary_grid(grid_imex.E, workspace.g2_E, dt * g2_coeff);
    get_intermediary_grid(grid_imex.bxc, workspace.g2_b_x_c, dt * g2_coeff);
    get_intermediary_grid(grid_imex.byc, workspace.g2_b_y_c, dt * g2_coeff);
    
    // this holds the new state values for every node
    //get_intermediary_grid(grid_imex.rho, workspace.g3_rho, dt * g3_coeff);
    //get_intermediary_grid(grid_imex.mx, workspace.g3_m_x, dt * g3_coeff);
    //get_intermediary_grid(grid_imex.my, workspace.g3_m_y, dt * g3_coeff);
    //get_intermediary_grid(grid_imex.E, workspace.g3_E, dt * g3_coeff);
    get_intermediary_grid(grid_imex.bxc, workspace.g3_b_x_c, dt * g3_coeff);
    get_intermediary_grid(grid_imex.byc, workspace.g3_b_y_c, dt * g3_coeff);

    sync_B_center_to_face_values(grid_imex);
    grid_imex.fillGhostPeriodic();
    grid_imex.computePrimitives();
}

int main(){

    auto start_time = chrono::high_resolution_clock::now();

    // create initial global state
    double animation_duration = 100.0;  // seconds
    const double time_interval = 0.05;  // seconds
    int x_node_count = 100;
    int y_node_count = 100;
    int ghost_node_count = 2;
    int node_total = (x_node_count + 2 * ghost_node_count) * (y_node_count + 2 * ghost_node_count);

    // initialise helper structs
    integrator_reserved_memory memory_storage(node_total, x_node_count * y_node_count);
    ReconstructedValues MUSCL_memory(node_total);
    IMEX_Butcher_Tableus coefficients_tableu;
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper> matrix_solver;

    Grid global_state(x_node_count, y_node_count, 2);   
    global_state.cell_width = 1.0;
    global_state.cell_height = 1.0;
    global_state.resistivity = 10.0;
    
    // setting up pulse size and location
    double x_mid = 0.5 * x_node_count * global_state.cell_width;
    double y_mid = 0.5 * y_node_count * global_state.cell_height;
    double radius_sq = 10.0 * 10.0; // width of the pulse
    
    for (int j = 0; j < y_node_count + 2 * ghost_node_count; ++j) {
        for (int i = 0; i < x_node_count + 2 * ghost_node_count; ++i) {
            
            int idx = global_state.indexC(i, j);
        
            // coordinates relative to center (for gaussian test)
            double x_relative = (i - ghost_node_count + 0.5) * global_state.cell_width;
            double y_relative = (j - ghost_node_count + 0.5) * global_state.cell_height;
            double dist_sq = (x_relative - x_mid) * (x_relative - x_mid) + (y_relative - y_mid) * (y_relative - y_mid);
            
            // acoustic wave test values
            double amplitude_x = 0.1;
            double amplitude_y = 0.1;
            double x_position = (i - ghost_node_count) * global_state.cell_width;
            double y_position = (j - ghost_node_count) * global_state.cell_height;
            double physical_grid_width = global_state.cell_width * x_node_count;
            double physical_grid_height = global_state.cell_height * y_node_count;
            constexpr double pi = 3.14159265359;
            constexpr double adiabatic_index = 5 / 3;
            double x_sin = amplitude_x * sin(2 * pi * x_position / physical_grid_width);
            double y_sin = amplitude_y * sin(2 * pi * y_position / physical_grid_height);

            // magnetic wave test values
            double amplitude_bx = 0.0;
            double amplitude_by = 1.0;
            double kx = 2.0 * pi / physical_grid_width;
            double ky = 2.0 * pi / physical_grid_height;
            double rho = 1.0;
            double vx  = 0.0;
            double vy  = 0.0;
            double p   = 1.0;
            double bx = 0.0;
            double by = amplitude_by * sin(kx * x_position) * sin(ky * y_position);

            // primitive values for 1D acoustic wave test
            //double rho = 1.0 + x_sin + y_sin;
            //double vx  = x_sin;
            //double vy  = y_sin;
            //double p   = (1 + adiabatic_index * (x_sin + y_sin)) * 0.6;
            //double bx  = 0.0;
            //double by  = 0.0;

            // primitive values for gaussian test
            //double rho = 1.0 + exp(-dist_sq / radius_sq);
            //double vx  = 1.0;
            //double vy  = 1.0;
            //double p   = 1.0;
            //double bx  = 1.0;
            //double by  = 0.0;
        
            PrimitiveState w{rho, vx, vy, p, bx, by};
            ConservedState u = ConservedState::fromPrimitive(w);
        
            // assign conserved fields
            global_state.rho[idx] = u.rho;
            global_state.mx[idx]  = u.mx;
            global_state.my[idx]  = u.my;
            global_state.E[idx]   = u.E;

            // asign B fields
            global_state.bxc[idx] = u.bx;
            global_state.byc[idx] = u.by;
            global_state.bxf[idx] = u.bx;
            global_state.byf[idx] = u.by;
        
            // assign primitive fields
            global_state.p[idx]  = p;
            global_state.vx[idx] = vx;
            global_state.vy[idx] = vy;
        }
    }

    Grid intermediary_grid_1(x_node_count, y_node_count, 2);
    Grid intermediary_grid_2(x_node_count, y_node_count, 2);
    intermediary_grid_1 = global_state;
    intermediary_grid_2 = global_state;

    global_state.fillGhostPeriodic();
    intermediary_grid_1.fillGhostPeriodic(); 
    intermediary_grid_2.fillGhostPeriodic(); 

    
    int frame_index = 0; 
    int snapshot_frequency = 10;

    for (double _ = 0; _ <= animation_duration; _ += time_interval){

        frame_index ++;

        IMEX(global_state, intermediary_grid_1, intermediary_grid_2, memory_storage, MUSCL_memory, coefficients_tableu, matrix_solver, time_interval);  

        if (frame_index % snapshot_frequency == 0){

            global_state.computePrimitives(); // calculate primitive values we want to output

            // write state data into binary output file

            fstream binary_fout; 
            std::ostringstream ss;
            ss << "snapshot_" << std::setw(5) << std::setfill('0') << frame_index << ".dat";
            string filename = ss.str();
            binary_fout.open(filename, ios::out | ios::binary);

            if (!binary_fout.is_open()){
                cout << "Binary file not accessed error" << '\n';
                return 1; // return error code
            }

            // adding header information to file
            double header[7] = {x_node_count, y_node_count, ghost_node_count, snapshot_frequency, global_state.cell_width, global_state.cell_height, time_interval}; // ignore
            binary_fout.write(reinterpret_cast<char*>(header), sizeof(header));

            // all data in global state written into a binary file sequentially
            binary_fout.write(reinterpret_cast<char*>(global_state.rho.data()), (global_state.rho.size() * sizeof(global_state.rho[0])));
            binary_fout.write(reinterpret_cast<char*>(global_state.E.data()), (global_state.E.size() * sizeof(global_state.E[0])));
            binary_fout.write(reinterpret_cast<char*>(global_state.p.data()), (global_state.p.size() * sizeof(global_state.p[0])));
            binary_fout.write(reinterpret_cast<char*>(global_state.vx.data()), (global_state.vx.size() * sizeof(global_state.vx[0])));
            binary_fout.write(reinterpret_cast<char*>(global_state.vy.data()), (global_state.vy.size() * sizeof(global_state.vy[0])));
            binary_fout.write(reinterpret_cast<char*>(global_state.bxc.data()), (global_state.bxc.size() * sizeof(global_state.bxc[0])));
            binary_fout.write(reinterpret_cast<char*>(global_state.byc.data()), (global_state.byc.size() * sizeof(global_state.byc[0])));

            binary_fout.close(); 

        }

    }

    // calculating total runtime
    auto end_time = chrono::high_resolution_clock::now();
    auto runtime_milliseconds = chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    auto runtime_seconds = chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    cout << "Total runtime: " << runtime_seconds.count() << " seconds, " << (runtime_milliseconds.count() % 1000) << " milliseconds" << "\n";
    
    return 0;
}
