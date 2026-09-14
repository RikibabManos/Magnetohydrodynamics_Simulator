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

void time_derivatives(integrator_reserved_memory& memory,  Grid& grid_div, global_state_dot& dU_explicit_rhs_memory) { // calculates the time derivatives of each quantity 
    
    std::vector<double>& rho_derivative = (*dU_explicit_rhs_memory.rho_dot);
    std::vector<double>& mx_derivative = (*dU_explicit_rhs_memory.m_dot_x);
    std::vector<double>& my_derivative = (*dU_explicit_rhs_memory.m_dot_y);
    std::vector<double>& E_derivative = (*dU_explicit_rhs_memory.E_dot);
    std::vector<double>& bxf_derivative = (*dU_explicit_rhs_memory.b_dot_x);
    std::vector<double>& byf_derivative = (*dU_explicit_rhs_memory.b_dot_y);

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

    int total_x_nodes = grid_B.nx;
    int total_y_nodes = grid_B.ny;

    int j_index_start = -1;
    int j_index_end = total_y_nodes + 1;
    int i_index_start = -1;
    int i_index_end = total_x_nodes + 1;

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
    // --- NOTE TO SELF, CHANGE TO USE POINTERS IN FUTURE FOR SPEED --- 
    
    // define PrimitiveState structs as FluxVector reuired those as inputs
    PrimitiveState left_face{};
    PrimitiveState right_face{};
    PrimitiveState top_face{};
    PrimitiveState bottom_face{};

    int total_x_nodes = grid_flux.nx;
    int total_y_nodes = grid_flux.ny;

    int j_index_start = -1;
    int j_index_end = total_y_nodes + 1;
    int i_index_start = -1;
    int i_index_end = total_x_nodes + 1;

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

void explicit_rhs(Grid& grid_ex, global_state_dot& dU_explicit, ReconstructedValues& MUSCL_output, integrator_reserved_memory& reserve){ // explicit half of IMEX integrator, returns a partial global_state_dot struct of only non-'stiff' variables

    grid_ex.fillGhostPeriodic();                               
    grid_ex.computePrimitives();                            
    MUSCLReconstructAll(grid_ex, MUSCL_output);           
    flux_population(grid_ex, MUSCL_output, reserve);      
    cell_corner_electric_fields(reserve, grid_ex); 
    time_derivatives(reserve, grid_ex, dU_explicit);       

}

void generate_A_matrix(Grid& grid_solve, integrator_reserved_memory& matrix_store, double alpha) { // builds a matrix representing the discretised Laplacian operator in 2D, using a 5-point finite-difference stencil

    int x_size = grid_solve.nx;
    int y_size = grid_solve.ny;
    int total_grid_size = x_size * y_size;
    double x_coeff = alpha / (grid_solve.cell_width * grid_solve.cell_width);
    double y_coeff = alpha / (grid_solve.cell_height * grid_solve.cell_height);

    std::vector<Eigen::Triplet<double>> matrix_values;
    matrix_values.reserve(total_grid_size * 5); // maximum of 5 non-zero values per row, 

    int j_start = 0;
    int j_end = y_size;
    int i_start = 0;
    int i_end = x_size;

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

struct divergence_checker_values {

    double root_mean_square_div;
    double max_norm_divergence;

};

divergence_checker_values calculate_current_div_B(Grid& grid_test) { // function to calculate current max and norm divergence of B field, used to test if constrained transport method if working as intended

    double max_divergence_norm = 0.0;
    double root_mean_square_norm_divergence = 0.0;

    for (int j = 0; j < grid_test.ny; j++) {
        for (int i = 0; i < grid_test.nx; i++) {

            int current_physical_index = grid_test.indexC(grid_test.gi(i), grid_test.gj(j));
            int right_physical_index = grid_test.indexC(grid_test.gi(i + 1), grid_test.gj(j));
            int top_physical_index = grid_test.indexC(grid_test.gi(i), grid_test.gj(j + 1));
            
            double current_x_divergence = (grid_test.bxf[right_physical_index] - grid_test.bxf[current_physical_index]) / grid_test.cell_width;
            double current_y_divergence = (grid_test.byf[top_physical_index] - grid_test.byf[current_physical_index]) / grid_test.cell_height;
            double total_current_divergence = current_x_divergence + current_y_divergence;
            root_mean_square_norm_divergence += total_current_divergence * total_current_divergence;

            if (std::abs(total_current_divergence) > max_divergence_norm) {
                max_divergence_norm = std::abs(total_current_divergence);
            }
        }
    }
    
    root_mean_square_norm_divergence = sqrt(root_mean_square_norm_divergence / (grid_test.nx * grid_test.ny));
    return {root_mean_square_norm_divergence, max_divergence_norm};

}

void resistive_B_term(Grid& grid_state, std::vector<double>& bx_rate, std::vector<double>& by_rate, Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper>& solver, integrator_reserved_memory& workspace, bool solve_implicitly) { // this function deals with the sole 'stiff' resistive magnetic term, with the option to calculate its value both explicitly and implicitly 
    
    for (int j = 0; j < grid_state.ny; j++) {
        for (int i = 0; i < grid_state.nx; i++) {

            int current_index = grid_state.indexC(grid_state.gi(i), grid_state.gj(j));
            int left_index = grid_state.indexC(grid_state.gi(i - 1), grid_state.gj(j));
            int bottom_index = grid_state.indexC(grid_state.gi(i), grid_state.gj(j - 1));

            double dBy_dx = (grid_state.byf[current_index] - grid_state.byf[left_index]) / grid_state.cell_width;
            double dBx_dy = (grid_state.bxf[current_index] - grid_state.bxf[bottom_index]) / grid_state.cell_height;
            
            workspace.Ez_corner_no_padding[j * grid_state.nx + i] = grid_state.resistivity * (dBy_dx - dBx_dy);
        }
    }

    Eigen::Map<const Eigen::VectorXd> Ez_rhs(workspace.Ez_corner_no_padding.data(), workspace.Ez_corner_no_padding.size());
    Eigen::VectorXd Ez_field;

    // solve implicitly or explicitly
    if (solve_implicitly) {Ez_field = solver.solve(Ez_rhs);} 
    else {Ez_field = Ez_rhs;}

    for (int j = 0; j < grid_state.ny; j++) {
        for (int i = 0; i < grid_state.nx; i++) {
            int i_right = (i + 1) % grid_state.nx;
            int j_top = (j + 1) % grid_state.ny;

            double Ez_current = Ez_field[j * grid_state.nx + i];
            double Ez_top     = Ez_field[j_top * grid_state.nx + i];
            double Ez_right   = Ez_field[j * grid_state.nx + i_right];

            int grid_index = grid_state.indexC(grid_state.gi(i), grid_state.gj(j));

            bx_rate[grid_index] = - (Ez_top - Ez_current) / grid_state.cell_height;
            by_rate[grid_index] =   (Ez_right - Ez_current) / grid_state.cell_width;
        }
    }
}

void IMEX(Grid& grid_imex, Grid& grid_intermediary, Grid& grid_whole_step, integrator_reserved_memory& workspace, ReconstructedValues& MUSCL_values, IMEX_Butcher_Tableus& coefficients, Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper>& solver, double dt) { // this is the main integrator function, all time stepping will be brought together within this function so once this is run, the next state will be populated

    Grid grid_initial = grid_imex;  
    divergence_checker_values divergence_test_0 = calculate_current_div_B(grid_imex);
    
    global_state_dot f1_storage(workspace.f1_rho, workspace.f1_m_x, workspace.f1_m_y, workspace.f1_b_x_f, workspace.f1_b_y_f, workspace.f1_E);
    explicit_rhs(grid_imex, f1_storage, MUSCL_values, workspace); 
    
    // explicit evaluation of initial magnetic resistive terms
    resistive_B_term(grid_imex, workspace.g1_b_x_f, workspace.g1_b_y_f, solver, workspace, false); 
    
    grid_intermediary = grid_initial;
    grid_whole_step = grid_initial;
    double diag_coeff = dt * coefficients.A_im[4]; 

    // ------------------------------------
    //          Intermediary Step
    // ------------------------------------

    double total_explicit_coeff_1 = dt * coefficients.A_ex[3];
    get_intermediary_grid(grid_imex, grid_intermediary.rho, workspace.f1_rho, total_explicit_coeff_1);
    get_intermediary_grid(grid_imex, grid_intermediary.mx, workspace.f1_m_x, total_explicit_coeff_1);
    get_intermediary_grid(grid_imex, grid_intermediary.my, workspace.f1_m_y, total_explicit_coeff_1);
    get_intermediary_grid(grid_imex, grid_intermediary.E, workspace.f1_E, total_explicit_coeff_1);
    get_intermediary_grid(grid_imex, grid_intermediary.bxf, workspace.f1_b_x_f, total_explicit_coeff_1);
    get_intermediary_grid(grid_imex, grid_intermediary.byf, workspace.f1_b_y_f, total_explicit_coeff_1);

    double total_implicit_coeff_1 = dt * coefficients.A_im[3];
    get_intermediary_grid(grid_imex, grid_intermediary.bxf, workspace.g1_b_x_f, total_implicit_coeff_1);
    get_intermediary_grid(grid_imex, grid_intermediary.byf, workspace.g1_b_y_f, total_implicit_coeff_1);

    grid_intermediary.fillGhostPeriodic();
    grid_intermediary.computePrimitives();

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

    double total_explicit_coeff_2 = dt * coefficients.A_ex[6];
    double total_explicit_coeff_3 = dt * coefficients.A_ex[7];
    double total_implicit_coeff_2 = dt * coefficients.A_im[6];
    double total_implicit_coeff_3 = dt * coefficients.A_im[7];

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
    
    divergence_checker_values divergence_test_2 = calculate_current_div_B(grid_whole_step);

    resistive_B_term(grid_whole_step, workspace.g3_b_x_f, workspace.g3_b_y_f, solver, workspace, false);

    // ------------------------------------
    //          FINAL COMBINATION 
    // ------------------------------------

    double f1_coeff = dt * coefficients.b_ex[0];
    double f2_coeff = dt * coefficients.b_ex[1];
    double g2_coeff = dt * coefficients.b_im[1];
    double g3_coeff = dt * coefficients.b_im[2];

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

void CFL_condition_checker(double dt, Grid& grid_checker) {
    
    double dx = grid_checker.cell_width;
    double dy = grid_checker.cell_height;
    double resistivity_value = grid_checker.resistivity;
    double maximum_dt_resistive = std::min(dx, dy) * std::min(dx, dy) / (4 * resistivity_value); // for resistive CFL limit
    double max_expected_speed = 3.0; 
    double maximum_dt_hyperbolic = 0.5 * std::min(dx, dy) / max_expected_speed; // for advective CFL limit
    double time_interval_limit = std::min(maximum_dt_hyperbolic, maximum_dt_resistive);

    if (time_interval_limit < dt) {

        std::cout << "Time step is too large reduce to at most " << time_interval_limit << ", otherwise will violate CFL condition, please lower" << "\n";
        std::exit(EXIT_FAILURE);
    }
}

void write_data_to_binary_file(Grid& grid_bin, int current_frame_index, int snapshot_rate, double time_step) {

    std::fstream binary_fout; 
    std::ostringstream ss;
    ss << "snapshot_" << std::setw(5) << std::setfill('0') << current_frame_index << ".dat";
    std::string filename = ss.str();
    binary_fout.open(filename, std::ios::out | std::ios::binary);

    if (!binary_fout.is_open()){
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

    // all data in global state written into a binary file sequentially
    binary_fout.write(reinterpret_cast<char*>(grid_bin.rho.data()), (grid_bin.rho.size() * sizeof(grid_bin.rho[0])));
    binary_fout.write(reinterpret_cast<char*>(grid_bin.E.data()),   (grid_bin.E.size()   * sizeof(grid_bin.E[0])));
    binary_fout.write(reinterpret_cast<char*>(grid_bin.p.data()),   (grid_bin.p.size()   * sizeof(grid_bin.p[0])));
    binary_fout.write(reinterpret_cast<char*>(grid_bin.vx.data()),  (grid_bin.vx.size()  * sizeof(grid_bin.vx[0])));
    binary_fout.write(reinterpret_cast<char*>(grid_bin.vy.data()),  (grid_bin.vy.size()  * sizeof(grid_bin.vy[0])));
    binary_fout.write(reinterpret_cast<char*>(grid_bin.bxc.data()), (grid_bin.bxc.size() * sizeof(grid_bin.bxc[0])));
    binary_fout.write(reinterpret_cast<char*>(grid_bin.byc.data()), (grid_bin.byc.size() * sizeof(grid_bin.byc[0])));

    binary_fout.close(); 
}

// functions below are for which test you want to run

void orszag_tang_initial_conditions(Grid& grid_ot, integrator_reserved_memory& mag_potential_storage) {

    // setup variables
    int grid_x_node_num = grid_ot.nxt;
    int grid_y_node_num = grid_ot.nyt;
    int ghost_cell_num = grid_ot.ng;
    double width_of_cell = grid_ot.cell_width;
    double height_of_cell = grid_ot.cell_height;
    int grid_node_total = grid_ot.nxt * grid_ot.nyt;

    std::vector<double> Az = mag_potential_storage.magnetic_potential;

    // constant values
    double rho =  25.0 / 9.0; // adiabatic index squared
    double p   =  5.0 / 3.0;  // adiabatic index (monotomic gas)

    for (int j = 0; j < grid_y_node_num; ++j) {
        for (int i = 0; i < grid_x_node_num; ++i) {

            // corner coordinates (bottom-left of cell i,j)
            double x_corner = (i - ghost_cell_num) * width_of_cell;
            double y_corner = (j - ghost_cell_num) * height_of_cell;

            // Orszag-Tang vector potential
            Az[grid_ot.indexC(i, j)] = 0.5 * cos(2 * x_corner) + cos(y_corner);

        }
    }

    for (int j = 0; j < grid_y_node_num; ++j) {
        for (int i = 0; i < grid_x_node_num; ++i) {
            
            int idx = grid_ot.indexC(i, j);
            int top = grid_ot.indexC(i, j + 1);
            int right = grid_ot.indexC(i + 1, j);
        
            double x_coordinate_of_current_node = (i - ghost_cell_num + 0.5) * width_of_cell;
            double y_coordinate_of_current_node = (j - ghost_cell_num + 0.5) * height_of_cell;

            double vx  = -sin(y_coordinate_of_current_node);
            double vy  =  sin(x_coordinate_of_current_node);
            double bx  =  (Az[top] - Az[idx]) / height_of_cell;
            double by  = -(Az[right] - Az[idx]) / width_of_cell;
        
            PrimitiveState w{rho, vx, vy, p, bx, by};
            ConservedState u = ConservedState::fromPrimitive(w);
        
            grid_ot.rho[idx] = u.rho;
            grid_ot.mx[idx]  = u.mx;
            grid_ot.my[idx]  = u.my;
            grid_ot.E[idx]   = u.E;
            grid_ot.bxf[idx] = u.bx;
            grid_ot.byf[idx] = u.by;
            grid_ot.p[idx]  = p;
            grid_ot.vx[idx] = vx;
            grid_ot.vy[idx] = vy;
        
        }
    }
}

void magnetic_wave_test_initial_values(Grid& grid_mw) {

    // setup variables
    int grid_x_node_num = grid_mw.nxt;
    int grid_y_node_num = grid_mw.nyt;
    int ghost_cell_num = grid_mw.ng;
    double width_of_cell = grid_mw.cell_width;
    double height_of_cell = grid_mw.cell_height;
    int grid_node_total = grid_mw.nxt * grid_mw.nyt;
    constexpr double pi = 3.14159265359;

    // constant values
    double amplitude_bx = 0.0;
    double amplitude_by = 1.0;
    double kx = 2.0 * pi / width_of_cell;
    double ky = 2.0 * pi / height_of_cell;
    double rho = 1.0;
    double vx  = 0.0;
    double vy  = 0.0;
    double p   = 1.0;

    for (int j = 0; j < grid_y_node_num; ++j) {
        for (int i = 0; i < grid_x_node_num; ++i) {
            
            int idx = grid_mw.indexC(i, j);
            double x_coordinate_of_current_node = (i - ghost_cell_num + 0.5) * width_of_cell;
            double y_coordinate_of_current_node = (j - ghost_cell_num + 0.5) * height_of_cell;

            // changing values
            double bx  = amplitude_bx * sin(ky * y_coordinate_of_current_node);
            double by  = amplitude_by * sin(kx * x_coordinate_of_current_node);

            PrimitiveState w{rho, vx, vy, p, bx, by};
            ConservedState u = ConservedState::fromPrimitive(w);
        
            grid_mw.rho[idx] = u.rho;
            grid_mw.mx[idx]  = u.mx;
            grid_mw.my[idx]  = u.my;
            grid_mw.E[idx]   = u.E;
            grid_mw.bxf[idx] = u.bx;
            grid_mw.byf[idx] = u.by;
            grid_mw.p[idx]  = p;
            grid_mw.vx[idx] = vx;
            grid_mw.vy[idx] = vy;

        }
    }
}

void gaussian_density_pulse_test_initial_values(Grid& grid_gauss) {

    // setup variables
    int physical_x_node_num = grid_gauss.nx;
    int physical_y_node_num = grid_gauss.ny;
    int grid_x_node_num = grid_gauss.nxt;
    int grid_y_node_num = grid_gauss.nyt;
    int ghost_cell_num = grid_gauss.ng;
    double width_of_cell = grid_gauss.cell_width;
    double height_of_cell = grid_gauss.cell_height;
    int grid_node_total = grid_gauss.nxt * grid_gauss.nyt;

    // setting up pulse size and location
    double x_mid = 0.5 * physical_x_node_num * grid_gauss.cell_width;
    double y_mid = 0.5 * physical_x_node_num * grid_gauss.cell_height;
    double pulse_radius = 10.0;
    double radius_sq = pulse_radius * pulse_radius;

    // constant values
    double vx  = 1.0;
    double vy  = 1.0;
    double p   = 1.0;
    double bx  = 0.0;
    double by  = 0.0;
    
    for (int j = 0; j < grid_y_node_num; ++j) {
        for (int i = 0; i < grid_x_node_num; ++i) {
            
            int idx = grid_gauss.indexC(i, j);
        
            // coordinates relative to center 
            double x_relative_to_centre = (i - grid_node_total + 0.5) * width_of_cell;
            double y_relative_to_centre = (j - grid_node_total + 0.5) * height_of_cell;
            double dist_sq = (x_relative_to_centre - x_mid) * (x_relative_to_centre - x_mid) + (y_relative_to_centre - y_mid) * (y_relative_to_centre - y_mid);

            // changing value
            double rho = 1.0 + exp(-dist_sq / radius_sq);
        
            PrimitiveState w{rho, vx, vy, p, bx, by};
            ConservedState u = ConservedState::fromPrimitive(w);

            grid_gauss.rho[idx] = u.rho;
            grid_gauss.mx[idx]  = u.mx;
            grid_gauss.my[idx]  = u.my;
            grid_gauss.E[idx]   = u.E;
            grid_gauss.bxf[idx] = u.bx;
            grid_gauss.byf[idx] = u.by;
            grid_gauss.p[idx]  = p;
            grid_gauss.vx[idx] = vx;
            grid_gauss.vy[idx] = vy;
        }
    }
}

void acoustic_wave_test_initial_values(Grid& grid_ac) {
    
    // setup variables
    int physical_x_node_num = grid_ac.nx;
    int physical_y_node_num = grid_ac.ny;
    int grid_x_node_num = grid_ac.nxt;
    int grid_y_node_num = grid_ac.nyt;
    int ghost_cell_num = grid_ac.ng;
    int grid_node_total = grid_ac.nxt * grid_ac.nyt;
    double width_of_cell = grid_ac.cell_width;
    double height_of_cell = grid_ac.cell_height;
    double total_physical_grid_width = width_of_cell * physical_x_node_num;
    double total_physical_grid_height = height_of_cell * physical_y_node_num;
    constexpr double pi = 3.14159265359;
    constexpr double adiabatic_index = 5.0 / 3.0;

    // constant values
    double amplitude_x = 0.1;
    double amplitude_y = 0.1;
    double bx  = 0.0;
    double by  = 0.0;

    for (int j = 0; j < grid_y_node_num; ++j) {
        for (int i = 0; i < grid_x_node_num; ++i) {
            
            int idx = grid_ac.indexC(i, j);

            double x_position = (i - ghost_cell_num) * width_of_cell;
            double y_position = (j - ghost_cell_num) * height_of_cell;
            double x_sin = amplitude_x * sin(2 * pi * x_position / total_physical_grid_width);
            double y_sin = amplitude_y * sin(2 * pi * y_position / total_physical_grid_height);

            // changing values
            double rho = 1.0 + x_sin + y_sin;
            double vx  = x_sin;
            double vy  = y_sin;
            double p   = (1 + adiabatic_index * (x_sin + y_sin)) / adiabatic_index;
        
            PrimitiveState w{rho, vx, vy, p, bx, by};
            ConservedState u = ConservedState::fromPrimitive(w);
        
            grid_ac.rho[idx] = u.rho;
            grid_ac.mx[idx]  = u.mx;
            grid_ac.my[idx]  = u.my;
            grid_ac.E[idx]   = u.E;
            grid_ac.bxf[idx] = u.bx;
            grid_ac.byf[idx] = u.by;
            grid_ac.p[idx]  = p;
            grid_ac.vx[idx] = vx;
            grid_ac.vy[idx] = vy;

        }
    }
}

void sausage_instability_initial_conditions(Grid& global_state) {

    const double adiabatic_index = 5.0 / 3.0; 
    const double background_B = 1.0;              
    const double background_rho = 1.0;            
    const double background_p = 1.0;              
    constexpr double pi = 3.14159265359;

    const double pertubation_amplitude = 0.15;    
    const double pertubation_wavenumber = 2.0 * pi;    
    const double radius_sq = 0.04;  // plasma pinch channel radius squared

    double x_mid = 0.5 * global_state.nxt * global_state.cell_width;

    for (int j = 0; j < global_state.nyt; ++j) {
        for (int i = 0; i < global_state.nxt; ++i) {
            
            int idx = global_state.indexC(i, j);

            // cell-centered spatial positions
            double x_position = (i - global_state.ng + 0.5) * global_state.cell_width;
            double y_position = (j - global_state.ng + 0.5) * global_state.cell_height;

            double r_dist = x_position - x_mid;

            // m=0 perturbation modulation along the channel axis (y-direction)
            double perturbation = 1.0 + pertubation_amplitude * std::cos(pertubation_wavenumber * y_position);

            // background floor of 0.001 to prevent 0/0 division
            double rho = background_rho * std::exp(-(r_dist * r_dist) / (radius_sq * perturbation)) + 0.001;
            double p   = background_p   * std::exp(-(r_dist * r_dist) / (radius_sq * perturbation)) + 0.001;
            double vx  = 0.0;
            double vy  = 0.0;
            double bx  = 0.0;
            double by  = background_B * (r_dist / std::sqrt(radius_sq)) * std::exp(-(r_dist * r_dist) / radius_sq);

            PrimitiveState w{rho, vx, vy, p, bx, by};
            ConservedState u = ConservedState::fromPrimitive(w);

            global_state.rho[idx] = u.rho;
            global_state.mx[idx]  = u.mx;
            global_state.my[idx]  = u.my;
            global_state.E[idx]   = u.E;
            global_state.bxf[idx] = bx;
            global_state.byf[idx] = by;
            global_state.p[idx]   = p;
            global_state.vx[idx]  = vx;
            global_state.vy[idx]  = vy;

        }
    }
}

void kink_instability_initial_conditions(Grid& global_state) {

    const double background_B   = 1.0;        
    const double background_rho = 1.0;      
    const double background_p   = 1.0;     
    constexpr double pi = 3.14159265359;   
    
    const double velocity_pertubation_amplitude = 0.05; // 5%  
    const double sinusoidal_wavenumber = 2.0 * pi; 
    const double radius_sq = 0.04; // channel width

    double x_mid = 0.5 * global_state.nxt * global_state.cell_width;

    for (int j = 0; j < global_state.nyt; ++j) {
        for (int i = 0; i < global_state.nxt; ++i) {
            
            int idx = global_state.indexC(i, j);

            // cell-centered spatial positions
            double x_position = (i - global_state.ng + 0.5) * global_state.cell_width;
            double y_position = (j - global_state.ng + 0.5) * global_state.cell_height;

            double r_dist = x_position - x_mid;

            double envelope = std::exp(-(r_dist * r_dist) / radius_sq);
            double rho = background_rho * envelope + 0.001; 
            double p   = background_p   * envelope + 0.001;

            // m=1 transverse velocity seed to trigger column displacement/buckling
            double vx = velocity_pertubation_amplitude * std::sin(sinusoidal_wavenumber * y_position) * envelope;
            double vy = 0.0;
            double bx = 0.0;
            double by = background_B * envelope;

            PrimitiveState w{rho, vx, vy, p, bx, by};
            ConservedState u = ConservedState::fromPrimitive(w);

            global_state.rho[idx] = u.rho;
            global_state.mx[idx]  = u.mx;
            global_state.my[idx]  = u.my;
            global_state.E[idx]   = u.E;
            global_state.bxf[idx] = bx;
            global_state.byf[idx] = by;
            global_state.p[idx]  = p;
            global_state.vx[idx] = vx;
            global_state.vy[idx] = vy;
        }
    }
}

int main(){

    auto start_time = std::chrono::high_resolution_clock::now();

    // create initial global state
    double animation_duration = 4.0;   // seconds
    const double time_interval = 0.0002;  // seconds
    int snapshot_frequency = 100;
    int x_node_count = 512;
    int y_node_count = 512;
    int ghost_node_count = 2;
    
    Grid global_state(x_node_count, y_node_count, ghost_node_count);   
    int node_total = global_state.nxt * global_state.nyt;
    constexpr double pi = 3.14159265359;
    global_state.cell_width = 2 * pi / x_node_count;
    global_state.cell_height = 2 * pi / y_node_count;
    global_state.resistivity = 0.0;

    CFL_condition_checker(time_interval, global_state);

    Grid intermediary_grid_1(x_node_count, y_node_count, ghost_node_count);
    Grid intermediary_grid_2(x_node_count, y_node_count, ghost_node_count);
    intermediary_grid_1 = global_state;
    intermediary_grid_2 = global_state;

    integrator_reserved_memory memory_storage(node_total, x_node_count * y_node_count, (global_state.nxt + 1) * (global_state.nyt + 1));
    ReconstructedValues MUSCL_memory(node_total);
    IMEX_Butcher_Tableus coefficients_tableu;
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper> matrix_solver;

    double matrix_coeff = time_interval * global_state.resistivity;
    generate_A_matrix(global_state, memory_storage, matrix_coeff * coefficients_tableu.A_im[4]);
    matrix_solver.compute(memory_storage.A_matrix);

    // -------------------------------------
    // --- CHOOSE WHICH TEST TO RUN HERE ---
    // -------------------------------------

    orszag_tang_initial_conditions(global_state, memory_storage);
    //sausage_instability_initial_conditions(global_state);
    //kink_instability_initial_conditions(global_state);
    //magnetic_wave_test_initial_values(global_state);
    //gaussian_density_pulse_test_initial_values(global_state);
    //acoustic_wave_test_initial_values(global_state);

    global_state.fillGhostPeriodic();
    global_state.computePrimitives();
    
    int frame_index = 0; 
    int frame_max = animation_duration / (time_interval * snapshot_frequency) ;

    // --- initial snapshot ---

    global_state.computePrimitives(); 
    divergence_checker_values div_B_values = calculate_current_div_B(global_state); 

    write_data_to_binary_file(global_state, frame_index, snapshot_frequency, time_interval);

    // --- successive snapshots ---

    for (double _ = 0; _ <= animation_duration; _ += time_interval) {

        frame_index ++;

        IMEX(global_state, intermediary_grid_1, intermediary_grid_2, memory_storage, MUSCL_memory, coefficients_tableu, matrix_solver, time_interval);  

        if (frame_index % snapshot_frequency == 0){

            std::cout << "Processing frame " << frame_index / snapshot_frequency << " / " << frame_max << "..." << "\n";

            global_state.fillGhostPeriodic();
            global_state.computePrimitives();      
            write_data_to_binary_file(global_state, frame_index, snapshot_frequency, time_interval);

        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto runtime_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    auto runtime_seconds = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    std::cout << "Total runtime: " << runtime_seconds.count() << " seconds, " << (runtime_milliseconds.count() % 1000) << " milliseconds" << "\n";
    
    return 0;
    
}
