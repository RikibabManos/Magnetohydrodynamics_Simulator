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
#include "Grid.hpp"
#include "MUSCL.hpp"
#include "HLL.hpp"
#include "integrator_help.hpp"

using namespace std;

void get_divergences(integrator_reserved_memory& memory,  Grid& grid_div, global_state_dot& dU_explicit_rhs_memory) { // helper function to calculate the divergence of momentum density, mass density, B field and total energy density for all nodes, at the centre of the cell
    
    // assigning usful values
    int total_x_nodes = grid_div.nx;
    int total_y_nodes = grid_div.ny;

    int j_index_initial = grid_div.gj(-1);
    int j_index_final = grid_div.gj(total_y_nodes);
    int i_index_initial = grid_div.gi(-1);
    int i_index_final = grid_div.gi(total_x_nodes);
    
    double cell_width = grid_div.cell_width;
    double cell_height = grid_div.cell_height; 

    // main assignment of divergences loop
    for (int j = j_index_initial; j <= j_index_final; j++){
        for (int i = i_index_initial; i <= i_index_final; i++){

            int current_node_index = grid_div.indexC(i, j);
            int left_node_index = grid_div.indexC(i - 1, j);
            int bottom_node_index = grid_div.indexC(i, j - 1);
            
            // update scrapboad memory
            dU_explicit_rhs_memory.rho_dot[current_node_index] = -( get_partial_derivative(memory.rho_flux_horizontal[current_node_index], memory.rho_flux_horizontal[left_node_index], cell_width) + get_partial_derivative(memory.rho_flux_vertical[current_node_index], memory.rho_flux_vertical[bottom_node_index], cell_height) );
            dU_explicit_rhs_memory.E_dot[current_node_index] = -( get_partial_derivative(memory.E_flux_horizontal[current_node_index], memory.E_flux_horizontal[left_node_index], cell_width) + get_partial_derivative(memory.E_flux_vertical[current_node_index], memory.E_flux_vertical[bottom_node_index], cell_height) );
            dU_explicit_rhs_memory.m_dot_x[current_node_index] = -( get_partial_derivative(memory.m_x_flux_horizontal[current_node_index], memory.m_x_flux_horizontal[left_node_index], cell_width) + get_partial_derivative(memory.m_x_flux_vertical[current_node_index], memory.m_x_flux_vertical[bottom_node_index], cell_height) );
            dU_explicit_rhs_memory.m_dot_y[current_node_index] = -( get_partial_derivative(memory.m_y_flux_horizontal[current_node_index], memory.m_y_flux_horizontal[left_node_index], cell_width) + get_partial_derivative(memory.m_y_flux_vertical[current_node_index], memory.m_y_flux_vertical[bottom_node_index], cell_height) );
            dU_explicit_rhs_memory.b_dot_x[current_node_index] = -( get_partial_derivative(memory.b_x_flux_horizontal[current_node_index], memory.b_x_flux_horizontal[left_node_index], cell_width) + get_partial_derivative(memory.b_x_flux_vertical[current_node_index], memory.b_x_flux_vertical[bottom_node_index], cell_height) );
            dU_explicit_rhs_memory.b_dot_y[current_node_index] = -( get_partial_derivative(memory.b_y_flux_horizontal[current_node_index], memory.b_y_flux_horizontal[left_node_index], cell_width) + get_partial_derivative(memory.b_y_flux_vertical[current_node_index], memory.b_y_flux_vertical[bottom_node_index], cell_height) );

        }
    }
    

}

void flux_population(Grid& grid_flux, ReconstructedValues& RV, integrator_reserved_memory& reserve_flux){ 
    // --- NOTE TO SELF, CHANGE TO USE POINTERS IN FUTURE FOR SPEED --- 
    // function that populates each node with its flux values and stores in designated integrator_memory_reserve vector
    
    // define structs for single node value storage
    PrimitiveState left_face;
    PrimitiveState right_face;
    PrimitiveState top_face;
    PrimitiveState bottom_face;

    int total_x_nodes = grid_flux.nx;
    int total_y_nodes = grid_flux.ny;

    int j_index_start = grid_flux.gj(-1);
    int j_index_end = grid_flux.gj(total_y_nodes);
    int i_index_start = grid_flux.gi(-1);
    int i_index_end = grid_flux.gi(total_x_nodes);

    for (int j = j_index_start; j <= j_index_end; j++){    
        for (int i = i_index_start; i <= i_index_end; i++){

            int current_cell_index = grid_flux.indexC(i, j);   
            int right_cell_index = grid_flux.indexC(i + 1, j); 
            int top_cell_index = grid_flux.indexC(i, j + 1);   

            // populate PrimitiveSate struct so we can use as inputs to the computeHLLFlux functions
            left_face.rho = RV.rhoXFaceR[current_cell_index];
            left_face.vx = RV.vxXFaceR[current_cell_index];
            left_face.vy = RV.vyXFaceR[current_cell_index];
            left_face.p = RV.pXFaceR[current_cell_index];
            left_face.bx = grid_flux.bxf[right_cell_index];
            left_face.by = RV.byXFaceR[current_cell_index];

            right_face.rho = RV.rhoXFaceL[right_cell_index];
            right_face.vx = RV.vxXFaceL[right_cell_index];
            right_face.vy = RV.vyXFaceL[right_cell_index];
            right_face.p = RV.pXFaceL[right_cell_index];
            right_face.bx = grid_flux.bxf[right_cell_index];
            right_face.by = RV.byXFaceL[right_cell_index];
            
            top_face.rho = RV.rhoYFaceL[top_cell_index];
            top_face.vx = RV.vxYFaceL[top_cell_index];
            top_face.vy = RV.vyYFaceL[top_cell_index];
            top_face.p = RV.pYFaceL[top_cell_index];
            top_face.bx = RV.bxYFaceL[top_cell_index];
            top_face.by = grid_flux.byf[top_cell_index];

            bottom_face.rho = RV.rhoYFaceR[current_cell_index];
            bottom_face.vx = RV.vxYFaceR[current_cell_index];
            bottom_face.vy = RV.vyYFaceR[current_cell_index];
            bottom_face.p = RV.pYFaceR[current_cell_index];
            bottom_face.bx = RV.bxYFaceR[current_cell_index];
            bottom_face.by = grid_flux.byf[current_cell_index];

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

//void implicit_rhs(Grid& grid_im, global_state_dot& dU_implicit, ReconstructedValues& MUSCL_im_output, integrator_reserved_memory& implicit_reserve){ // calculates the implicit rhs values at a given state (stiff terms)
    // --- REQUIRES explicit_rhs FUNCTION TO BE RUN BEFORE TO WORK ---
    
//}


void IMEX(Grid& grid_imex, Grid& grid_intermediary_1, Grid& grid_intermediary_2, integrator_reserved_memory& workspace, global_state_dot& dU_explicit_rhs, ReconstructedValues& MUSCL_values, IMEX_Butcher_Tableus& coefficients, double dt){ // function that calculates the new global state of the system using a combination of implicit and explicit methods

    double intermediary_time_step = dt * coefficients.gamma;

    explicit_rhs(grid_imex, dU_explicit_rhs, MUSCL_values, workspace); // calculate the explicit derivative terms at the initial state
    
    // asign the output of the first explicit step to f1 memory reserve

    workspace.f1_rho = dU_explicit_rhs.rho_dot;
    workspace.f1_m_x = dU_explicit_rhs.m_dot_x;
    workspace.f1_m_y = dU_explicit_rhs.m_dot_y;
    workspace.f1_E = dU_explicit_rhs.E_dot;
    workspace.f1_b_x_c = dU_explicit_rhs.b_dot_x;
    workspace.f1_b_y_c = dU_explicit_rhs.b_dot_y;

    // calculate psuedo grid state due to explicit rhs only

    double total_explicit_coeff_1 = intermediary_time_step * coefficients.A_ex[3];
    get_intermediary_grid(grid_intermediary_1.rho, workspace.f1_rho, total_explicit_coeff_1);
    get_intermediary_grid(grid_intermediary_1.mx, workspace.f1_m_x, total_explicit_coeff_1);
    get_intermediary_grid(grid_intermediary_1.my, workspace.f1_m_y, total_explicit_coeff_1);
    get_intermediary_grid(grid_intermediary_1.E, workspace.f1_E, total_explicit_coeff_1);
    get_intermediary_grid(grid_intermediary_1.bxc, workspace.f1_b_x_c, total_explicit_coeff_1);
    get_intermediary_grid(grid_intermediary_1.byc, workspace.f1_b_y_c, total_explicit_coeff_1);

    grid_intermediary_1.fillGhostPeriodic();
    grid_intermediary_1.computePrimitives();
    
    // --- IMPLICIT STUFF 1 ---

    // calculate final grid state due to explicit and implicit rhs contributions

    //double total_implicit_coeff_1 = intermediary_time_step * coefficients.A_im[3];
    //get_intermediary_grid(grid_intermediary_1.rho, workspace.g1_rho, total_implicit_coeff_1);
    //get_intermediary_grid(grid_intermediary_1.mx, workspace.g1_m_x, total_implicit_coeff_1);
    //get_intermediary_grid(grid_intermediary_1.my, workspace.g1_m_y, total_implicit_coeff_1);
    //get_intermediary_grid(grid_intermediary_1.E, workspace.g1_E, total_implicit_coeff_1);
    //get_intermediary_grid(grid_intermediary_1.bxc, workspace.g1_b_x_c, total_implicit_coeff_1);
    //get_intermediary_grid(grid_intermediary_1.byc, workspace.g1_b_y_c, total_implicit_coeff_1);

    //grid_intermediary_1.fillGhostPeriodic();
    //grid_intermediary_1.computePrimitives();

    // now use the calculated rhs to find intermidiary step explicit derivative terms

    explicit_rhs(grid_intermediary_1, dU_explicit_rhs, MUSCL_values, workspace);
        
    // asign the output of the second explicit step to f2 memory reserve

    workspace.f2_rho = dU_explicit_rhs.rho_dot;
    workspace.f2_m_x = dU_explicit_rhs.m_dot_x;
    workspace.f2_m_y = dU_explicit_rhs.m_dot_y;
    workspace.f2_E = dU_explicit_rhs.E_dot;
    workspace.f2_b_x_c = dU_explicit_rhs.b_dot_x;
    workspace.f2_b_y_c = dU_explicit_rhs.b_dot_y;

    // calculate psuedo grid state due to explicit rhs only

    double total_explicit_coeff_2 = intermediary_time_step * coefficients.A_ex[6];
    get_intermediary_grid(grid_intermediary_2.rho, workspace.f2_rho, total_explicit_coeff_2);
    get_intermediary_grid(grid_intermediary_2.mx, workspace.f2_m_x, total_explicit_coeff_2);
    get_intermediary_grid(grid_intermediary_2.my, workspace.f2_m_y, total_explicit_coeff_2);
    get_intermediary_grid(grid_intermediary_2.E, workspace.f2_E, total_explicit_coeff_2);
    get_intermediary_grid(grid_intermediary_2.bxc, workspace.f2_b_x_c, total_explicit_coeff_2);
    get_intermediary_grid(grid_intermediary_2.byc, workspace.f2_b_y_c, total_explicit_coeff_2);

    grid_intermediary_2.fillGhostPeriodic();
    grid_intermediary_2.computePrimitives();

    // --- IMPLICIT STUFF 2 ---

    // calculate final grid state due to explicit and implicit rhs contributions

    //double total_implicit_coeff_2 = intermediary_time_step * coefficients.A_im[6];
    //get_intermediary_grid(grid_intermediary_2.rho, workspace.g2_rho, total_implicit_coeff_2);
    //get_intermediary_grid(grid_intermediary_2.mx, workspace.g2_m_x, total_implicit_coeff_2);
    //get_intermediary_grid(grid_intermediary_2.my, workspace.g2_m_y, total_implicit_coeff_2);
    //get_intermediary_grid(grid_intermediary_2.E, workspace.g2_E, total_implicit_coeff_2);
    //get_intermediary_grid(grid_intermediary_2.bxc, workspace.g2_b_x_c, total_implicit_coeff_2);
    //get_intermediary_grid(grid_intermediary_2.byc, workspace.g2_b_y_c, total_implicit_coeff_2);

    //grid_intermediary_2.fillGhostPeriodic();
    //grid_intermediary_2.computePrimitives();

    // now compute the new state as a weighted sum of the initial and intermidiary state contributions

    double f1_coeff = coefficients.b_ex[0];
    double f2_coeff = coefficients.b_ex[1];
    double g2_coeff = coefficients.b_im[1];
    double g3_coeff = coefficients.b_im[2];
    
    get_intermediary_grid(grid_imex.rho, workspace.f1_rho, intermediary_time_step * f1_coeff);
    get_intermediary_grid(grid_imex.mx, workspace.f1_m_x, intermediary_time_step * f1_coeff);
    get_intermediary_grid(grid_imex.my, workspace.f1_m_y, intermediary_time_step * f1_coeff);
    get_intermediary_grid(grid_imex.E, workspace.f1_E, intermediary_time_step * f1_coeff);
    get_intermediary_grid(grid_imex.bxc, workspace.f1_b_x_c, intermediary_time_step * f1_coeff);
    get_intermediary_grid(grid_imex.byc, workspace.f1_b_y_c, intermediary_time_step * f1_coeff);

    get_intermediary_grid(grid_imex.rho, workspace.f2_rho, intermediary_time_step * f2_coeff);
    get_intermediary_grid(grid_imex.mx, workspace.f2_m_x, intermediary_time_step * f2_coeff);
    get_intermediary_grid(grid_imex.my, workspace.f2_m_y, intermediary_time_step * f2_coeff);
    get_intermediary_grid(grid_imex.E, workspace.f2_E, intermediary_time_step * f2_coeff);
    get_intermediary_grid(grid_imex.bxc, workspace.f2_b_x_c, intermediary_time_step * f2_coeff);
    get_intermediary_grid(grid_imex.byc, workspace.f2_b_y_c, intermediary_time_step * f2_coeff);

    //get_intermediary_grid(grid_imex.rho, workspace.g2_rho, intermediary_time_step * g2_coeff);
    //get_intermediary_grid(grid_imex.mx, workspace.g2_m_x, intermediary_time_step * g2_coeff);
    //get_intermediary_grid(grid_imex.my, workspace.g2_m_y, intermediary_time_step * g2_coeff);
    //get_intermediary_grid(grid_imex.E, workspace.g2_E, intermediary_time_step * g2_coeff);
    //get_intermediary_grid(grid_imex.bxc, workspace.g2_b_x_c, intermediary_time_step * g2_coeff);
    //get_intermediary_grid(grid_imex.byc, workspace.g2_b_y_c, intermediary_time_step * g2_coeff);
    //
    //// this holds the new state values for every node
    //get_intermediary_grid(grid_imex.rho, workspace.g3_rho, intermediary_time_step * g3_coeff);
    //get_intermediary_grid(grid_imex.mx, workspace.g3_m_x, intermediary_time_step * g3_coeff);
    //get_intermediary_grid(grid_imex.my, workspace.g3_m_y, intermediary_time_step * g3_coeff);
    //get_intermediary_grid(grid_imex.E, workspace.g3_E, intermediary_time_step * g3_coeff);
    //get_intermediary_grid(grid_imex.bxc, workspace.g3_b_x_c, intermediary_time_step * g3_coeff);
    //get_intermediary_grid(grid_imex.byc, workspace.g3_b_y_c, intermediary_time_step * g3_coeff);
    
    grid_imex.fillGhostPeriodic();
    grid_imex.computePrimitives();
    


}

int main(){

    // create initial global state
    double animation_duration = 300; // seconds
    double time_interval = 0.1;     // seconds
    int x_node_count = 100;
    int y_node_count = 100;
    int ghost_node_count = 2;
    int node_total = (x_node_count + ghost_node_count) * (y_node_count + ghost_node_count);

    Grid global_state(x_node_count, y_node_count, 2);   
    global_state_dot dU_rhs(node_total);

    global_state.cell_width = 10;
    global_state.cell_height = 10;
    Grid intermediary_grid_1(x_node_count, y_node_count, 2);
    Grid intermidiary_grid_2(x_node_count, y_node_count, 2);
    intermediary_grid_1 = global_state;
    intermidiary_grid_2 = global_state;

    global_state.fillGhostPeriodic();
    intermediary_grid_1.fillGhostPeriodic(); 
    intermidiary_grid_2.fillGhostPeriodic(); 

    // initialise helper structs
    integrator_reserved_memory memory_storage(node_total);
    ReconstructedValues MUSCL_memory(node_total);
    IMEX_Butcher_Tableus coefficients_tableu;

    int frame_index = 0; 

    for (double _ = 0; _ <= animation_duration; _ += time_interval){

        frame_index ++;

        IMEX(global_state, intermediary_grid_1, intermidiary_grid_2, memory_storage, dU_rhs, MUSCL_memory, coefficients_tableu, time_interval);

        if (frame_index % 100 == 0){

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


}