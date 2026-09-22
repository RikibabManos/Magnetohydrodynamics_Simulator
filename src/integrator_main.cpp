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
#include "solvers.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "diagnostics.hpp"
#include "benchmarks.hpp"

int main() {

    auto start_time = std::chrono::high_resolution_clock::now();

    // --- MAIN TWEAKABLE PROPERTIES ---
    const double animation_duration = 5.0; // seconds
    const double time_interval_override = 0.0; // seconds, make this != 0.0 if you would like to specify a specific, constant time-step, otherwise when 0.0 the code will utilise a dynamic time step
    const double resistivity = 0.0; // units: grid_length^2 * seconds^-1. note that the speed of the code will significantly decrease if resitivity is non-zero AND time step is variable
    const int snapshot_frequency = 20; // the code will only write binary file data every 'snapshot_frequency' iterations, smaller values lead to a higher resolution, but use more memory (more .dat files)
    const int x_node_count = 200; // 200x200 grid size is a good balance between grid resolution and short run time, you  can change if you wish
    const int y_node_count = 200;
    // ----------------------------------
    
    const int ghost_node_count = 2; 
    Grid global_state(x_node_count, y_node_count, ghost_node_count);   
    const int node_total = global_state.nxt * global_state.nyt;
    constexpr double pi = 3.14159265359;
    global_state.cell_width = 2 * pi / x_node_count; // you could also alter this, but not recommended
    global_state.cell_height = 2 * pi / y_node_count;
    global_state.resistivity = resistivity; 

    Grid intermediary_grid_1(x_node_count, y_node_count, ghost_node_count);
    Grid intermediary_grid_2(x_node_count, y_node_count, ghost_node_count);
    intermediary_grid_1 = global_state;
    intermediary_grid_2 = global_state;
    integrator_reserved_memory memory_storage(node_total, x_node_count * y_node_count, (global_state.nxt + 1) * (global_state.nyt + 1));
    ReconstructedValues MUSCL_memory(node_total);
    IMEX_Butcher_Tableus coefficients_tableu;
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper> matrix_solver;

    test_selection(global_state, memory_storage);

    global_state.fillGhostPeriodic();
    global_state.computePrimitives();

    double optimal_time_interval = CFL_condition_checker(global_state); 
    double time_interval = (time_interval_override == 0.0) ? optimal_time_interval : time_interval_override;
    if (time_interval > optimal_time_interval) {std::cout << "Maximum safe time interval is: " << optimal_time_interval << ". Code may crash if entered time interval is not reduced!" << "\n";}
    double matrix_coeff = time_interval * global_state.resistivity;
    generate_A_matrix(global_state, memory_storage, matrix_coeff * coefficients_tableu.A_im[4]);
    matrix_solver.compute(memory_storage.A_matrix);
    
    int frame_index = 0; 
    double current_sim_time = 0.0;

    // --- initial snapshot ---

    global_state.computePrimitives(); 
    divergence_checker_values div_B_values = calculate_current_div_B(global_state); 

    write_data_to_binary_file(global_state, frame_index, snapshot_frequency, time_interval);

    // --- successive snapshots ---

    double previous_time_step = time_interval; // initialise previous time step to avoid matrix regeneration on initial loop

    while (current_sim_time < animation_duration) {

        // uncomment the line below if you would like to see if the code is still working (if process seems to run slowly, or for whatever reason)
        //std::cout << "I'm still working!" << "\n";
        double current_time_step = (time_interval_override == 0.0) ? CFL_condition_checker(global_state) : time_interval_override; 
        
        if (current_sim_time + current_time_step > animation_duration) {current_time_step = animation_duration - current_sim_time;}
        
        frame_index ++;

        if (global_state.resistivity > 0.0) { 
            if (std::abs(current_time_step - previous_time_step) > 1e-10 * current_time_step) { // only triggers a new matrix generated if time step changes by 1e-8%
                             
                double matrix_coeff = current_time_step * global_state.resistivity;
                generate_A_matrix(global_state, memory_storage, matrix_coeff * coefficients_tableu.A_im[4]);
                matrix_solver.compute(memory_storage.A_matrix);

            }
        }

        previous_time_step = current_time_step;

        IMEX(global_state, intermediary_grid_1, intermediary_grid_2, memory_storage, MUSCL_memory, coefficients_tableu, matrix_solver, current_time_step);  
        current_sim_time += current_time_step;

        if (frame_index % snapshot_frequency == 0){

            if (current_time_step > optimal_time_interval) {std::cout << "Current maximum safe time interval is: " << optimal_time_interval << ". Code may crash if entered time interval is not reduced!" << "\n";}
            std::cout << static_cast<int>(current_sim_time * 100.0 / animation_duration) << "% done..." << "\n";

            global_state.fillGhostPeriodic();
            global_state.computePrimitives();      
            write_data_to_binary_file(global_state, frame_index, snapshot_frequency, current_time_step);

        }
    }

    std::cout << "Simulation Finished!" << "\n";
    auto end_time = std::chrono::high_resolution_clock::now();
    auto runtime_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    auto runtime_seconds = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    std::cout << "Total runtime: " << runtime_seconds.count() << " seconds, " << (runtime_milliseconds.count() % 1000) << " milliseconds";
    
    return 0;
    
}
