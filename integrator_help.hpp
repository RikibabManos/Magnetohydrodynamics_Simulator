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

using namespace std;

struct global_state_dot { // structure that holds addresses of all the derivative values of important variables, specificaly used to transfer arrays from one reserve to another 

    vector<double>* rho_dot = nullptr; // density derivative
    vector<double>* m_dot_x = nullptr; // momentum density derivative in x direction
    vector<double>* m_dot_y = nullptr; // momentum density derivative in y direction
    vector<double>* b_dot_x = nullptr; // magnetic field strength derivative in x direction
    vector<double>* b_dot_y = nullptr; // magnetic field strength derivative in y direction
    vector<double>* E_dot = nullptr;   // total energy density derivative

    global_state_dot() = default;

    // constructor allocate addresses to pointers
    global_state_dot(vector<double>& rho, vector<double>& m_x, vector<double>& m_y, vector<double>& b_x, vector<double>& b_y, vector<double>& E)
    :rho_dot(&rho),
     m_dot_x(&m_x),
     m_dot_y(&m_y),
     b_dot_x(&b_x),
     b_dot_y(&b_y),
     E_dot(&E)
    {}
};

/// @brief struct that holds the Butcher Tableu coefficients for use in the IMEX integrator, tableus based off the ARS(2,2,2) scheme
struct IMEX_Butcher_Tableus{ // based off the ARS(2,2,2) scheme
    
    // defining useful constants
    const double gamma = (2.0 - sqrt(2.0)) / 2.0; 
    const double delta = 1.0 - (1.0 / (2.0 * gamma));

    int num_stages = 3;          // number of mini stages before final derivative found
    vector<double> A_ex = {      // explicit Butcher Table (note in collapsed 1D form)

        0.0, 0.0, 0.0,
        gamma, 0.0, 0.0,
        delta, 1.0 - delta, 0.0

    };

    vector<double> c_ex = {0.0, gamma, 1.0};         // time steps for explicit part
    vector<double> b_ex = {delta, 1.0 - delta, 0.0}; // final weight of each explicit step

    vector<double> A_im = { // implicit Butcher Table (note in collapsed 1D form)

        0.0, 0.0, 0.0,
        0.0, gamma, 0.0,
        0.0, 1.0 - gamma, gamma

    };

    vector<double> c_im = {0.0, gamma, 1.0};         // time steps for implicit part
    vector<double> b_im = {0.0, 1.0 - gamma, gamma}; // final weight of each implicit step

};

struct integrator_reserved_memory{ // 'scratchpad', allocates memory on stack once for IMEX which can be written on/ cleared each cycle 

    // initialise vectors for each variable
    // f1, f2 hold the non-'stiff' derivative rates for the respective mini step for the explicit integrator part (no f3 needed as no 3rd column dependance; see explicit Butcher tableu)
    // g1, g2, g3 hold the 'stiff' derivative rates for each implicit mini step
    // 1 --> 3 beacuse ARS(2,2,2) has 3 steps
    // rhs holds final weighted sum of all g and f lists, to be used in the implicit integrator calculation

    // mass density
    vector<double> f1_rho;
    vector<double> f2_rho;    
    vector<double> g1_rho;
    vector<double> g2_rho;
    vector<double> g3_rho;

    // momentum density in x direction

    vector<double> f1_m_x;
    vector<double> f2_m_x;
    vector<double> g1_m_x;
    vector<double> g2_m_x;
    vector<double> g3_m_x;
 
    // momentum density in y direction

    vector<double>  f1_m_y;
    vector<double>  f2_m_y;
    vector<double>  g1_m_y;
    vector<double>  g2_m_y;    
    vector<double>  g3_m_y;    

    // total energy density 

    vector<double> f1_E;
    vector<double> f2_E;
    vector<double> g1_E;   
    vector<double> g2_E;
    vector<double> g3_E;

    // cell centred B field in x direction

    vector<double> f1_b_x_f;
    vector<double> f2_b_x_f;
    vector<double> g1_b_x_f;
    vector<double> g2_b_x_f;    
    vector<double> g3_b_x_f;

    // cell centred B field in y direction

    vector<double> f1_b_y_f;
    vector<double> f2_b_y_f;
    vector<double> g1_b_y_f;
    vector<double> g2_b_y_f;
    vector<double> g3_b_y_f;

    // flux values for each direction

    // mass density fluxs
    vector<double> rho_flux_horizontal;
    vector<double> rho_flux_vertical;

    // momentum density flux in x direction
    vector<double> m_x_flux_horizontal;     
    vector<double> m_x_flux_vertical;

    // momentum density flux in y direction
    vector<double> m_y_flux_horizontal;
    vector<double> m_y_flux_vertical;

    // total Energy flux
    vector<double> E_flux_horizontal;    
    vector<double> E_flux_vertical;  
    
    // B field flux in x direction
    vector<double> b_x_flux_horizontal;
    vector<double> b_x_flux_vertical;

    // B field flux in y direction
    vector<double> b_y_flux_horizontal;
    vector<double> b_y_flux_vertical;

    // stiff values
    vector<double> b_x_laplacian;
    vector<double> b_y_laplacian;

    // matrix storage (Ax = b)
    Eigen::SparseMatrix<double> A_matrix;

    // no ghost padding vectors
    vector<double> b_x_f_no_padding;
    vector<double> b_y_f_no_padding;

    // for electric field
    vector<double> electric_field_at_corners;


    integrator_reserved_memory(int cell_total, int cell_total_no_ghost) // constructor which allocates required memory on heap once based off number of nodes
    
    // initialise variables
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

     A_matrix(cell_total_no_ghost * cell_total_no_ghost, cell_total_no_ghost * cell_total_no_ghost),
    
     b_x_f_no_padding(cell_total_no_ghost), b_y_f_no_padding(cell_total_no_ghost),

     electric_field_at_corners(cell_total, 0.0)
    {}
};

inline double get_partial_derivative(double right_top_quantity, double left_bottom_quantity, double length) { // helper function to calculate the partial derivative of a quantity based off input values
    // right_top_quantity refers to either the right or top faces of a cell, left_bottom_quantity refers to left or bottom
    
    return (right_top_quantity - left_bottom_quantity) / length;

}

inline double get_dot_prod(double vec1_x, double vec1_y, double vec2_x, double vec2_y){ // helper function to return 2D dot product of 2 vectors based off components
    return (vec1_x * vec2_x) + (vec1_y * vec2_y);
}

inline void get_intermediary_grid(vector<double>& grid, const vector<double>& list, double scalar){ // helper function to compute the new grid vector states 
    // lists must be equal length --- REDO LATER TO CHECK THIS ---
    int list_length = grid.size();

    for (int i = 0; i < list_length; i++){

        grid[i] += list[i] * scalar;

    }

}

inline void generate_cell_properties(PrimitiveState& left_face, PrimitiveState& right_face, PrimitiveState& top_face, PrimitiveState& bottom_face, Grid& grid_cell_properties, ReconstructedValues& reconstructed_val, int i_value, int j_value){

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
            bottom_face.by = grid_cell_properties.byf[bottom_cell_index];
}