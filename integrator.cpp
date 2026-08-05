#pragma once
#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include "Grid.hpp"

using namespace std;

struct global_state_dot { // structure that holds all the derivative values of important variables, note that the 

    vector<double> rho_dot; // density derivative
    vector<double> m_dot_x; // momentum density derivative in x direction
    vector<double> m_dot_y; // momentum density derivative in y direction
    vector<double> b_dot_x; // magnetic field strength derivative in x direction
    vector<double> b_dot_y; // magnetic field strength derivative in y direction
    vector<double> E_dot;   // total energy density derivative

    // constructor to reserve memory for vectors when called based on input
    global_state_dot(int cell_total = 0)
    :rho_dot(cell_total, 0.0),
     m_dot_x(cell_total, 0.0),
     m_dot_y(cell_total, 0.0),
     b_dot_x(cell_total, 0.0),
     b_dot_y(cell_total, 0.0),
     E_dot(cell_total, 0.0)
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
    // grid size
    int size;

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
    vector<double> rhs_rho;

    // momentum density in x direction

    vector<double> f1_m_x;
    vector<double> f2_m_x;
    vector<double> g1_m_x;
    vector<double> g2_m_x;
    vector<double> g3_m_x;
    vector<double> rhs_m_x;
 
    // momentum density in y direction

    vector<double>  f1_m_y;
    vector<double>  f2_m_y;
    vector<double>  g1_m_y;
    vector<double>  g2_m_y;    
    vector<double>  g3_m_y;    
    vector<double>  rhs_m_y;

    // total energy density 

    vector<double> f1_E;
    vector<double> f2_E;
    vector<double> g1_E;   
    vector<double> g2_E;
    vector<double> g3_E;
    vector<double> rhs_E;

    // cell centred B field in x direction

    vector<double> f1_b_x_c;
    vector<double> f2_b_x_c;
    vector<double> g1_b_x_c;
    vector<double> g2_b_x_c;    
    vector<double> g3_b_x_c;
    vector<double> rhs_b_x_c;

    // cell centred B field in y direction

    vector<double> f1_b_y_c;
    vector<double> f2_b_y_c;
    vector<double> g1_b_y_c;
    vector<double> g2_b_y_c;
    vector<double> g3_b_y_c;
    vector<double> rhs_b_y_c;

    // divergence values (used in get_divergence helper function)

    vector<double> rho_div; // rho rhs divergence
    vector<double> E_ex_div; // energy conservation explicit rhs divergence
    vector<double> m_div_x; // momentum density rhs divergence x component
    vector<double> m_div_y; // momentum density rhs divergence y component
    vector<double> b_rhs_div_x; // B field rhs explicit term in 
    vector<double> b_rhs_div_y; // B field rhs explicit term in 


    integrator_reserved_memory(int cell_total) // constructor which allocates required memory on heap once based off number of nodes
    
    // initialise variables
    :size(cell_total),
     f1_rho(cell_total, 0.0), f2_rho(cell_total, 0.0), g1_rho(cell_total, 0.0), g2_rho(cell_total, 0.0), g3_rho(cell_total, 0.0), rhs_rho(cell_total, 0.0),
     f1_m_x(cell_total, 0.0), f2_m_x(cell_total, 0.0), g1_m_x(cell_total, 0.0), g2_m_x(cell_total, 0.0), g3_m_x(cell_total, 0.0), rhs_m_x(cell_total, 0.0),     
     f1_m_y(cell_total, 0.0), f2_m_y(cell_total, 0.0), g1_m_y(cell_total, 0.0), g2_m_y(cell_total, 0.0), g3_m_y(cell_total, 0.0), rhs_m_y(cell_total, 0.0),
     f1_E(cell_total, 0.0), f2_E(cell_total, 0.0), g1_E(cell_total, 0.0), g2_E(cell_total, 0.0), g3_E(cell_total, 0.0), rhs_E(cell_total, 0.0),
     f1_b_x_c(cell_total, 0.0), f2_b_x_c(cell_total, 0.0), g1_b_x_c(cell_total, 0.0), g2_b_x_c(cell_total, 0.0), g3_b_x_c(cell_total, 0.0), rhs_b_x_c(cell_total, 0.0),
     f1_b_y_c(cell_total, 0.0), f2_b_y_c(cell_total, 0.0), g1_b_y_c(cell_total, 0.0), g2_b_y_c(cell_total, 0.0), g3_b_y_c(cell_total, 0.0), rhs_b_y_c(cell_total, 0.0),
     m_div_x(cell_total, 0.0), m_div_y(cell_total, 0.0), E_ex_div(cell_total, 0.0), b_rhs_div_x(cell_total, 0.0), b_rhs_div_y(cell_total, 0.0)
    
    {}
};

// defining helper functions (mostly for finite volume method calculations)

inline double get_derivative(double right_top_quantity, double left_bottom_quantity, double length) { // helper function to calculate the derivative of a quantity based off input values (just to clean code)
    // right_top_quantity refers to either the right or top faces of a cell, left_bottom_quantity refers to left or bottom
    
    return (right_top_quantity - left_bottom_quantity) / length;

}

void get_divergences(int total_x_cells, int total_y_cells, int ghost_cell_num, integrator_reserved_memory& memory, const Grid& grid_div) { // helper function to calculate the divergence of momentum density, mass density, B field and total energy density for all nodes
        
    // define reciprocal of permeability of free space
    constexpr double pi = 3.14159265358979323846;
    constexpr double mew_recp = 1 / (4 * pi * 1e-7);
    
    double cell_width;
    double cell_height; 
    
    // calculate face values of quantities for finite volume method

    // introducing scrapboards
    vector<double>& rho_div = memory.rho_div;
    vector<double>& E_div = memory.E_ex_div;
    vector<double>& m_div_x = memory.m_div_x;
    vector<double>& m_div_y = memory.m_div_y;
    vector<double>& b_rhs_div_x = memory.b_rhs_div_x;
    vector<double>& b_rhs_div_y = memory.b_rhs_div_y;

    // clearing scrapboards
    fill(rho_div.begin(), rho_div.end(), 0.0);
    fill(E_div.begin(), E_div.end(), 0.0);
    fill(m_div_x.begin(), m_div_x.end(), 0.0);
    fill(m_div_y.begin(), m_div_y.end(), 0.0);
    fill(b_rhs_div_x.begin(), b_rhs_div_x.end(), 0.0);
    fill(b_rhs_div_y.begin(), b_rhs_div_y.end(), 0.0);

    // uses finite volume method

    for (int i = ghost_cell_num; i < total_x_cells - ghost_cell_num; i++){     // loop to cover all nodes in a single row
                                                                               // note that the loops start and end on real nodes (avoid ghost cell padding)
        for (int j = ghost_cell_num; j < total_y_cells - ghost_cell_num; j++){ // loop to cover all rows           

            // 1. non-'stiff' rhs term of mass density rate of change (momentum density divergence) 
            rho_div.at(grid_div.indexC(i, j)) = get_derivative(m_face_x[grid_div.indexC(i + 1, j)], m_face_x[grid_div.indexC(i, j)], cell_width) + get_derivative(m_face_y[grid_div.indexC(i, j + 1)], m_face_y[grid_div.indexC(i, j)], cell_height);

            // 2. divergence of non-'stiff' rhs of energy conservation equation
            E_div.at(grid_div.indexC(i, j)) =   get_derivative( (E_face_x[grid_div.indexC(i + 1, j)] + pressure_face_x[grid_div.indexC(i + 1, j)] + (mew_recp * 0.5 * (b_x_face_x[grid_div.indexC(i + 1, j)] * b_x_face_x[grid_div.indexC(i + 1, j)] + b_y_face_x[grid_div.indexC(i + 1, j)] * b_y_face_x[grid_div.indexC(i + 1, j)]) ) ) * v_x_face_x[grid_div.indexC(i + 1, j)] - ,
                                                (E_face_x[grid_div.indexC(i, j)] + pressure_face_x[grid_div.indexC(i, j)]+ (mew_recp * 0.5 * (b_x_face_x[grid_div.indexC(i, j)] * b_x_face_x[grid_div.indexC(i, j)] + b_y_face_x[grid_div.indexC(i j)] * b_y_face_x[grid_div.indexC(i, j)])) * v_x_face_x[grid_div.indexC(i, j)]),
                                                cell_width) 
                                                +
                                                get_derivative( (E_face_y[grid_div.indexC(i, j + 1)] + pressure_face_y[grid_div.indexC(i, j + 1)] + (mew_recp * 0.5 * (b_y_face_y[grid_div.indexC(i, j + 1)] * b_y_face_y[grid_div.indexC(i, j + 1)] + b_x_face_y[grid_div.indexC(i, j + 1)] * b_x_face_y[grid_div.indexC(i, j + 1)]) ) ) * v_y_face_y[grid_div.indexC(i, j + 1)],
                                                (E_face_y[grid_div.indexC(i, j)] + pressure_face_y[grid_div.indexC(i, j)]+ (mew_recp * 0.5 * (b_y_face_y[grid_div.indexC(i, j)] * b_y_face_y[grid_div.indexC(i, j)] + b_x_face_y[grid_div.indexC(i, j)] * b_x_face_y[grid_div.indexC(i, j)])) * v_face_y[grid_div.indexC(i, j)]),
                                                cell_height);
            
            // 3. non_'stiff' rhs terms of magnetic flield density
            // first calculate values at cell faces
            double q_right = v_x_face_x[grid_div.indexC(i + 1, j)] * b_y_face_x[grid_div.indexC(i + 1, j)] - v_y_face_x[grid_div.indexC(i + 1, j)] * b_x_face_x[grid_div.indexC(i + 1, j)]; // v_x_face_y -> x component of velocity on y face
            double q_left = v_x_face_x[grid_div.indexC(i, j)] * b_y_face_x[grid_div.indexC(i, j)] - v_y_face_x[grid_div.indexC(i, j)] * b_x_face_x[grid_div.indexC(i, j)];
            double q_top = v_x_face_y[grid_div.indexC(i, j + 1)] * b_y_face_y[grid_div.indexC(i, j + 1)] - v_y_face_y[grid_div.indexC(i, j + 1)] * b_x_face_y[grid_div.indexC(i, j + 1)];
            double q_bottom = v_x_face_y[grid_div.indexC(i, j)] * b_y_face_y[grid_div.indexC(i, j)] - v_y_face_y[grid_div.indexC(i, j)] * b_x_face_y[grid_div.indexC(i, j)]; 

            // now calculate derivatives
            b_rhs_div_x.at(grid_div.indexC(i, j)) = get_derivative(q_top, q_bottom, cell_height);
            b_rhs_div_y.at(grid_div.indexC(i, j)) = - get_derivative(q_right, q_left, cell_width);

            // 4. non-'stiff' rhs of momentum density derivative
            // first calculate individual matrix terms at the required faces [a, b,
            //                                                                c, d]
            double a_right = ( rho_face_x[grid_div.indexC(i + 1, j)] * v_x_face_x[grid_div.indexC(i + 1, j)] * v_x_face_x[grid_div.indexC(i + 1, j)] ) -
                             ( mew_recp * b_x_face_x[grid_div.indexC(i + 1, j)] * b_x_face_x[grid_div.indexC(i + 1, j)] ) +
                             ( 0.5 * mew_recp * ((b_x_face_x[grid_div.indexC(i + 1, j)] * b_x_face_x[grid_div.indexC(i + 1, j)]) + (b_y_face_x[grid_div.indexC(i + 1, j)] * b_y_face_x[grid_div.indexC(i + 1, j)]) ) ) +
                             pressure_face_x[grid_div.indexC(i + 1, j)];
            
            double a_left = ( rho_face_x[grid_div.indexC(i, j)] * v_x_face_x[grid_div.indexC(i, j)] * v_x_face_x[grid_div.indexC(i, j)] ) -
                            ( mew_recp * b_x_face_x[grid_div.indexC(i, j)] * b_x_face_x[grid_div.indexC(i, j)] ) +
                            ( 0.5 * mew_recp * ( (b_x_face_x[grid_div.indexC(i, j)] * b_x_face_x[grid_div.indexC(i, j)]) + (b_y_face_x[grid_div.indexC(i, j)] * b_y_face_x[grid_div.indexC(i, j)]) ) ) +
                            pressure_face_x[grid_div.indexC(i, j)];

            double b_top = ( rho_face_y[grid_div.indexC(i, j + 1)] * v_x_face_y[grid_div.indexC(i, j + 1)] * v_y_face_y[grid_div.indexC(i, j + 1)] ) -
                           ( mew_recp * b_x_face_y[grid_div.indexC(i, j + 1)] * b_y_face_y[grid_div.indexC(i, j + 1)] );
            
            double b_bottom = ( rho_face_y[grid_div.indexC(i, j)] * v_x_face_y[grid_div.indexC(i, j)] * v_y_face_y[grid_div.indexC(i, j)] ) -
                              ( mew_recp * b_x_face_y[grid_div.indexC(i, j)] * b_y_face_y[grid_div.indexC(i, j)] );

            double c_right = ( rho_face_x[grid_div.indexC(i + 1, j)] * v_x_face_x[grid_div.indexC(i + 1, j)] * v_y_face_x[grid_div.indexC(i + 1, j)] ) -
                             ( mew_recp * b_x_face_x[grid_div.indexC(i + 1, j)] * b_y_face_x[grid_div.indexC(i + 1, j)] );

            double c_left = ( rho_face_x[grid_div.indexC(i, j)] * v_x_face_x[grid_div.indexC(i, j)] * v_y_face_x[grid_div.indexC(i, j)] ) -
                            ( mew_recp * b_x_face_x[grid_div.indexC(i, j)] * b_y_face_x[grid_div.indexC(i, j)] );
            
            double d_top = ( rho_face_y[grid_div.indexC(i, j + 1)] * v_y_face_y[grid_div.indexC(i, j + 1)] * v_y_face_y[grid_div.indexC(i, j + 1)] ) -
                           ( mew_recp * b_y_face_y[grid_div.indexC(i, j + 1)] * b_y_face_y[grid_div.indexC(i, j + 1)] ) +
                           ( 0.5 * mew_recp * ( (b_y_face_y[grid_div.indexC(i, j + 1)] * b_y_face_y[grid_div.indexC(i, j + 1)]) + (b_x_face_y[grid_div.indexC(i, j + 1)] * b_x_face_y[grid_div.indexC(i, j + 1)]) ) ) +
                           pressure_face_y[grid_div.indexC(i, j + 1)];
            
            double d_bottom = ( rho_face_y[grid_div.indexC(i, j)] * v_y_face_y[grid_div.indexC(i, j)] * v_y_face_y[grid_div.indexC(i, j)] ) -
                              ( mew_recp * b_y_face_y[grid_div.indexC(i, j)] * b_y_face_y[grid_div.indexC(i, j)] ) +
                              ( 0.5 * mew_recp * ( (b_y_face_y[grid_div.indexC(i, j)] * b_y_face_y[grid_div.indexC(i, j)]) + (b_x_face_y[grid_div.indexC(i, j)] * b_x_face_y[grid_div.indexC(i, j)]) ) ) +
                              pressure_face_y[grid_div.indexC(i, j)];
            
            // now calculate derivatives to give divergence components
            m_div_x.at(grid_div.indexC(i, j)) = get_derivative(a_right, a_left, cell_width) + get_derivative(b_top, b_bottom, cell_height);
            m_div_y.at(grid_div.indexC(i, j)) = get_derivative(c_right, c_left, cell_width) + get_derivative(d_top, d_bottom, cell_height);

        };
    };
};

void explicit_rhs(const Grid& grid_ex, global_state_dot& dU_explicit){ // explicit half of IMEX integrator, returns a partial global state derivative array of only non-'stiff' variables
    // takes in the current state values from grid for each node, then calculates the values of the non-'stiff' rhs terms

    // clear scrapboards here

    // variable values
    // creates constant double vectors that are references to the grid vector data, avoids copying data
    const vector<double>& node_rho_values = grid_ex.rho;    // density (centred on node)
    const vector<double>& node_m_x_values = grid_ex.mx;     // momentum density in x direcction (centred on node)
    const vector<double>& node_m_y_values = grid_ex.my;     // momentum density in y direction (centred on node)
    const vector<double>& face_b_x_values = grid_ex.bxf;  // magnetic B field on right face
    const vector<double>& face_b_y_values = grid_ex.byf;  // magnetic B field on bottom face
    const vector<double>& node_energy_values = grid_ex.rho; // energy density (centred on node) 
    


};


void IMEX(Grid& grid, integrator_reserved_memory& workspace, double dt){ // function that calculates the new global state of the system using a combination of implicit and explicit methods
    // workspace holds the RAM addresses of the vectors required for caluclations, saves efficiency by using address rather than copying data each time IMEX function is called
    // grid works similarly, but holds addresses for current (global) state variables

    // unpack global state conservative values (using addresses)
    vector<double>& rho = grid.rho;   // density array
    vector<double>& m_x = grid.mx;    // momentum density in x direction
    vector<double>& m_y = grid.my;    // momentum density in y direction
    vector<double>& E = grid.E;       // total energy density
    vector<double>& b_x_c = grid.bxc; // magnetic field density, centred on node, x in direction 




}