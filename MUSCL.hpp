// Code for MUSCL at first, may develop into WENO after
#pragma once
#include <vector>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "Grid.hpp"

// Structure:
    // - Performs 2D spatial reconstruction to calculate interface states from cell-centered averages
    // - This code implements a MUSCL algorithm
        // - Uses linear extrapolation within each cell to calculate values at faces
        // - Uses slope limiters to maintain TVD (Total Variation Diminishing) properties 
            // - Prevents non-physical oscillations (spurious overshoot/undershoot) near shocks and steep gradients
    // - Stores reconstructed state variables at cell boundaries in `ReconstructedValues`
        // - Holds left (L) and right (R) state vectors for X-direction faces, and Up (U) / Down (D) for Y-direction faces
        // - Hydrodynamic primitive variables (rho, vx, vy, p) are reconstructed along both X and Y directions
        // - Magnetic fields are reconstructed tangentially using cell-centered averages (bxc, byc)
            // - B_x is reconstructed across Y-faces (bxYFaceL/R) using cell-centered bxc
            // - B_y is reconstructed across X-faces (byXFaceL/R) using cell-centered byc
            // - Normal B components on faces do not need MUSCL reconstruction as they are handled directly by Constrained Transport (CT)
    // - Reconstruction loop iterates over the physical grid plus adjacent ghost cells (from gi(-1) to gi(nx))
        // - Guarantees that interface states at the physical domain boundaries are fully populated for the Riemann solver

// Functions:
    // - minmod(): slope limiter helper function
        // - Compares left and right gradients; returns the gradient with the smaller magnitude if both have the same sign
        // - Returns 0.0 if gradients have opposite signs (identifies local extrema/peaks and drops locally to 1st order)
    // - FaceValues: simple data structure bundling the reconstructed left (L) and right (R) face values for a single cell
    // - Direction: enum class used to select the spatial sweep axis (Direction::X or Direction::Y)
    // - MUSCLReconstructOne(): performs 1D linear spatial extrapolation for a single scalar variable in cell (i, j)
        // - Calculates forward and backward differences (leftSlope and rightSlope) along the direction (X/Y)
            // - Left = lower index face and Right = higher index face
            // - For cell (i,j) in x-dir: L = (i-0.5,j) and R = (i+0.5,j) 
        // - Limits the slope via minmod() and extrapolates values to the cell boundaries:
            // - L = F[i,j] - 0.5 * slope (value at the left/bottom interface of cell i,j)
            // - R = F[i,j] + 0.5 * slope (value at the right/top interface of cell i,j)
    // - ReconstructedValues: container struct holding 1D contiguous vectors for all reconstructed interface quantities
        // - Constructor allocates and zeros out memory for all face arrays using .assign(nCell, 0.0)
    // - FieldEntry: helper struct linking a target grid quantity (e.g. g.rho) to its corresponding 4 face output vectors (L, R, U, D)
        // - Allows batched iteration over primitive variables in MUSCLReconstructAll() to reduce repetitive code
    // - MUSCLReconstructAll(): executes MUSCLReconstructOne() across the full grid for all values
        // - Loops over the grid (inc. ghost cells) and populates all face-reconstructed values in `RV`

double minmod(double a, double b) {
    if (a * b > 0.0) {
        return (std::abs(a) < std::abs(b)) ? a : b;
    }
    return 0.0;
}

struct FaceValues {double L, R;};

enum class Direction { X, Y };

FaceValues MUSCLReconstructOne(const std::vector<double>& F, int i, int j, Direction dir, const Grid& g) {
    double leftSlope = 0;
    double rightSlope = 0;
    if(dir == Direction::X){
        leftSlope  = F[g.indexC(i,j)]   - F[g.indexC(i-1,j)];
        rightSlope = F[g.indexC(i+1,j)] - F[g.indexC(i,j)];
    }
    else{
        leftSlope  = F[g.indexC(i,j)]   - F[g.indexC(i,j-1)];
        rightSlope = F[g.indexC(i,j+1)] - F[g.indexC(i,j)];
    }
    
    double slope = minmod(leftSlope, rightSlope);

    double L = F[g.indexC(i,j)] - 0.5*slope;
    double R = F[g.indexC(i,j)] + 0.5*slope;
    return {L, R};
}



struct ReconstructedValues{
std::vector<double> rhoXFaceL, rhoXFaceR, rhoYFaceL, rhoYFaceR,
                    vxXFaceL,  vxXFaceR,  vxYFaceL,  vxYFaceR,
                    vyXFaceL,  vyXFaceR,  vyYFaceL,  vyYFaceR,
                    pXFaceL,   pXFaceR,   pYFaceL,   pYFaceR,
                    bxYFaceL, bxYFaceR,
                    byXFaceL, byXFaceR;


    ReconstructedValues(const size_t nCell){
        rhoXFaceL.assign(nCell, 0.0); rhoXFaceR.assign(nCell, 0.0); rhoYFaceL.assign(nCell, 0.0); rhoYFaceR.assign(nCell, 0.0);
        vxXFaceL.assign(nCell, 0.0);  vxXFaceR.assign(nCell, 0.0);  vxYFaceL.assign(nCell, 0.0);  vxYFaceR.assign(nCell, 0.0);
        vyXFaceL.assign(nCell, 0.0);  vyXFaceR.assign(nCell, 0.0);  vyYFaceL.assign(nCell, 0.0);  vyYFaceR.assign(nCell, 0.0);
        pXFaceL.assign(nCell, 0.0);   pXFaceR.assign(nCell, 0.0);   pYFaceL.assign(nCell, 0.0);   pYFaceR.assign(nCell, 0.0);
        bxYFaceL.assign(nCell, 0.0); bxYFaceR.assign(nCell, 0.0);
        byXFaceL.assign(nCell, 0.0); byXFaceR.assign(nCell, 0.0);
    }

};

struct FieldEntry {
    const std::vector<double>& field;   // (e.g. g.rho)
    std::vector<double>& L;            
    std::vector<double>& R;
    std::vector<double>& U;
    std::vector<double>& D;
};


void MUSCLReconstructAll(const Grid& g, ReconstructedValues& RV){
    std::vector<FieldEntry> Fields = {
      {g.rho, RV.rhoXFaceL, RV.rhoXFaceR, RV.rhoYFaceL, RV.rhoYFaceR},
      {g.vx, RV.vxXFaceL, RV.vxXFaceR, RV.vxYFaceL, RV.vxYFaceR},
      {g.vy, RV.vyXFaceL, RV.vyXFaceR, RV.vyYFaceL, RV.vyYFaceR},
      {g.p, RV.pXFaceL, RV.pXFaceR, RV.pYFaceL, RV.pYFaceR},
    };

    
    for (size_t i = g.gi(-1); i <= g.gi(g.nx); i++){
        for (size_t j = g.gj(-1); j <= g.gj(g.ny); j++){
        int id = g.indexC(i, j);

            for (size_t a = 0; a < Fields.size(); a++){
            FaceValues valuesX = MUSCLReconstructOne(Fields[a].field, i, j, Direction::X, g);
            Fields[a].L[id] = valuesX.L; Fields[a].R[id] = valuesX.R;
            
            FaceValues valuesY = MUSCLReconstructOne(Fields[a].field, i, j, Direction::Y, g);
            Fields[a].U[id] = valuesY.L; Fields[a].D[id] = valuesY.R;
            }
        
            // --- Bx Reconstruction ---
            // Y-dir (Tangential): Slope reconstruction using cell-centered bxc
            FaceValues bx_Y = MUSCLReconstructOne(g.bxc, i, j, Direction::Y, g);
            RV.bxYFaceL[id] = bx_Y.L;                     // Bottom face (j - 1/2)
            RV.bxYFaceR[id] = bx_Y.R;                     // Top face (j + 1/2)


            // --- By Reconstruction ---
            // X-dir (Tangential): Slope reconstruction using cell-centered byc
            FaceValues by_X = MUSCLReconstructOne(g.byc, i, j, Direction::X, g);
            RV.byXFaceL[id] = by_X.L;                     // Left face (i - 1/2)
            RV.byXFaceR[id] = by_X.R;                     // Right face (i + 1/2)
        }
    }
}