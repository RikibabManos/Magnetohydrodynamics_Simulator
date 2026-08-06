// Code for MUSCL at first, may develop into WENO after
#pragma once
#include <vector>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "Grid.hpp"

// returns the smallest of the two slopes
double minmod(double a, double b) {
    if (a * b > 0.0) {
        return (std::abs(a) < std::abs(b)) ? a : b;
    }
    // if a < 0 < b etc. then somewhere within this cell is a local extrema so take 0 as slope
    return 0.0;
}

// struct that holds values at left and right face
struct FaceValues {double L, R;};

// enum class for x and y direction for specificity
enum class Direction { X, Y };

// function that completes MUSCL on one cell for one field
FaceValues MUSCLReconstructOne(const std::vector<double>& F, int i, int j, Direction dir, const Grid& g) {
    double leftSlope = 0;
    double rightSlope = 0;
    
    // calculates slope values by subtracting from the left or right neighbour (or up/down in y)
    if(dir == Direction::X){
        leftSlope  = F[g.indexC(i,j)]   - F[g.indexC(i-1,j)];
        rightSlope = F[g.indexC(i+1,j)] - F[g.indexC(i,j)];
    }
    else{
        leftSlope  = F[g.indexC(i,j)]   - F[g.indexC(i,j-1)];
        rightSlope = F[g.indexC(i,j+1)] - F[g.indexC(i,j)];
    }

    // takes the lower value slope to avoid sharp changes in field
    double slope = minmod(leftSlope, rightSlope);

    // extrapolates to faces from the cell centre using slope
    double L = F[g.indexC(i,j)] - 0.5*slope;
    double R = F[g.indexC(i,j)] + 0.5*slope;
    return {L, R};
}



// struct that holds all output arrays from MUSCL 
struct ReconstructedValues{
std::vector<double> rhoXFaceL, rhoXFaceR, rhoYFaceL, rhoYFaceR,
                    vxXFaceL,  vxXFaceR,  vxYFaceL,  vxYFaceR,
                    vyXFaceL,  vyXFaceR,  vyYFaceL,  vyYFaceR,
                    pXFaceL,   pXFaceR,   pYFaceL,   pYFaceR,
                    bxYFaceL, bxYFaceR,
                    byXFaceL, byXFaceR;

    // empty construction
    ReconstructedValues(const size_t nCell){
        rhoXFaceL.assign(nCell, 0.0); rhoXFaceR.assign(nCell, 0.0); rhoYFaceL.assign(nCell, 0.0); rhoYFaceR.assign(nCell, 0.0);
        vxXFaceL.assign(nCell, 0.0);  vxXFaceR.assign(nCell, 0.0);  vxYFaceL.assign(nCell, 0.0);  vxYFaceR.assign(nCell, 0.0);
        vyXFaceL.assign(nCell, 0.0);  vyXFaceR.assign(nCell, 0.0);  vyYFaceL.assign(nCell, 0.0);  vyYFaceR.assign(nCell, 0.0);
        pXFaceL.assign(nCell, 0.0);   pXFaceR.assign(nCell, 0.0);   pYFaceL.assign(nCell, 0.0);   pYFaceR.assign(nCell, 0.0);
        bxYFaceL.assign(nCell, 0.0); bxYFaceR.assign(nCell, 0.0);
        byXFaceL.assign(nCell, 0.0); byXFaceR.assign(nCell, 0.0);
    }

};

// helper struct to group a field into one variable
struct FieldEntry {
    const std::vector<double>& field;   // (e.g. g.rho)
    std::vector<double>& L;            
    std::vector<double>& R;
    std::vector<double>& U;
    std::vector<double>& D;
};

// main function that performs MUSCL on full grid
void MUSCLReconstructAll(const Grid& g, ReconstructedValues& RV){
    // creates an array of FieldEntrys to loop through when performing MUSCL on all fields
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
            // Perform MUSCL on values in x and y separately
                FaceValues valuesX = MUSCLReconstructOne(Fields[a].field, i, j, Direction::X, g);
                Fields[a].L[id] = valuesX.L; Fields[a].R[id] = valuesX.R;
                
                FaceValues valuesY = MUSCLReconstructOne(Fields[a].field, i, j, Direction::Y, g);
                Fields[a].U[id] = valuesY.L; Fields[a].D[id] = valuesY.R;
            }
        
            // B field must be constructed manually as div B = 0 so bxXFaces and byYFaces come directly from grid
            // Y-dir (Tangential): Slope reconstruction using cell-centered bxc
            FaceValues bx_Y = MUSCLReconstructOne(g.bxc, i, j, Direction::Y, g);
            RV.bxYFaceL[id] = bx_Y.L;                    
            RV.bxYFaceR[id] = bx_Y.R;                     


            // --- By Reconstruction ---
            // X-dir (Tangential): Slope reconstruction using cell-centered byc
            FaceValues by_X = MUSCLReconstructOne(g.byc, i, j, Direction::X, g);
            RV.byXFaceL[id] = by_X.L;                   
            RV.byXFaceR[id] = by_X.R;                    
        }
    }
}
