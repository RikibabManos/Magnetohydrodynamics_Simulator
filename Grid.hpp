// Code for the grid structure and how it works
#pragma once
#include <vector>
#include <cstddef>
#include <cmath>

// Structure = 1 large grid with an array for each variable (Struct of Arrays) 
// 2 types of arrays: Cell centered and Face centered
    // Cell centered = conservative variables that evolve with FVM --> (nxt)*(nyt) elements per array 
    // Face centered = B fields that vary at boundaries that evolve with CT --> (nxt)*(nyt) elements per array
struct Grid {

public:
    int nx, ny;     // number of physical visible cells in x and y
    int ng;     // number of ghost cells on each side (2 for MUSCL, 3 for WENO)
    int nxt, nyt;      // total number of cells in x and y

    // Variables on cell centres: Conservative variables
    std::vector<double> rho, mx, my, E, bxc, byc; // Cell B field = average of face values

    // Variables on face centres: B field
    std::vector<double> bxf, byf;

    // Primitive variables (cell centered)
    std::vector<double> vx, vy, p;

    // Constructor function
    Grid(int nx_, int ny_, int ng_ = 2) // 2 for MUSCL for now
        // Assign variables
        : nx(nx_), ny(ny_), ng(ng_),
          nxt(nx_ + 2 * ng_), nyt(ny_ + 2 * ng_)

    {
        // Number of cells
        const size_t nCell = static_cast<size_t>(nxt) * nyt;
        
        // Cell centered arrays
        rho.assign(nCell, 0.0);
        mx.assign(nCell, 0.0);
        my.assign(nCell, 0.0);
        E.assign(nCell, 0.0);
        vx.assign(nCell, 0.0);
        vy.assign(nCell, 0.0);
        p.assign(nCell, 0.0);
        bxc.assign(nCell, 0.0);
        byc.assign(nCell, 0.0);
 
        // Face centered arrays --> Cells wrap around so 4 cells have 4 faces as face 1 and 5 are the same face
        bxf.assign(nCell, 0.0);
        byf.assign(nCell, 0.0);
    }

    // i, j here are in TOTAL grid coordinates: Grid spans (0,0) to (nxt-1, nyt-1)
        // i.e. (ng,ng) corresponds to the first physical cell.
    inline int indexC(int i, int j) const { return j * nxt + i; } // Cell index
 
    // convert physical-cell coordinates (0..nx-1) to grid coords (Useful for easier switch between MUSCL and WENO as diff ng values)
    inline int gi(int iPhys) const { return iPhys + ng; }
    inline int gj(int jPhys) const { return jPhys + ng; }

    void fillGhostPeriodic() {
        // Helper function to avoid multiple function calls at once
        // Fill ghost cell's variables with data from opposite side (wrap-around)
        fillOneField(rho);
        fillOneField(mx);
        fillOneField(my);
        fillOneField(E);
        fillOneField(bxf);
        fillOneField(byf);
        computeCellB();
    }

    void computeCellB() {
        // Compute cell centered B fields based on average of surrounding faces in relevant direction
        for (int j = 0; j < nyt; ++j) {
            for (int i = 0; i < nxt; ++i) {
                int ip1 = (i + 1) % nxt;   // wrap: nxt-1 -> 0
                int jp1 = (j + 1) % nyt;   // wrap: nyt-1 -> 0

                bxc[indexC(i, j)] = (bxf[indexC(i, j)] + bxf[indexC(ip1, j)]) / 2.0;
                byc[indexC(i, j)] = (byf[indexC(i, j)] + byf[indexC(i, jp1)]) / 2.0;
            }
        }
    }

    // ensure fillGhostPeriodic() is called before this
    void computePrimitives() {
        // compute vx vy p based on conserved values and EoS
        const double gamma = 5.0/3.0;

        for (int j = 0; j < nyt; ++j) {
            for (int i = 0; i < nxt; ++i) {
                // index of current cell
                int id = indexC(i, j);

                // calculate velocities
                vx[id] = mx[id]/rho[id];
                vy[id] = my[id]/rho[id];

                // calculate energies
                double kinE = 0.5*rho[id]*(pow(vx[id],2) + pow(vy[id],2));
                double magE = (pow(bxc[id],2)  + pow(byc[id],2))/2;
                double intE = E[id] - kinE - magE;

                // calculate p from EoS
                p[id] = (gamma-1)*intE;
            }
        }
    }


private:
    // Fill a field for the ghost cell (Use this 8 times in helper function as 8 variables)
    void fillOneField(std::vector<double>& f) {
        // left/right (x) ghost columns
        // For each row (increment j)
        for (int j = 0; j < nyt; ++j) {
            // g = index of ghost cell
            for (int g = 0; g < ng; ++g) {
                f[indexC(g, j)]              = f[indexC(nx + g, j)];       // left <- right
                f[indexC(nx + ng + g, j)]    = f[indexC(ng + g, j)];       // right <- left
            }
        }
        // top/bottom (y) ghost rows, including corners (safe since x ghosts already filled)
        // For each column (increment i)
        for (int i = 0; i < nxt; ++i) {
            for (int g = 0; g < ng; ++g) {
                f[indexC(i, g)]              = f[indexC(i, ny + g)]; // top <- bottom
                f[indexC(i, ny + ng + g)]    = f[indexC(i, ng + g)]; // bottom <- top
            }
        }
    }
};