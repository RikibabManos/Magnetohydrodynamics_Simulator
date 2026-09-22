#pragma once
#include <vector>
#include <cstddef>
#include <cmath>

struct Grid {

public:
    int nx, ny;     // number of physical visible cells in x and y
    int ng;     // number of ghost cells on each side of the physical grid (2 for MUSCL, 3 for WENO)
    int nxt, nyt;      // total number of cells in x and y

    double cell_width; // meters
    double cell_height; // meters

    double resistivity;

    // Variables on cell centres: Conservative variables (density, momentum (x/y), Energy, B field (x/y)
    std::vector<double> rho, mx, my, E, bxc, byc; // Cell B field = average of face values

    // Variables on face centres: B field
    std::vector<double> bxf, byf;
    std::vector<double> Ez; // vector potential

    // Primitive variables (cell centered) (velocity (x/y) and pressure)
    std::vector<double> vx, vy, p;

    // Constructor function
    Grid(int nx_, int ny_, int ng_ = 2) // 2 for MUSCL for now
        // Assign variables
        : nx(nx_), ny(ny_), ng(ng_),
          nxt(nx_ + 2 * ng_), nyt(ny_ + 2 * ng_)

    {
        // Total number of cells in the grid
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

        Ez.assign(nCell, 0.0); // on corners
 
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

        // - bxc/byc are NOT independently wrapped as they're calculated from bxf/byf 
        //   every step via computeCellB() to stay consistent with the current face values.
        // - must come after fillOneField(bxf/byf) as this reads their ghost values
        computeCellB();
        fillOneField(bxc);
        fillOneField(byc);
    }

    void computeCellB() {
        // Compute cell centered B fields based on average of surrounding faces.
        // DO NOT use modulo wrapping. The ghost cells in bxf/byf already guarantee
        // that [i + 1] and [j + 1] contain the correct periodic data.

        for (int j = 0; j < nyt - 1; ++j) {
            for (int i = 0; i < nxt - 1; ++i) {
                
                // For x, (i,j) = left face, (i+1, j) = right face
                bxc[indexC(i, j)] = (bxf[indexC(i, j)] + bxf[indexC(i + 1, j)]) / 2.0;
                
                // For y, (i,j) = bottom face, (i, j+1) = top face
                byc[indexC(i, j)] = (byf[indexC(i, j)] + byf[indexC(i, j + 1)]) / 2.0;
            }
        }
    }

    // ensure fillGhostPeriodic() is called before this
    void computePrimitives() {
        // compute vx vy p based on conserved values and EoS

        // adiabatic index (ratio of specific heats); 5/3 = monatomic ideal gas
        const double gamma = 5.0/3.0;

        for (int j = 0; j < nyt; ++j) {
            for (int i = 0; i < nxt; ++i) {
                // index of current cell
                int id = indexC(i, j);

                // calculate velocities
                vx[id] = mx[id]/rho[id];
                vy[id] = my[id]/rho[id];

                if (rho[id] < 1e-8) {rho[id] = 1e-8;} // density floor

                // calculate energies
                double kinE = 0.5*rho[id]*(pow(vx[id],2) + pow(vy[id],2));
                double magE = (pow(bxc[id],2)  + pow(byc[id],2))/2;
                double intE = E[id] - kinE - magE;

                // calculate p from EoS
                p[id] = (gamma-1)*intE;
                if (p[id] < 1e-8) {
                    p[id] = 1e-8;      // prevents negative pressure crash

                    // re-sync total Energy (E) so the solver doesn't drift
                    kinE = 0.5*rho[id]*(pow(vx[id],2) + pow(vy[id],2));
                    magE = (pow(bxc[id],2)  + pow(byc[id],2))/2;
                    intE = E[id] - kinE - magE;

                } 

            }
        }
    }


private:
    // - Fill a field for the ghost cell (Use this 8 times in helper function as 8 variables)
    // - Assumes fully periodic domain in both x and y — do NOT use for non-periodic boundary conditions (e.g. Brio-Wu) 
    //   without a different fill rule.
    void fillOneField(std::vector<double>& f) {
        // left/right (x) ghost columns
        // For each row (increment j)
        for (int j = ng; j < nyt - ng; ++j) {
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
