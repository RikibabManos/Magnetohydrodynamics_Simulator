// Code for the grid structure and how it works
#pragma once
#include <vector>
#include <cstddef>
#include <cmath>

// Structure:
    // - 1 large grid with an array for each variable (Struct of Arrays data layout) 
    // - Each array has the details for a quantity at each grid index
    // - Two types of arrays based on quantity type: Cell centered and face centered 
        // - Most quantities are cell centered, and the value refers to the average within that cell
            // - e.g. Cell centered B_x (bxc) is the average B field in the x direction within the cell  
        // - B is calculated on the edges of the cells (the faces) as part of the constrained transport method of 
        //   enforcing div B = 0 because you can use the values at each face to calculate the flux within the cell
    // - The size of the physical grid is nx in x-dir and ny in y-dir --> total grid = (0,0) to (nx-1,ny-1) 
    // - The spatial reconstruction methods that are used to calculate the flux requires 2-3 cells on either side of 
    //   the cell in question
        // - In order to make sure the boundary cells can read values correctly (e.g. cell at (0,y) has no cells to its left), 
        //   an amount of ghost cells are included in the grid (cells that arent visible but take values) 
        // - For MUSCL, 2 ghost cells are needed on either side of the reference cell, so the
        //   total grid size is (0,0) to (nx + 2*ng, ny + 2*ng), which is simplified to (nxt,nyt)
    // - There are two types of quantities that are relevant in fluid dynamics: primitive and conserved quantities 
        // - Primitive quantities are the direct physical properties of a flow
        // - Conserved quantities are quantities that directly result from conservation laws

// Functions:
    // - Constructor initialises each array and makes them empty using .assign(nCell, 0.0)
    // - indexC converts the cell coordinates into its position in the list
        // - i.e. for nxt = nyt = 10, indexC(4,3) would return 34
    // - gi, gj (grid i/j) = optional QoL function, can be used to convert coordinates in the real grid into the actual grid
        // - i.e. if you wanted (2,3) from the PHYSICAL grid, this is (4,5) in the actual due to ghost cells
        // - gi and gj functions skip the need for that logic if you would like to just focus on physical grid spaces so that
        //   you dont accidentally reference ghost cells
        // - May also be useful later if we change ng = 2 to ng = 3 for WENO as it avoids having to change every index in code later
    // - fillOneField() takes in a quantity, and fills in all the ghost cells with the correct values so they are up to date
        // - Takes the value from the cell on the opposite end, 2D grid can be considered looped and that the 
        //   left+right edges are joined together etc. (like a torus) 
    // - computeCellB() takes the average of the face-centered B fields to calculate the cell centered B field
    // - fillGhostPeriodic() is a helper function that calls fillOneField() for all variables
    // - computePrimitives() calculates the primitive quantities from the conserved quantities using standard equations
struct Grid {

public:
    int nx, ny;     // number of physical visible cells in x and y
    int ng;     // number of ghost cells on each side of the physical grid (2 for MUSCL, 3 for WENO)
    int nxt, nyt;      // total number of cells in x and y

    // Variables on cell centres: Conservative variables (density, momentum (x/y), Energy, B field (x/y)
    std::vector<double> rho, mx, my, E, bxc, byc; // Cell B field = average of face values

    // Variables on face centres: B field
    std::vector<double> bxf, byf;

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
    }

    void computeCellB() {
        // Compute cell centered B fields based on average of surrounding faces in relevant direction
        for (int j = 0; j < nyt; ++j) {
            for (int i = 0; i < nxt; ++i) {
                // MOD taken as if i = nxt-1, this is the rightmost cell so the right-side face is at 0
                int ip1 = (i + 1) % nxt;

                // same occurs with y-dir
                int jp1 = (j + 1) % nyt; 

                // for x, (i,j) = left face, (ip1, j) = right face
                bxc[indexC(i, j)] = (bxf[indexC(i, j)] + bxf[indexC(ip1, j)]) / 2.0;
                byc[indexC(i, j)] = (byf[indexC(i, j)] + byf[indexC(i, jp1)]) / 2.0;
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
    // - Fill a field for the ghost cell (Use this 8 times in helper function as 8 variables)
    // - Assumes fully periodic domain in both x and y — do NOT use for non-periodic boundary conditions (e.g. Brio-Wu) 
    //   without a different fill rule.
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
