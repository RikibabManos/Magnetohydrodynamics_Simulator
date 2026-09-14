// Code for HLL at first, develop into HLLD later
#pragma once
#include <vector>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "Grid.hpp"
#include "MUSCL.hpp"

// struct that holds primitive quantities for a state
struct PrimitiveState {
    double rho; // Mass density (ρ)
    double vx;  // Velocity x (v_x)
    double vy;  // Velocity y (v_y)
    double p;   // Thermal pressure (P)
    double bx;  // Magnetic field x (B_x)
    double by;  // Magnetic field y (B_y)
};

// struct that holds conserved quantities for a state
struct ConservedState {
    double rho; // Mass density (ρ)
    double mx;  // Momentum density x (m_x = ρ * v_x)
    double my;  // Momentum density y (m_y = ρ * v_y)
    double E;   // Total energy density (E)
    double bx;  // Magnetic field x (B_x)
    double by;  // Magnetic field y (B_y)

    // function that takes primitive quantities and converts them into conserved quantities 
    static ConservedState fromPrimitive(const PrimitiveState& w, double gamma = 5.0/3.0) {
        ConservedState u;

        u.rho = w.rho;
        u.mx  = w.rho * w.vx;
        u.my  = w.rho * w.vy;
        
        double v_sq = w.vx * w.vx + w.vy * w.vy;
        double b_sq = w.bx * w.bx + w.by * w.by;
        
        // E_tot = sum of energies = internal + kinetic + magnetic
        // Total energy density E = p/(gamma-1) + 0.5*rho*v^2 + 0.5*B^2
        u.E  = w.p / (gamma - 1.0) + 0.5 * w.rho * v_sq + 0.5 * b_sq;
        u.bx = w.bx;
        u.by = w.by;
        return u;
    }
};

// struct that holds the fluxes of conserved variables
struct FluxVector {
    double rho; // Mass flux
    double mx;  // Momentum X flux
    double my;  // Momentum Y flux
    double E;   // Total Energy flux
    double bx;  // Induction flux Bx
    double by;  // Induction flux By
};

// function that calculates the FluxVector from the primitive state in x-dir
FluxVector convertPrimitiveX(PrimitiveState p) {
    double gamma = 5.0 / 3.0;

    // helper variables
    double v_sq  = p.vx * p.vx + p.vy * p.vy;
    double B_sq  = p.bx * p.bx + p.by * p.by;
    double P_tot = p.p + 0.5 * B_sq;               // P_tot = thermal pressure + 0.5*B^2
    double v_dot_B = p.vx * p.bx + p.vy * p.by;    // Velocity dot Magnetic field
    double E = p.p / (gamma - 1.0) + 0.5*(p.rho * v_sq + B_sq); // Total Energy Density E

    // Populate Flux components
    FluxVector F;
    F.rho = p.rho * p.vx;
    F.mx  = p.rho * p.vx * p.vx + P_tot - p.bx * p.bx;
    F.my  = p.rho * p.vx * p.vy - p.bx * p.by;
    F.E   = (E + P_tot) * p.vx - p.bx * v_dot_B;
    F.bx  = 0.0;
    F.by  = p.vx * p.by - p.vy * p.bx;

    return F;
}

// function that calculates the FluxVector from the primitive state in y-dir
FluxVector convertPrimitiveY(PrimitiveState p) {
    double gamma = 5.0 / 3.0;

    // helper variables
    double v_sq  = p.vx * p.vx + p.vy * p.vy;
    double B_sq  = p.bx * p.bx + p.by * p.by;
    double P_tot = p.p + 0.5 * B_sq;               // P_tot = thermal pressure + 0.5*B^2
    double v_dot_B = p.vx * p.bx + p.vy * p.by;    // Velocity dot Magnetic field
    double E = p.p / (gamma - 1.0) + 0.5*(p.rho * v_sq + B_sq); // Total Energy Density E

    // Populate Flux components
    FluxVector F;
    F.rho = p.rho * p.vy;
    F.my  = p.rho * p.vy * p.vy + P_tot - p.by * p.by;
    F.mx  = p.rho * p.vx * p.vy - p.bx * p.by;
    F.E   = (E + P_tot) * p.vy - p.by * v_dot_B;
    F.by  = 0.0;
    F.bx  = p.vy * p.bx - p.vx * p.by;

    return F;
}

// helper function that calculates the fast magnetosonic speed from primitive quantities
double getFastWaveSpeed(PrimitiveState p, double bn){
    double gamma = 5.0 / 3.0;
    double B_sq  = p.bx * p.bx + p.by * p.by;

    double sound_v_sq = gamma * p.p / p.rho; // sound speed squared (a^2)
    double tot_alfven_v_sq = B_sq / p.rho; // total alfven speed squared (b^2)
    double norm_alfven_v_sq = bn * bn/ p.rho; // alfven speed squared normal to interface (c^2)

    double sum = tot_alfven_v_sq + sound_v_sq; // a^2 + b^2
    double term = sum * sum - 4.0 * sound_v_sq * norm_alfven_v_sq; // (a^2 + b^2)^2 - 4 * a^2 * c^2

    // sqrt(0.5 * (a^2 + b^2 + sqrt((a^2 + b^2)^2 - 4 * a^2 * c^2)))
    double fast_ms_v = std::sqrt(0.5 * (sum+std::sqrt(std::max(0.0, term))));

    return fast_ms_v;
}

// function that calculates the flux in x-dir using HLL at a single face
// L = left side of interface, R = right side of interface
FluxVector computeHLLFluxX(PrimitiveState L, PrimitiveState R){
    // flux vectors for both sides of interface
    FluxVector F_L = convertPrimitiveX(L);
    FluxVector F_R = convertPrimitiveX(R);

    // fast magnetosonic speed at both sides of interface
    double fast_ms_vL = getFastWaveSpeed(L, L.bx);
    double fast_ms_vR = getFastWaveSpeed(R, R.bx);

    // Slowest wave going left and fastest wave going right
    double signal_L = std::min(L.vx - fast_ms_vL, R.vx - fast_ms_vR);
    double signal_R = std::max(L.vx + fast_ms_vL, R.vx + fast_ms_vR);

    // 3 cases for HLL:
        // if slowest wave > 0, then all waves > 0 so return left flux
        // if fastest wave < 0, then all waves < 0 so return right flux
        // if slowest wave < 0 < fastest wave, then waves are colliding so flux vector needs to be calculated for each quantity
            // HLL vector = [SR * FL - SL * FR + SR * SL * (UR - UL)]/(SR - SL)
                // SL/SR = signal L/R, FL/FR = flux variables L/R, UL/UR = conserved variables L/R
    if (signal_L >= 0)
    {
        return F_L;        
    }
    else if (signal_R <= 0)
    {
        return F_R;        
    }
    else{
        // Conserved variables from primitives at both sides of interface
        ConservedState U_L = ConservedState::fromPrimitive(L);
        ConservedState U_R = ConservedState::fromPrimitive(R);

        // Common denominator of HLL vectors
        double inv_S = 1.0/(signal_R - signal_L);

        // Construct HLL FluxVector based on equations
        FluxVector F_HLL;
        F_HLL.rho = (signal_R * F_L.rho - signal_L* F_R.rho + signal_L * signal_R * (U_R.rho - U_L.rho)) * inv_S;
        F_HLL.mx  = (signal_R * F_L.mx  - signal_L * F_R.mx  + signal_L * signal_R * (U_R.mx  - U_L.mx))  * inv_S;
        F_HLL.my  = (signal_R * F_L.my  - signal_L * F_R.my  + signal_L * signal_R * (U_R.my  - U_L.my))  * inv_S;
        F_HLL.E   = (signal_R * F_L.E   - signal_L * F_R.E   + signal_L * signal_R * (U_R.E   - U_L.E))   * inv_S;
        F_HLL.bx  = 0.0; // Bx flux across X-interface is always zero
        F_HLL.by  = (signal_R * F_L.by  - signal_L * F_R.by  + signal_L * signal_R * (U_R.by  - U_L.by))  * inv_S;

        return F_HLL;
    }
}

// function that calculates the flux in x-dir using HLL at a single face
// function that calculates the flux in y-dir using HLL at a single face
// D = Down (bottom) side of interface (j - 1/2), U = Up (top) side of interface (j + 1/2)
FluxVector computeHLLFluxY(PrimitiveState D, PrimitiveState U){
    // flux vectors for both sides of interface
    FluxVector F_D = convertPrimitiveY(D);
    FluxVector F_U = convertPrimitiveY(U);

    // fast magnetosonic speed at both sides of interface
    double fast_ms_vD = getFastWaveSpeed(D, D.by);
    double fast_ms_vU = getFastWaveSpeed(U, U.by);

    // Slowest wave going down (signal_D) and fastest wave going up (signal_U)
    double signal_D = std::min(D.vy - fast_ms_vD, U.vy - fast_ms_vU);
    double signal_U = std::max(D.vy + fast_ms_vD, U.vy + fast_ms_vU);

    // 3 cases for HLL:
    if (signal_D >= 0)
    {
        return F_D; // Flow moving up everywhere -> upwind is Down state
    }
    else if (signal_U <= 0)
    {
        return F_U; // Flow moving down everywhere -> upwind is Up state
    }
    else {
        // Conserved variables from primitives at both sides of interface
        ConservedState U_D = ConservedState::fromPrimitive(D);
        ConservedState U_U = ConservedState::fromPrimitive(U);

        // Common denominator of HLL vectors
        double inv_S = 1.0 / (signal_U - signal_D);

        // Construct HLL FluxVector based on equations
        FluxVector F_HLL;
        F_HLL.rho = (signal_U * F_D.rho - signal_D * F_U.rho + signal_D * signal_U * (U_U.rho - U_D.rho)) * inv_S;
        F_HLL.mx  = (signal_U * F_D.mx  - signal_D * F_U.mx  + signal_D * signal_U * (U_U.mx  - U_D.mx))  * inv_S;
        F_HLL.my  = (signal_U * F_D.my  - signal_D * F_U.my  + signal_D * signal_U * (U_U.my  - U_D.my))  * inv_S;
        F_HLL.E   = (signal_U * F_D.E   - signal_D * F_U.E   + signal_D * signal_U * (U_U.E   - U_D.E))   * inv_S;
        F_HLL.bx  = (signal_U * F_D.bx  - signal_D * F_U.bx  + signal_D * signal_U * (U_U.bx  - U_D.bx))  * inv_S;
        F_HLL.by  = 0.0; // By flux across Y-interface is always zero

        return F_HLL;
    }
}
