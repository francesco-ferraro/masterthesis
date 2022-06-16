/*
    Simulation of dynamics of a overdamped Brownian particle in a potential.
    Outputs average of particle position and average of square of particle
    position on stdout. Outputs simulations details and progress on stderr.

    Sample stdout:
        1.0     0.9     0.85    ...
        0.013   0.025   0.035   ...

    See https://arxiv.org/abs/xxyy.zzzzz for more details.
    
    Authored by:
    Ferraro, Francesco - francesco.ferraro.vr@gmail.com
    Venturelli, Davide - dventure@sissa.it
*/ 

#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <random>
#include <boost/format.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/normal_distribution.hpp>
#include "npy.hpp"

void print_info();
int dist_pbc(int, int, int);
double dH(double);

double T, D, r, nu, k, lamb, R, dx, dt;
int i0, L, n_traj, t_steps, t_therm, print_traj, print_step;

int main(int argc, char** argv) {

    /****************************  Initialization  ****************************/
        
    // Parameter setting
    T    = 1;       // Temperature
    nu   = 1;       // Particle diffusivity
    k    = 0.1;     // Potential strength
    i0   = 5;       // Starting monomer relative to center

    L       = 1000;     // Number of monomers 
    dx      = 1;        // Space-step
    dt      = std::strtod(argv[1], NULL);   // Time-step      
    n_traj  = std::stoi(argv[2]);           // Number of experiments
    t_steps = std::stoi(argv[3]);           // Number of timesteps for each exp.
    
    print_traj = 5000;    // Print every "print_traj" trajs
    print_step = 0;       // Print every "print_step" steps
                          // Set to 0  no output
    
    // Print simulation parameters (to stderr)
    print_info();
    std::cerr << "\ttheta = " << 2*nu*dt/dx/dx << std::endl << std::endl;
    
    // Various constants
    int center = (L-1)/2;   // Rounded down for even number of monomers
    double dx2 = dx*dx;

    int particle, particle_next;    

    std::vector<double> mean_particle (t_steps, 0);
    std::vector<double> mean_particle_sq (t_steps, 0);
    
    // Random number generators
    std::random_device rd;
    boost::random::mt19937_64 gen(rd());
    boost::random::uniform_01<double> uniform;
    boost::random::normal_distribution<double> gaussian;
    
    double x, w_R, w_L, p_R, p_L, R_unif;
    int left = 0, right = 0;    // Number of times the particle moves left
                                // or right. Used for debugging only.
    
    
    /********************  Realization of each trajectory  ********************/
       
    for (size_t n=0; n!=n_traj; n++) {
        // Output progress
        if (print_traj && n % print_traj == 0) {
            std::cerr << "Trajectory: " << n << " / " << n_traj
                      << "\t" << (double)n/n_traj*100 << " %" << std::endl;
        }

        // Initial condition of particle
        particle = center + i0;
        mean_particle[0] += particle - center;
        mean_particle_sq[0] += (particle - center)*(particle - center);      
        
        // Time evolution of particle and field
        for (size_t t=0; t!=t_steps-1; t++) {
            if (print_step && t % print_step == 0) {
                if (print_traj) std::cerr << "\t";
                std::cerr << "Step: " << t << " / " << t_steps
                          << "\t" << (double)t/t_steps*100 << " %" << std::endl;
            }
            
            x = (particle - center)*dx;
            
            // Linear
            // w_L = nu*T*(1+dH(x)*dx/(2*T));
            // w_R = nu*T*(1-dH(x)*dx/(2*T));
            
            // Exponential
            w_L = nu*T*exp(+dH(x)*dx/(2*T));
            w_R = nu*T*exp(-dH(x)*dx/(2*T));
            
            // min-type
            // w_L = exp(+dH(x)*dx/T);
            // w_R = exp(-dH(x)*dx/T);
            // if (w_L > 1) {w_L = nu*T;} else {w_L = nu*T*w_L;}
            // if (w_R > 1) {w_R = nu*T;} else {w_R = nu*T*w_R;}
            
            p_L = w_L*dt/dx2;
            p_R = w_R*dt/dx2;
            
            R_unif = uniform(gen);
            if (R_unif < p_L) {
                particle_next = particle - 1;
                left += 1;
            } else if (R_unif < (p_L+p_R)) {
                particle_next = particle + 1;
                right += 1;
            } else {
                particle_next = particle;
            }

            // Update of system state
            particle = particle_next;
            
            // Adding to statistics
            mean_particle[t+1] += particle - center;
            mean_particle_sq[t+1] += (particle - center)*(particle - center);
        }
    }
    
    
    /*******************************  Closing  ********************************/
    
    for (size_t i = 0; i!=t_steps; i++) {
        mean_particle[i] *= dx/n_traj;
        mean_particle_sq[i] *= dx2/n_traj;
    }
    
    // Output to numpy .npy
    std::vector<double> output;
    output.reserve(mean_particle.size() + mean_particle_sq.size());
    output.insert(output.end(), mean_particle.begin(), mean_particle.end());
    output.insert(output.end(), mean_particle_sq.begin(), mean_particle_sq.end());
    
    unsigned long output_length;
    output_length = t_steps;    
    unsigned long shape [] = {2, static_cast<unsigned long>(t_steps)};
    
    npy::SaveArrayAsNumpy(argv[4], false, 2, shape, output);
    
    /*
    // Output to file
    std::ofstream out_file;
    out_file.open("./out.txt");
    
    for (size_t t=0; t!=t_steps; t++) {
         out_file << mean_particle[t]*dx / n_traj << " ";
    }
    out_file << std::endl;
    
    for (size_t t=0; t!=t_steps; t++) {
        out_file << mean_particle_sq[t]*dx2 / n_traj << " ";
    }
    
    out_file.close();
    */

    std::cerr << "Trajectory: " << n_traj << " / " << n_traj
              << "\t" << "100 %" << std::endl;

}

void print_info() {
    std::cerr << "Parameters:" << std::endl
              << "\tT    = " << T << std::endl
              << "\tnu   = " << nu << std::endl
              << "\tk    = " << k << std::endl
              << "\ti0   = " << i0 << std::endl << std::endl
              << "\tL       = " << L << std::endl                  
              << "\tdx      = " << dx << std::endl
              << "\tdt      = " << dt << std::endl
              << "\tn_traj  = " << n_traj << std::endl
              << "\tt_steps = " << t_steps << std::endl << std::endl;
}

int dist_pbc(int x, int y, int L) {
    // Distance between points x and y on a line of length L and with
    // periodic boundary conditions. Assumes 0 <= x < L and 0 <= y < L
    
    int dist = std::abs(x-y);
    return std::min(dist, L-dist);
}

double dH(double x) {
    // Derivative of the potential in which the particle moves.
    
    return k*x;                 // H(x) = k*x^2/2
    return 4*k*(x/9)*(x*x/9-1); // H(x) = k((x/3)^2-1)^2
    return 4*k*x*(x*x-1);       // H(x) = k(x^2-1)^2
}