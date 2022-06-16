/*
    Simulation of dynamics of a overdamped Brownian particle coupled to a model
    A Gaussian field. Outputs average of particle position and average of square
    of particle position on stdout. Outputs simulations details and progress on
    stderr.

    Input arguments:
        T               Temperature
        D               Field mobility
        r               Field mass (distance from criticality)
        nu              Particle mobility
        k               Trap stifness
        lamb            Field-particle interaction strength
        R               Field-particle interaction radius (number of monomers)
        i0              Starting monomer (relative to center)

        L               Number of monomers 
        dx              Space-step
        dt              Time-step      
        n_traj          Number of trajectories
        t_steps         Number of timesteps for each trajectory
        t_therm         Number of steps for field thermalization

        print_traj      Print every "print_traj" trajectories
        print_step      Print every "print_step" steps
                        (set to 0 one or both fror no output)
       
    Sample stdout:
        1.0     0.9     0.85    ...
        0.013   0.025   0.035   ...

    Please see https://arxiv.org/abs/yymm.xxxxx for more details.
    
    Authored in June 2021 by:
        Francesco Ferraro - fferraro@sissa.it
        Davide Venturelli - dventure@sissa.it  
*/ 


#include <iostream>
#include <vector>
#include <math.h>
#include <random>
// #include <boost/format.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/bernoulli_distribution.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/normal_distribution.hpp>


double T, D, r, nu, k, lamb, R, dx, dt;
int i0, L, n_traj, t_steps, t_therm, print_traj, print_step;

void print_info();
void read_parameters(int, char**);
int dist_pbc(int, int, int);


int main(int argc, char** argv) {
    /****************************  Initialization  ****************************/
    read_parameters(argc, argv);
    print_info();
    
    // Various constants
    int center = (L-1)/2;   // Rounded down for even number of monomers
    double dx2 = dx*dx;
    double zetac = sqrt(2*D*T/(dx*dt));

    // State variables
    int particle, particle_next;
    std::vector<double> avg_particle (t_steps, 0);
    std::vector<double> avg_particle_sq (t_steps, 0);
    std::vector<double> field (L, 0);   // Initial field = 0
    std::vector<double> field_next (L, 0);
    
    // Random number generators
    std::random_device rd;
    boost::random::mt19937_64 gen(rd());
    boost::random::bernoulli_distribution<double> random_bool(0.5);
    boost::random::uniform_01<double> uniform;
    boost::random::normal_distribution<double> gaussian;
        
    // Other variables
    double dU_R, dU_L, p_R, p_L, R_unif, f, p , zeta;
    bool left_or_right;
    
    /********************  Realization of each trajectory  ********************/      
    for (size_t n=0; n!=n_traj; n++) {
        // Output progress
        if (print_traj && n % print_traj == 0) {
            std::cerr << "Trajectory: " << n << " / " << n_traj
                      << "\t" << (double)n/n_traj*100 << " %" << std::endl;
        }

        // Thermalization of the field
        for (size_t t=0; t!=t_therm; t++) {
            // Evolution of bulk field
            for (size_t i=1; i!=L-1; i++) {
                // Field and random forcing
                f = (field[i+1] + field[i-1] - 2*field[i])/dx2 - r*field[i];
                zeta = zetac * gaussian(gen);       
                field_next[i] = field[i] + dt * (D*f + zeta);
            }
                    
            // Evolution of first monomer, with p.b.c.
            f = (field[1] + field[L-1] - 2*field[0])/dx2 - r*field[0];
            zeta = zetac * gaussian(gen);
            field_next[0] = field[0] + dt * (D*f + zeta);
        
            // Evolution of last monomer, with p.b.c.
            f = (field[L-2] + field[0] - 2*field[L-1])/dx2 - r*field[L-1];
            zeta = zetac * gaussian(gen);
            field_next[L-1] = field[L-1] + dt * (D*f + zeta);
            
            // Update of field state
            for (size_t i=0; i!=L; i++) {
                field[i] = field_next[i];
            }
        }

        // Initial condition of particle
        particle = center + i0;
        avg_particle[0] += particle - center;
        avg_particle_sq[0] += (particle - center)*(particle - center);      
        
        // Time evolution of particle and field
        for (size_t t=0; t!=t_steps-1; t++) {
            if (print_step && t % print_step == 0) {
                if (print_traj) std::cerr << "\t";
                std::cerr << "Step: " << t << " / " << t_steps
                          << "\t" << (double)t/t_steps*100 << " %" << std::endl;
            }
            
            // Evolution of particle
            // Energy difference on the left and on the right
            dU_L = - k * (particle-center) * dx2
                    - lamb * (field[particle-R-1] - field[particle+R]) * dx;
            dU_R = + k * (particle-center) * dx2
                    - lamb * (field[particle+R+1] - field[particle-R]) * dx;
             
            // Probabilities of moving left or right
            p_L = 1 - exp(-nu*T*exp(-dU_L/(2*T))*dt/dx2);
            p_R = 1 - exp(-nu*T*exp(-dU_R/(2*T))*dt/dx2);
            
            // Since p_L and p_R can be > 1 this can be used to avoid
            // introducing bias in the left or right direction 
            left_or_right = random_bool(gen);
            
            if (left_or_right == 0){
                R_unif = uniform(gen);
                if (R_unif < p_L) {particle_next = particle - 1;}
                else if (R_unif < p_L+p_R) {particle_next = particle + 1;}
                else {particle_next = particle;}
            } 
            else {
                R_unif = uniform(gen);
                if (R_unif < p_R) {particle_next = particle + 1;}
                else if (R_unif < p_L+p_R) {particle_next = particle - 1;}
                else {particle_next = particle;} 
            }

            // Evolution of bulk field
            for (size_t i=1; i!=L-1; i++) {
                // Field, particle, random forcing
                f = (field[i+1] + field[i-1] - 2*field[i])/dx2 - r*field[i];
                if (dist_pbc(particle, i, L) <= R) {p = lamb;} else {p = 0;}
                zeta = zetac * gaussian(gen);

                field_next[i] = field[i] + dt * (D * (f + p) + zeta);
            }

            // Evolution of first monomer, with p.b.c.
            f = (field[1] + field[L-1] - 2*field[0])/dx2 - r*field[0];
            if (dist_pbc(particle, 0, L) <= R) {p = lamb;} else {p = 0;}
            zeta = zetac * gaussian(gen);
            field_next[0] = field[0] + dt * (D * (f + p) + zeta);

            // Evolution of last monomer, with p.b.c.
            f = (field[0] + field[L-2] - 2*field[L-1])/dx2 - r*field[L-1];
            if (dist_pbc(particle, L-1, L) <= R) {p = lamb;} else {p = 0;}
            zeta = zetac * gaussian(gen);
            field_next[L-1] = field[L-1] + dt * (D * (f + p) + zeta);
 
            // Update of system state
            particle = particle_next;
            for (size_t i=0; i!=L; i++) {field[i] = field_next[i];}
            
            // Calculation of average
            avg_particle[t+1] += particle - center;
            avg_particle_sq[t+1] += (particle - center)*(particle - center);
        }
    }
    
    
    /*******************************  Closing  ********************************/    
    // Output to stdout
    for (size_t t=0; t!=t_steps; t++) {
        std::cout << avg_particle[t]*dx / n_traj << " ";
    }
    std::cout << std::endl;

    for (size_t t=0; t!=t_steps; t++) {
        std::cout << avg_particle_sq[t]*dx2 / n_traj << " ";
    }

    std::cerr << "Trajectory: " << n_traj << " / " << n_traj
              << "\t" << "100 %" << std::endl;
    
    return 0;
}

void read_parameters(int argc, char** argv) {
    // Check if correct number of arguments are passed
    if (argc != 17) {
        std::cerr << "Usage: " << argv[0] << " "
                  << "T D r nu k lamb R i0 L dx dt "
                  << "n_traj t_steps t_therm "
                  << "print_traj print_step"
                  << std::endl;
        exit(1);
    }        

    T    = std::strtod(argv[1], NULL);    
    D    = std::strtod(argv[2], NULL);    
    r    = std::strtod(argv[3], NULL);    
    nu   = std::strtod(argv[4], NULL);    
    k    = std::strtod(argv[5], NULL);    
    lamb = std::strtod(argv[6], NULL);    
    R    = std::stoi(argv[7]);            
    i0   = std::stoi(argv[8]);            

    L       = std::stoi(argv[9]);         
    dx      = std::strtod(argv[10], NULL);
    dt      = std::strtod(argv[11], NULL);
    n_traj  = std::stoi(argv[12]);        
    t_steps = std::stoi(argv[13]);        
    t_therm = std::stoi(argv[14]);        
    
    print_traj = std::stoi(argv[15]);     
    print_step = std::stoi(argv[16]);     
                                    
}

void print_info() {
    std::cerr << "Parameters:" << std::endl
              << "\tT    = " << T << std::endl
              << "\tD    = " << D << std::endl
              << "\tr    = " << r << std::endl
              << "\tnu   = " << nu << std::endl
              << "\tk    = " << k << std::endl
              << "\tlamb = " << lamb << std::endl
              << "\tR    = " << R << std::endl 
              << "\ti0   = " << i0 << std::endl << std::endl
              << "\tL       = " << L << std::endl                  
              << "\tdx      = " << dx << std::endl
              << "\tdt      = " << dt << std::endl
              << "\tn_traj  = " << n_traj << std::endl
              << "\tt_steps = " << t_steps << std::endl
              << "\tt_therm = " << t_therm << std::endl << std::endl;
}

int dist_pbc(int x, int y, int L) {
    // Distance between points x and y on a line of length L and with
    // periodic boundary conditions. Assumes 0 <= x < L and 0 <= y < L
    
    int dist = std::abs(x-y);
    return std::min(dist, L-dist);
}