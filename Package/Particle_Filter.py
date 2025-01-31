import numpy as np
from GPS_Simulator import GPS
from System_Dynamics import Rocket
import random

class ParticleFilter:
    def __init__(self, total_steps, n_particles, confidence_radius, resampling_frequency = 10, dt = 0.01):
        self.n_steps = total_steps
        self.n_particles = n_particles
        self.confidence_radius = confidence_radius
        self.standard_deviation = 3
        self.resampling_frequency = resampling_frequency
        """
        State Vector:
        3 dimensions for position, 3 dim. for vel. and 1 for the weight
        """

        self.particles = np.zeros((self.n_steps, self.n_particles, 7)) 
        # TODO: Instead of storing all the particles for the entire discrete time space, just store two separate 2D np.arrays: self.particles_old, self.particles_new

        self.gps_current = np.zeros((1,3))
        self.gps_old = np.zeros((1,3))
        self.gps_velocity = np.zeros((1,3))

        self.dt = dt

    def state_progression(self, current_time, A_matrix):
        A_matrix = A_matrix[:6,:6]
        for i, particle in enumerate(self.particles[current_time-1]):
            # particle is the state vector of a single particle, at the current time step
            self.particles[current_time][i] = A_matrix @ particle

    def _GPS_Update(self, gps_new):
        self.gps_old = self.gps_current
        self.gps_current = gps_new
        self.gps_velocity = (self.gps_current - self.gps_old) / self.dt

    def _normalization(weights):
        maxv = max(weights)
        minv = max(weights)
        if isinstance(weights, np.ndarray):
            if maxv != minv:
                weights = [(i - minv) / (maxv - minv) for i in weights]
                sum_w = sum(weights)
                weights = [i/sum_w for i in weights]
                return weights
            else:
                return np.zeros_like(weights)
        if maxv != minv:
            weights = [(i - minv) / (maxv - minv) for i in weights]
            sum_w = sum(weights)
            weights = [i/sum_w for i in weights]
            return weights
        else:
            return [0 for i in weights]

    def update_weights(self, gps_coordinates, current_time):
        """
        Updating Weights can depend on two factors:
        (i)     w1 = (cos(theta) + 1)/2 where theta is the angle between gps velocity vector and particle's velocity vector
        (ii)    w2 = how far the particle is from the current gps coordinates
        (iii)   w3 = change in distance from the GPS(compared to the previous time step)
        Updation:
        - calculate standardized w1, w2, and w3
        - Take the weighted sum: (a1 * w1 + a2 * w2  + a3 * w3)
        - Renormalization of all the weights
        """
        self._GPS_Update(gps_coordinates)
        a1 = 1/2
        a2 = 1/3
        a3 = 1/3
        w1, w2, w3 = [], [], []
        # TODO: Fix the loop
        for i, particle in enumerate(self.particles[current_time - 1]):
            # Factor 1:
            cos_theta = (self.gps_velocity @ (particle[4:7]).T) / (np.linalg.norm(self.gps_velocity) * np.linalg.norm(particle[4:7]))
            w1.append(cos_theta)
            
            # Factor 2:
            if current_time - 1 < 0:
                euclidean_dist_old = np.zeros((1,3))
                euclidean_dist_new = np.zeros((1,3))
            else:
                euclidean_dist_old = np.linalg.norm(self.particles[current_time - 3][i][:3] - self.particles[current_time - 2][i][:3])
                euclidean_dist_new = np.linalg.norm(self.particles[current_time - 2][i][:3] - self.particles[current_time - 1][i][:3])
            
            if euclidean_dist_new > self.standard_deviation * self.confidence_radius:
                w2.append(0)
            else:
                w2.append(1 / (self.standard_deviation * self.confidence_radius) * euclidean_dist_new)

            # Factor 3:
        w3.append((euclidean_dist_old - euclidean_dist_new) / euclidean_dist_old if euclidean_dist_old!=0 else (euclidean_dist_old - euclidean_dist_new)/0.001)


        w1 = self._normalization(w1)
        w2 = self._normalization(w2)
        w3 = self._normalization(w3)
        final_weights = []
        for i in range(len(w1)):
            I = w1[i]
            J = w2[i]
            K = w3[i]
            final_weights.append(a1 * I + a2 * J + a3 * K)
        
        self.particles[current_time-1][:,-1] = self._normalization(self.particles[current_time-1][:,-1])
    
    def _is_extreme(self, particle):
        """Extreme Particles: Outside the 3 sigma S.D. of the GPS
        Returns true if particle is extreme
        Else false
        """
        euclidean_distance = np.linalg.norm(particle - self.gps_current)
        if euclidean_distance > self.standard_deviation * self.confidence_radius:
            return True
        else:
            return False

    def filter_extremes(self, current_time):
        filter_out_indices = [] # indices that won't be filtered out will be saved here:
        for i, particle in enumerate(self.particles[current_time-1]):
            if self._is_extreme(particle) == True:
                filter_out_indices.append(i)
        self.particles[current_time - 1] = np.delete(self.particles[current_time - 1], filter_out_indices, axis=0)
        
    def adaptive_monte_carlo_resampling(self, current_time, resampling_noise_SD = 0.1, R1 = 0.5, R2 = 0.75):
        """
        Args: 
            - resampling_noise_SD: Standard deviation of resampling
            - R1: Percentage of points that survive in the first filteration.
            - R2: Percentage of points that survive in the second filteration.

        Implementation:
        Filter 1: Filter the top R1 particles(ranked in the order of weights)
        Filter 2: Out of these R2 particles, Filter random q particles

        Resampling(creating new particles): Add noise to the elements of state vector of a particle to create new particles from it
        Here we are creating two new particles from each filtered out particle.
        Resulting number of particles at each resampling = 0.75, we can tune R1 and R2 values to adjust this number.
        Converence speed will directly depend upon it. To account for harsher situations you can converge slower by increase R1, R2.
        """
        new_n_particles = np.ceil(R1 * R2 * self.n_particles)
        new_particles = np.zeros((self.resampling_frequency,new_n_particles, 7)) # For 10 time steps
        filtered_indices = []
        # Populating new_particles:
        sorted_weights = np.sort(self.particles[current_time - 1][:,-1])[:-1] # Descending Order
        sorted_weights = sorted_weights[:new_n_particles] # Clip the top R1
        # Filter 1
        for i, particle_state in enumerate(self.particles[current_time - 1]): # O(n^2) Operation
            if particle_state[-1] in sorted_weights:
                filtered_indices.append(i)
        filtered_particles = self.particles[current_time - 1][filtered_indices] # 2D array ONLY for the current timestep!!
        # Filter 2
        np.random.choice(filtered_particles, size = R2 * len(filtered_indices), replace=True)

        # We'll be creating two noisy successor points from each of the old points
        new_weight = 1/new_n_particles # all of them will bve the same
        for i, particle_state in enumerate(filtered_particles):
            for i in range(0,2):
                new_particles[current_time].append(np.array([
                    particle_state[0] + random.gauss(0,resampling_noise_SD),
                    particle_state[1] + random.gauss(0,resampling_noise_SD),
                    particle_state[2] + random.gauss(0,resampling_noise_SD),
                    particle_state[3] + random.gauss(0,resampling_noise_SD),
                    particle_state[4] + random.gauss(0,resampling_noise_SD),
                    particle_state[5] + random.gauss(0,resampling_noise_SD),
                    new_weight
                ]))
        self.n_particles = new_n_particles
        self.particles[current_time] = new_particles # TODO: Structure of particles class structure needs to be changed ig, make things compatible
        
class Visualization:
    def __init__(self):
        pass
    def particles_convergence(self):
        """
        Let's vizualize how the particles converged, by trailing individual particle's positions over time.
        """
        pass

def main():
    dt = 0.01
    a1,a2,a3 = 1,1,1
    dummy_A_matrix = np.ndarray(np.array([1,0,0,dt,0,0]),
                                np.array([0,1,0,0,dt,0]),
                                np.array([0,0,1,0,0,dt]),
                                np.array([1,0,0,a1,0,0]),
                                np.array([0,1,0,0,a2,0]),
                                np.array([0,0,1,0,0,a3]),
                                )

    n_steps = 10000
    rocket = Rocket(n_steps)
    gps_sensor = GPS(4000, n_steps)
    particle_filter = ParticleFilter()
    # TIME LOOP:
    for t in range(0, n_steps):
        rocket.clock += 1
        # Particles
        # particle_filter.state_progression(t, rocket.dynamics())   # Can't use this rn, because dynamics isn't debugged.
        particle_filter.state_progression(t, dummy_A_matrix)        # 
        file = open("particles.csv", mode="w")
        file.writelines(f"{particle_filter.particles[t]}")
        # GPS coordinates simulation
        gps_coodinates = gps_sensor.simulate_gps(t)
        
        weight_frequency = 2 # lower -> better but worse time complexity
        resampling_frequency = 10 # lower -> better but worse time complexity
        if t % weight_frequency == 0:
            particle_filter.update_weights(gps_coodinates)
        if t % resampling_frequency == 0:
            particle_filter.filter_extremes()
            particle_filter.adaptive_monte_carlo_resampling()

if __name__ == "__main__":
    main()

