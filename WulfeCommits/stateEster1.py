
import numpy as np
import random
from StateVisualization1 import *

# from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting

    # def visualize_particle_filter_full(self,gps_history, best_particle_history):
    #     # Convert histories to numpy arrays for easier slicing.
    #     gps_history = np.array(gps_history)
    #     best_particle_history = np.array(best_particle_history)
        
    #     # Split the arrays into positions, velocities, and accelerations.
    #     gps_pos = gps_history[:, 0:3]  # x, y, z positions
    #     gps_vel = gps_history[:, 3:6]  # x, y, z velocities
    #     gps_acc = gps_history[:, 6:9]  # x, y, z accelerations

    #     best_pos = best_particle_history[:, 0:3]
    #     best_vel = best_particle_history[:, 3:6]
    #     best_acc = best_particle_history[:, 6:9]
        
    #     # Create a time-step array (assuming each row represents one time step)
    #     time_steps = np.arange(gps_history.shape[0])
        
    #     # Set up the 3D figure and axis.
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
        
    #     # --- Plot Positions with a Time-Based Color Gradient ---
    #     # GPS positions: blue gradient.
    #     scatter_gps = ax.scatter(gps_pos[:, 0], gps_pos[:, 1], gps_pos[:, 2],
    #                             c=time_steps, cmap='Blues', label='GPS Positions', marker='o')
    #     ax.plot(gps_pos[:, 0], gps_pos[:, 1], gps_pos[:, 2],
    #             color='blue', alpha=0.3)
        
    #     # Best particle positions: red gradient.
    #     scatter_best = ax.scatter(best_pos[:, 0], best_pos[:, 1], best_pos[:, 2],
    #                             c=time_steps, cmap='Reds', label='Best Particle Positions', marker='^')
    #     ax.plot(best_pos[:, 0], best_pos[:, 1], best_pos[:, 2],
    #             color='red', alpha=0.3)
        
    #     # --- Plot Velocities as Arrows ---
    #     # Using quiver to draw velocity arrows at each position.
    #     # Adjust the arrow length as necessary.
    #     ax.quiver(gps_pos[:, 0], gps_pos[:, 1], gps_pos[:, 2],
    #             gps_vel[:, 0], gps_vel[:, 1], gps_vel[:, 2],
    #             length=0.2, normalize=True, color='cyan', label='GPS Velocities')
        
    #     ax.quiver(best_pos[:, 0], best_pos[:, 1], best_pos[:, 2],
    #             best_vel[:, 0], best_vel[:, 1], best_vel[:, 2],
    #             length=0.2, normalize=True, color='magenta', label='Best Particle Velocities')
        
    #     # --- Plot Accelerations as Arrows ---
    #     # Again using quiver to draw acceleration arrows.
    #     ax.quiver(gps_pos[:, 0], gps_pos[:, 1], gps_pos[:, 2],
    #             gps_acc[:, 0], gps_acc[:, 1], gps_acc[:, 2],
    #             length=0.2, normalize=True, color='green', label='GPS Accelerations')
        
    #     ax.quiver(best_pos[:, 0], best_pos[:, 1], best_pos[:, 2],
    #             best_acc[:, 0], best_acc[:, 1], best_acc[:, 2],
    #             length=0.2, normalize=True, color='orange', label='Best Particle Accelerations')
        
    #     # Add a colorbar for the GPS positions to indicate time progression.
    #     cbar = plt.colorbar(scatter_gps, ax=ax, pad=0.1)
    #     cbar.set_label('Time Step')
        
    #     # Label axes and add a title.
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #     ax.set_title('Particle Filter Visualizer with Velocities and Accelerations')
    #     print("shitting on them hoes")
        
    #     # Note: Legend entries for quiver plots may require custom handling if overlapping;
    #     # you can create proxy artists if needed.
    #     ax.legend(loc='upper left')
        
    #     plt.show()



class GPSwAccelerometer:
    def __init__(self, time_step):
        dt = .1
        self.num_steps = int(time_step / dt)
        self.time_step = time_step
        self.stateVector = np.zeros((self.num_steps,9))
        self.true_positions = np.zeros((self.num_steps, 3))  # (x, y, z) positions
        self.true_velocities = np.zeros((self.num_steps, 3))  # (vx, vy, yz)
        self.true_accelerations = np.zeros((self.num_steps, 3))
    
    def createStillData(self):
        # Reset true positions, velocities, and accelerations to zero
        self.true_positions.fill(0)
        self.true_velocities.fill(0)
        self.true_accelerations.fill(0)
        
        # Noise parameters
        gps_noise_std = 2.0  # GPS noise standard deviation (meters)
        accel_noise_std = 0.2  # Accelerometer noise standard deviation (m/s²)
        
        # Initialize sensor data arrays
        gps_positions = np.zeros((self.num_steps, 3))
        accelerometer_readings = np.zeros((self.num_steps, 3))
        
        for t in range(self.num_steps):
            # Simulate GPS data: true position (0,0,0) + noise
            gps_positions[t] = self.true_positions[t] + np.random.normal(0, gps_noise_std, 3)
            
            # Simulate Accelerometer data: gravity (0,0,9.81) + noise
            accelerometer_readings[t] = np.array([0.0, 0.0, 9.81]) + np.random.normal(0, accel_noise_std, 3)
        
        return (gps_positions, self.true_velocities, accelerometer_readings)
    
    def initializeVals(self):
        dt = .1 # Time step (seconds)
        total_time = self.time_step  # Total simulation time (seconds)
          # Number of steps

        # True State Variables (Position, Velocity, Acceleration)
  # (ax, ay, az)

        # Simulated Sensor Readings (GPS and Accelerometer)
        gps_positions = np.zeros((self.num_steps, 3))  # GPS noisy positions
        accelerometer_readings = np.zeros((self.num_steps, 3))  # Noisy accelerometer readings

        # Noise Parameters
        gps_noise_std = 2.0  # GPS noise (meters)
        accel_noise_std = 0.2  # Accelerometer noise (m/s²)

        # Initial Conditions
        self.true_positions[0] = [0, 0,0]  # Start at origin
        self.true_velocities[0] = [2, 1,1]  # Initial velocity (2 m/s right, 1 m/s up)
        self.true_accelerations[0] = [0.2, -0.1,0]  # Constant acceleration

        # Simulate Motion
        for t in range(1, self.num_steps):
            # Update velocity using acceleration
            self.true_velocities[t] = self.true_velocities[t-1] + self.true_accelerations[t-1] * dt
            
            # Update position using velocity
            self.true_positions[t] = self.true_positions[t-1] + self.true_velocities[t-1] * dt + 0.5 * self.true_accelerations[t-1] * dt**2
            
            # Simulate GPS readings (adding noise)
            gps_positions[t] = self.true_positions[t] + np.random.normal(0, gps_noise_std)
            
            # Simulate Accelerometer readings (true acceleration + noise)
            accelerometer_readings[t] = self.true_accelerations[t-1] + np.random.normal(0, accel_noise_std)

        return (gps_positions,self.true_velocities,accelerometer_readings)

    def clean_and_formulate(self,gps_positions,velocities,accelerometer_readings):
        gamma = 0.6
        for i in range(self.num_steps):
            for j in range(9):
                if j<3:
                    self.stateVector[i,j] = gps_positions[i,j]
                elif j<6:
                    self.stateVector[i,j]=velocities[i,j-3]
                elif j<8:
                    if i ==0:
                        self.stateVector[i,j]=accelerometer_readings[i,j-6]
                    else:
                        self.stateVector[i,j]=((gamma * accelerometer_readings[i,j-6])+ ((1-gamma)*self.stateVector[i-1,j]))
                else:
                    self.stateVector[i,j]=((gamma * accelerometer_readings[i,j-6])+ ((1-gamma)*self.stateVector[i-1,j]))-9.81
        return self.stateVector



class ParticleFilter:

    def __init__(self, N = None, GPSArray = None, GPS_accuracy_radius = None):
        self.N = 100
        self.std_dev = 2
        self.noise_std = 1
        #PARTICLE LENGTH CHANGED TO 9
        self.old_distances = np.zeros((1,self.N))
        self.new_distances = np.zeros((1,self.N))
        self.old_particles = np.zeros((self.N,9))
        self.particles_curr = np.zeros((self.N,9), dtype=float)
        self.weights = np.zeros((self.N,1))
        #REFRESH RATE (DT) IS SET TO ADJUSTED WHERE GT-U7 = MPU6050 10Hz = .1s
        self.refresh_rate = .1
        
        self.X_GPS_curr = np.array(GPSArray[0])
        self.X_GPS_old = np.array(GPSArray[0])
        #!!!! changed particle dimensions to just 3


    #TODO CRACKA!
    def initialize_particles(self, currState):
        for i in range(len(self.particles_curr)):
            for j in range(9):
                r=np.random.normal(loc=0,scale=2)
                if j<6:
                    self.particles_curr[i,j]=(self.particles_curr[i,j]+(self.particles_curr[i,j+3]*self.refresh_rate))+r
                else:
                    self.particles_curr[i,j]=currState[j]+r


    def calculate_theta(self, prev_pos, current_pos, particle_pos):
        # Calculate direction vectors
        prev_vector = np.array(current_pos).flatten() - np.array(prev_pos).flatten()
        particle_vector = np.array(particle_pos).flatten() - np.array(current_pos).flatten()
        
        # Normalize the vectors
        prev_vector_norm = prev_vector / np.linalg.norm(prev_vector)
        particle_vector_norm = particle_vector / np.linalg.norm(particle_vector)
        
        # Calculate the dot product
        dot_product = np.dot(prev_vector_norm, particle_vector_norm)
        
        # Clip the dot product to avoid numerical errors outside the range [-1, 1]
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Calculate the angle in radians and ensure it's a scalar
        theta = np.arccos(dot_product)

        return float(theta)

    def calculate_weight(self,particle_pos, prev_pos, current_pos, sigma_d = 10.0, kappa=1.0):
        d = np.linalg.norm(particle_pos-current_pos)
        #weighted by distance and angle with gaussian distribution
        distance_weight = np.exp(-d**2 / (2*sigma_d**2))
        #TODO
        theta = self.calculate_theta(prev_pos, current_pos, particle_pos)
        angle_weight = np.exp(-theta/kappa)

        return distance_weight * angle_weight
    
    def calculate_all_particle_weights(self,particles_curr, prev_pos, current_pos, sigma_d=10.0, kappa=1.0):

        for i in range(len(particles_curr)):
            self.weights[i] = self.calculate_weight(particles_curr[i], prev_pos, current_pos, 
                                sigma_d, kappa)
        self.weights += 1e-10  # Add a small constant to avoid underflow
        self.weights /= np.sum(self.weights)
        return self.weights


    def filterLowWeightParticles(self,threshold=.95):

            cumSum = 0
            counter=0
            sortedIndeces = np.argsort(self.weights)
            for i in range(len(sortedIndeces)):
                if (1 - cumSum) <=threshold:
                    break
                cumSum += self.weights[sortedIndeces[i]]
                counter+=1

            newParticleArray = np.ones((self.N,9))
            
            # We only want to copy N-counter particles (the ones we're keeping)
            for i in range(self.N - counter):
                newParticleArray[i+counter] = self.particles_curr[sortedIndeces[i+counter]]
            #from particlescurr of 0 to counter uniformally resample

            Vectoruniform_sample = np.random.uniform(low=0, high=1, size=(counter))
            cdfArray = np.zeros(len(self.particles_curr))
        
        # Compute the cumulative sum of the weights
            for i in range(len(self.particles_curr)):
                cdfArray[i] = self.weights[i] + (cdfArray[i - 1] if i > 0 else 0)

            indices = np.searchsorted(cdfArray, Vectoruniform_sample)

            #TODO Filter repopulation here is wrong, counter num of indeces of sortedIndeces[i] should be repopulated based on cdfArray
            for i in range(counter):
                r = np.random.normal(loc=1,scale=1)
                newParticleArray[i]=self.particles_curr[sortedIndeces[indices[i]]]+r

            weights_sum = sum(self.weights)
            newWeights = np.zeros(self.N)

            # Only copy weights for the particles we're keeping (N-counter particles)
            for i in range(self.N - counter):
                newWeights[i + counter] = self.weights[sortedIndeces[i + counter]]
            #from newWeights of 0 to counter uniformally resample
            #TODO RESAMPLED WEIGHTS ARE DISTRIBUTED EVENLY THEY SHOULD BE DISTRIBUTED BY A DIFFERENT PROPORTION BASED ON THEIR SAMPLE
            for i in range(counter):
                newWeights[i]=self.weights[indices[i]]  # or some other small constant value
            # Normalize all weights to sum to 1
            newWeights /= np.sum(newWeights)
            self.weights = newWeights
            self.particles_curr=newParticleArray
            
 #VELOCITIES BE UPDATED BY SOME COMBINATION OF (gamma)accel(dt) + (1-gamma) and central difference of differentiated gps
 #TODO: 
    def progress_particles(self, currState):
        for i in range(len(self.particles_curr)):
            for j in range(9):
                r=np.random.normal(loc=0,scale=2)
                if j<6:
                    self.particles_curr[i,j]=(self.particles_curr[i,j]+(self.particles_curr[i,j+3]*self.refresh_rate))+r
                else:
                    self.particles_curr[i,j]=currState[j]+r

                
            
        
          # Weighting factor for model vs. central difference                        


    def resampling(self):
        samplingThreshold = 2*len(self.particles_curr) / 3
        # print(f"Sampling threshold: {samplingThreshold}")

        # Initialize the cumulative distribution function (CDF)
        cdfArray = np.zeros(len(self.particles_curr))
        
        # Compute the cumulative sum of the weights
        for i in range(len(self.particles_curr)):
            cdfArray[i] = self.weights[i] + (cdfArray[i - 1] if i > 0 else 0)

        # Print the sum of squared weights (for debugging purpose)
        squaredWeights = 1/ np.sum(self.weights**2)
        # print(f"Squared weights: {squaredWeights}")
        
        if samplingThreshold > squaredWeights:
            print("we are resampling")
            rng = np.random.default_rng()  # Initialize the random number generator

            #TODO change resampled data to of self.N
            resampled_data = rng.uniform(low=0.0, high=1.0, size=int(self.N))
            # print("Resampled data:", resampled_data)
            
            # Use np.searchsorted to get the indices of the resampled particles
            indices = np.searchsorted(cdfArray, resampled_data)
            # print("Resampling indices:", indices)
            
            # Initialize new particles array with the correct size
            new_particles = np.zeros((len(resampled_data), 9))
            # Assign resampled particles
            # print("Selected resampled particles (before noise):")
            for i in range(len(resampled_data)):
                # Ensure the index is within bounds
                index = indices[i]  # Use modulo to wrap around if needed
                new_particles[i] = self.particles_curr[index]
                # print(f"Resampled particle {i}: {new_particles[i]}")

            # Add noise to resampled particles
            for i in range(len(new_particles)):       
            #TODO REPOPULATE WEIGHTS 
                # Update the particles with noise
                for j in range(9):
                    r= np.random.normal(loc=0, scale=self.std_dev)
                    new_particles[i][j] += r
                # print(f"Updated particle {i}: {new_particles[i]}")

            for i in range(len(new_particles)):
                self.weights[i]=1/self.N
            # Update the particles with the resampled and noise-modified particles
            self.particles_curr = new_particles
            # print("Final particles after resampling and noise addition:")
            # for i in range(len(self.particles_curr)):
                # print(f"Particle {i}: {self.particles_curr[i]}")
def main():
    random.seed(42)
    counter = 0
    gps_history = []
    best_particle_history = []
    total_particle_history = []
    total_weight_history = []
    avg_distance = []
    weightedavg_particle_history = []
    outputVector = np.zeros([9],dtype=np.float64)
    dataReadings = GPSwAccelerometer(time_step=110)
    positionData,velocityData,accelerationData=dataReadings.initializeVals()
    # positionData,velocityData,accelerationData = dataReadings.createStillData()
    dataVector=dataReadings.clean_and_formulate(positionData,velocityData,accelerationData)

    currParticleFilter = ParticleFilter(N=1000,GPSArray=dataVector,GPS_accuracy_radius=3)
    print("same shit")

    for i in range(400):
        print(f"we are at time step {i}")
        if i==0:
            currParticleFilter.initialize_particles(dataVector[0])
        else:
            currParticleFilter.progress_particles(dataVector[i])
            currParticleFilter.calculate_all_particle_weights(currParticleFilter.particles_curr, dataVector[i-1], dataVector[i])
            currParticleFilter.filterLowWeightParticles()
            currParticleFilter.resampling()     

            ok = np.argmax(currParticleFilter.weights)
            print("weight of particle: ",(max(currParticleFilter.weights)))
            print("Best particle's position: ",currParticleFilter.particles_curr[ok])
            print("gps coordinate",dataVector[i])
            print("distance of gps per best particle position", abs(currParticleFilter.particles_curr[ok]-dataVector[i]))
            gps_history.append(dataVector[i])
            best_particle_history.append(currParticleFilter.particles_curr[ok])
            total_particle_history.append(currParticleFilter.particles_curr)
            total_weight_history.append(currParticleFilter.weights)

            outputVector = np.zeros([9],dtype=np.float64)
            for i in range(len(currParticleFilter.particles_curr)):  # Iterate over all particles
                particle = currParticleFilter.particles_curr[i]
                # Ensure each particle has 9 dimensions
                if len(particle) != 9:
                    raise ValueError(f"Particle {i} does not have 9 dimensions: {particle}")
                
                for j in range(9):  # Iterate over each of the 9 dimensions of the particle
                    outputVector[j] += currParticleFilter.weights[i] * particle[j]

            weightedavg_particle_history.append(outputVector)
            print("\n\n\nWeighted Average Particle: ", outputVector)

            avg_distance.append(abs(currParticleFilter.particles_curr[ok]-dataVector[i]))
    
    accumulator=0

    for i in range(int(len(avg_distance)/3)):
        accumulator+=avg_distance[i]
    print("distance 1/3",accumulator/(len(avg_distance)/3))

    accumulator=0
    for i in range(int(len(avg_distance)/2)):
        accumulator+=avg_distance[i]
    
    print("distance 1/2",accumulator/(len(avg_distance)/3))

    accumulator=0

    for i in range(int(2*len(avg_distance)/3)):
        accumulator+=avg_distance[i]

    print("distance 2/3",accumulator/int(2*(len(avg_distance)/3)))

    accumulator=0
    for i in range(int(2*len(avg_distance)/3),int(len(avg_distance))):
        accumulator+=avg_distance[i]

    print("distance last 3rd",accumulator/(len(avg_distance)))

    gps_history = np.array(gps_history)
    best_particle_history = np.array(best_particle_history)
    total_weight_history = np.array(total_weight_history)
    weightedavg_particle_history = np.array(weightedavg_particle_history)
    total_particle_history = np.array(total_particle_history)

    print("gps shape",gps_history.shape)
    print("best particle history shape",best_particle_history.shape)
    print("total weight shape",total_weight_history.shape)
    print("weighted avg particle history shape",weightedavg_particle_history.shape)
    print( "total particle history shape",total_particle_history.shape)
    # dataReadings.visualize_particle_filter_full(gps_history,best_particle_history)
    print("\n\n\nWeighted Average Particle: ", weightedavg_particle_history)
    print("\n\n\nBest Weighted Particle: ", best_particle_history)

    #TODO avg weighted particle is broke asf, cant compute like that cracker!

    # animate_particle_filter(partilces, )
    #total_particles, best_weighted_particle, gps, total_wei
        
    #TODO slice and append a new numpy array of T,4 dimensions where 4 is the current weight

    new_array = np.concatenate([
        total_particle_history[..., :3],  # Take first 3 position values from (T, N, 9)
        total_weight_history[..., None]   # Add weights as 4th dimension (None adds new axis)
    ], axis=2)

    new_gps_array = np.concatenate([gps_history[:, :3]],axis=1)
    animate_particle_filter(new_array,new_gps_array)

if __name__=="__main__":
    main()
