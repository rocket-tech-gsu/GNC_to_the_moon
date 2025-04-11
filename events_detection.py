import numpy as np
import matplotlib.pyplot as plt

# Uncomment the next line if you want to force interactive mode
# plt.ion()

# Define parabola parameters and generate data
a, b, c = 2, 0, 0  # Parabola parameters for y = -a*(x-shift)^2 (concave down)
shift = 100
x_values = np.linspace(0, 2*shift, 2000)

# Generate noisy points on the concave parabola
points = []
# Pressure data:
old_window = []
new_window = []


# APOGEE
apogee = False
counter = 0
star_x = 0
star_y = 0
window_size = 10

# LAUNCH
launch = False
launch_window_size = 0
launch_counter = 0

############################ FLIGHT LOOP ############################
for j,x in enumerate(x_values):
    # Compute the parabola value and add Gaussian noise with std=4
    y = -a * (x - shift) ** 2 + np.random.normal(0, 10)
    points.append((x, y))
 


    ######### APOGEE DETECTION #########

    if apogee == False:
        try:
            old_window = [float(i[1]) for i in points[- 2 * window_size: - window_size]]
            new_window = [float(i[1]) for i in points[-window_size:]]
            if j%100 == 0:
                print("Old Window = ",old_window)
                print("New Window = ",new_window)
            A_old = sum(old_window) / window_size
            A_new = sum(new_window) / window_size
            
            if A_new < A_old:
                counter += 1
                print("New Average = ", A_new)
                print("Old Average = ", A_old)
                print(j)
                if counter > 21:
                    print("\n\n\n APOGEE DETECTED \n\n\n")
                    apogee = True
                    star_x, star_y = x, y

            else:
                counter = 0
        except:
            pass
    else:
        pass
    
    ######### LANDING DETECTION #########
    

    # ######### LAUNCH DETECTION #########
    # if launch == False:
    #     try:
    #         alt_window_old = [float(i[1]) for i in points[-window_size:]]
    #         A_avg_old = sum(alt_window_old) / window_size
    #         launch_counter += 1
    #         if launch_counter >

    #     except:
    #         pass

# Separate x and y coordinates for noisy points
x_points, y_points = zip(*points)

# Compute the noise-free parabola for reference
y_deterministic = -a * (x_values - shift) ** 2

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_deterministic, label="y = -x² (deterministic)", color='b')
plt.scatter(x_points, y_points, color='r', s=10, label="Noisy Points")

# Plot a star marker at a sample location (adjust as needed)
# star_x, star_y = 0, -1
plt.scatter(star_x, star_y, color='gold', marker='*', s=200,
            edgecolors='black', linewidth=1.5, label="Star Marker")

# Add grid, legend, and labels
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.title("Concave Parabola with Noise")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

# Display the plot
plt.show()

# Optional: Keep the window open until you press Enter (if needed)
input("Press Enter to exit...")
