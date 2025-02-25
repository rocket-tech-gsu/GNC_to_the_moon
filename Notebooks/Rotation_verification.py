import numpy as np
import matplotlib.pyplot as plt

# Define the yaw rotation matrix (rotation about the Z-axis)
def yaw(angle):
    """
    Returns the yaw rotation matrix (rotation about the z-axis).
    
    Parameters:
        angle (float): Rotation angle in radians.
        
    Returns:
        numpy.ndarray: 3x3 yaw rotation matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [ c, -s, 0],
        [ s,  c, 0],
        [ 0,  0, 1]
    ])

# Define the pitch rotation matrix (rotation about the Y-axis)
def pitch(angle):
    """
    Returns the pitch rotation matrix (rotation about the y-axis).
    
    Parameters:
        angle (float): Rotation angle in radians.
        
    Returns:
        numpy.ndarray: 3x3 pitch rotation matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [ c,  0, s],
        [ 0,  1, 0],
        [-s,  0, c]
    ])

# Define the roll rotation matrix (rotation about the X-axis)
def roll(angle):
    """
    Returns the roll rotation matrix (rotation about the x-axis).
    
    Parameters:
        angle (float): Rotation angle in radians.
        
    Returns:
        numpy.ndarray: 3x3 roll rotation matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1,  0,  0],
        [0,  c, -s],
        [0,  s,  c]
    ])

# Main execution: compute the transformed vectors and plot them
if __name__ == "__main__":
    # Define the original vector (customize as needed)
    original_vector = np.array([5, 2, 3])
    
    # Define rotation angles (in radians)
    yaw_angle   = np.radians(30)   # 30° yaw rotation about the Z-axis
    pitch_angle = np.radians(45)   # 45° pitch rotation about the Y-axis
    roll_angle  = np.radians(60)   # 60° roll rotation about the X-axis

    un_yaw_angle   = np.radians(-30) 
    un_pitch_angle = np.radians(-45) 
    un_roll_angle  = np.radians(-60) 
    
    # Calculate the rotation matrices
    yaw_matrix   = yaw(yaw_angle)
    pitch_matrix = pitch(pitch_angle)
    roll_matrix  = roll(roll_angle)
    
    # Calculate the UN rotation matrices
    un_yaw_matrix   = yaw(un_yaw_angle)
    un_pitch_matrix = pitch(un_pitch_angle)
    un_roll_matrix  = roll(un_roll_angle)
    
    # Compute the transformed vectors
    vector_yaw   = yaw_matrix.dot(original_vector)
    vector_pitch = pitch_matrix.dot(vector_yaw)
    vector_roll  = roll_matrix.dot(vector_pitch)
 
    vector_un_roll  = un_roll_matrix.dot(vector_roll)
    vector_un_pitch = un_pitch_matrix.dot(vector_un_roll)
    vector_un_yaw   = un_yaw_matrix.dot(vector_un_pitch)
    
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    origin = [0, 0, 0]  # Starting point for all vectors
    
    # Plot the original vector (blue)
    ax.quiver(origin[0], origin[1], origin[2],
              original_vector[0], original_vector[1], original_vector[2],
              color='blue', linewidth=2, arrow_length_ratio=0.1, label='Original Vector')
    
    # # Plot the yaw-rotated vector (red)
    # ax.quiver(origin[0], origin[1], origin[2],
    #           vector_yaw[0], vector_yaw[1], vector_yaw[2],
    #           color='red', linewidth=2, arrow_length_ratio=0.1, label='Yaw Rotation')
    
    # # Plot the pitch-rotated vector (green)
    # ax.quiver(origin[0], origin[1], origin[2],
    #           vector_pitch[0], vector_pitch[1], vector_pitch[2],
    #           color='green', linewidth=2, arrow_length_ratio=0.1, label='Pitch Rotation')
    
    # Plot the roll-rotated vector (orange)
    ax.quiver(origin[0], origin[1], origin[2],
              vector_roll[0], vector_roll[1], vector_roll[2],
              color='orange', linewidth=2, arrow_length_ratio=0.1, label='Roll Rotation')

    # Un - rolled > un - pitched > un - yawed:
    ax.quiver(origin[0], origin[1], origin[2],
              vector_un_yaw[0], vector_un_yaw[1], vector_un_yaw[2],
              color='red', linewidth=2, arrow_length_ratio=0.1, label='Roll Rotation')
    

    
    # Determine an appropriate axis limit based on all vector lengths
    all_vectors = np.array([original_vector, vector_yaw, vector_pitch, vector_roll])
    max_val = np.abs(all_vectors).max() * 1.2
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])
    
    # Labeling the axes and plot
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Original and Rotated Vectors (Yaw, Pitch, Roll)')
    ax.legend()
    
    plt.show()
