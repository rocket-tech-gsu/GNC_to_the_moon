import numpy as np
import plotly.graph_objects as go

# ----- Utility function to scale opacities for particles -----
def scale_opacity(weight, w_min, w_max, min_op=0.01, max_op=0.25):
    """Linearly scale a weight into an opacity value between min_op and max_op."""
    if w_max - w_min == 0:
        return max_op
    return min_op + (weight - w_min) / (w_max - w_min) * (max_op - min_op)

# ----- Main Animation Function -----
# def animate_particle_filter(particles, gps, trail_length=20):
#     """
#     Creates an animated 3D visualization of a particle filter.
    
#     Parameters:
#       - particles: np.array of shape (T, N, 4)
#           For each time step T and each particle (of N particles),
#           the first three entries are x, y, z and the 4th is weight.
#       - gps: np.array of shape (T, 3)
#           Each row is a GPS coordinate.
#       - trail_length: int (default=20)
#           Number of previous time steps to show as a fading trail.
#     """
#     T, N, _ = particles.shape
#     frames = []

#     # Compute overall axis ranges based on the entire particles data (with a margin)
#     x_min, x_max = np.min(particles[:, :, 0]), np.max(particles[:, :, 0])
#     y_min, y_max = np.min(particles[:, :, 1]), np.max(particles[:, :, 1])
#     z_min, z_max = np.min(particles[:, :, 2]), np.max(particles[:, :, 2])
#     margin = 1  # Increase this value if you want more padding around your data
#     a = max([np.absolute(x_min - margin), np.absolute(y_min - margin), np.absolute(z_min - margin)])
#     b = max([np.absolute(x_max + margin), np.absolute(y_max + margin), np.absolute(z_max + margin)])
#     m = max([a,b])
#     # Pre-compute the fixed scene layout (axes fixed and no autoscaling)
#     fixed_scene = dict(
#         xaxis=dict(range=[-m, m], autorange=False),
#         yaxis=dict(range=[-m, m], autorange=False),
#         zaxis=dict(range=[-m, m], autorange=False),
#         aspectmode='manual',
#         aspectratio=dict(x=1, y=1, z=1)
#     )

#     # Build each frame’s data as a list of traces.
#     for t in range(T):
#         frame_data = []

#         # --- Determine highest weighted particle at this time step ---
#         weights = particles[t, :, 3]
#         max_index = int(np.argmax(weights))
        
#         # --- Plot non-highlighted particles (blue) ---
#         # Exclude the best particle (which is plotted separately)
#         non_idx = [i for i in range(N) if i != max_index]
#         if non_idx:
#             pos = particles[t, non_idx, :3]
#             w_non = particles[t, non_idx, 3]
#             # Scale opacities between 1% and 25%
#             w_min = np.min(w_non)
#             w_max = np.max(w_non)
#             colors = [
#                 f'rgba(0, 0, 255, {scale_opacity(w, w_min, w_max):.3f})'
#                 for w in w_non
#             ]
#             scatter_non = go.Scatter3d(
#                 x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
#                 mode='markers',
#                 marker=dict(size=4, color=colors),
#                 name='Particles'
#             )
#             frame_data.append(scatter_non)
        
#         # --- Plot highlighted (best) particle (red, full opacity) ---
#         pos_best = particles[t, max_index, :3]
#         scatter_best = go.Scatter3d(
#             x=[pos_best[0]], y=[pos_best[1]], z=[pos_best[2]],
#             mode='markers',
#             marker=dict(size=6, color='red', opacity=1),
#             name='Best Particle'
#         )
#         frame_data.append(scatter_best)
        
#         # --- Plot current GPS reading (green) ---
#         pos_gps = gps[t]
#         scatter_gps = go.Scatter3d(
#             x=[pos_gps[0]], y=[pos_gps[1]], z=[pos_gps[2]],
#             mode='markers',
#             marker=dict(size=6, color='green', opacity=1),
#             name='GPS'
#         )
#         frame_data.append(scatter_gps)
        
#         # --- Draw trail for Best Particle (fading red line) ---
#         start = max(0, t - trail_length + 1)
#         trail_points = particles[start:t+1, max_index, :3]
#         num_segments = len(trail_points) - 1
#         if num_segments > 0:
#             # Draw each segment separately with increasing opacity for newer segments
#             for seg in range(num_segments):
#                 # Newer segments are more opaque (opacity goes from 0.2 to 1.0)
#                 seg_op = 0.2 + 0.8 * ((seg + 1) / num_segments)
#                 seg_trace = go.Scatter3d(
#                     x=[trail_points[seg, 0], trail_points[seg+1, 0]],
#                     y=[trail_points[seg, 1], trail_points[seg+1, 1]],
#                     z=[trail_points[seg, 2], trail_points[seg+1, 2]],
#                     mode='lines',
#                     line=dict(color=f'rgba(255, 0, 0, {seg_op:.3f})', width=4),
#                     showlegend=False
#                 )
#                 frame_data.append(seg_trace)
        
#         # --- Draw trail for GPS (fading green line) ---
#         start = max(0, t - trail_length + 1)
#         trail_gps = gps[start:t+1]
#         num_segments_gps = len(trail_gps) - 1
#         if num_segments_gps > 0:
#             for seg in range(num_segments_gps):
#                 seg_op = 0.2 + 0.8 * ((seg + 1) / num_segments_gps)
#                 seg_trace = go.Scatter3d(
#                     x=[trail_gps[seg, 0], trail_gps[seg+1, 0]],
#                     y=[trail_gps[seg, 1], trail_gps[seg+1, 1]],
#                     z=[trail_gps[seg, 2], trail_gps[seg+1, 2]],
#                     mode='lines',
#                     line=dict(color=f'rgba(0, 255, 0, {seg_op:.3f})', width=4),
#                     showlegend=False
#                 )
#                 frame_data.append(seg_trace)
        
#         frames.append(go.Frame(data=frame_data, name=str(t)))
    
#     # Create the initial figure from the first frame.
#     fig = go.Figure(
#         data=frames[0].data,
#         frames=frames
#     )
    
#     # Update layout with fixed axis ranges and disable autoscaling
#     fig.update_layout(
#         scene=fixed_scene,
#         title="3D Particle Filter Visualization",
#         uirevision='constant',  # prevents re-rendering of the scene (i.e. keeps axes and camera fixed)
#         updatemenus=[
#             dict(
#                 type="buttons",
#                 showactive=False,
#                 buttons=[dict(label="Play",
#                               method="animate",
#                               args=[None, {"frame": {"duration": 100, "redraw": True},
#                                            "fromcurrent": True,
#                                            "transition": {"duration": 0}}]
#                               )]
#             )
#         ]
#     )
    
#     # Add an optional slider to scrub through frames
#     sliders = [{
#         "currentvalue": {"prefix": "Time step: "},
#         "pad": {"t": 50},
#         "steps": [{
#             "args": [[str(k)], {"frame": {"duration": 100, "redraw": True},
#                                 "mode": "immediate",
#                                 "transition": {"duration": 0}}],
#             "label": str(k),
#             "method": "animate"
#         } for k in range(T)]
#     }]
#     fig.update_layout(sliders=sliders)
    
#     fig.show()
def animate_particle_filter(particles, gps, trail_length=20):
    """
    Creates an animated 3D visualization of a particle filter with fading trails.
    
    Parameters:
      - particles: np.array of shape (T, N, 4)
          For each time step T and each particle (of N particles),
          the first three entries are x, y, z and the 4th is weight.
      - gps: np.array of shape (T, 3)
          Each row is a GPS coordinate.
      - trail_length: int (default=20)
          Number of previous time steps to show as a fading trail.
    """
    T, N, _ = particles.shape
    frames = []

    # Compute overall axis ranges based on the entire particles data (with a margin)
    x_min, x_max = np.min(particles[:, :, 0]), np.max(particles[:, :, 0])
    y_min, y_max = np.min(particles[:, :, 1]), np.max(particles[:, :, 1])
    z_min, z_max = np.min(particles[:, :, 2]), np.max(particles[:, :, 2])
    margin = 1
    a = max([np.absolute(x_min - margin), np.absolute(y_min - margin), np.absolute(z_min - margin)])
    b = max([np.absolute(x_max + margin), np.absolute(y_max + margin), np.absolute(z_max + margin)])
    m = max([a, b])
    fixed_scene = dict(
        xaxis=dict(range=[-m, m], autorange=False),
        yaxis=dict(range=[-m, m], autorange=False),
        zaxis=dict(range=[-m, m], autorange=False),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=1)
    )

    for t in range(T):
        frame_data = []

        # --- Highlighted particles and GPS ---
        weights = particles[t, :, 3]
        max_index = np.argmax(weights)
        
        # Non-highlighted particles
        non_idx = [i for i in range(N) if i != max_index]
        if non_idx:
            pos = particles[t, non_idx, :3]
            w_non = particles[t, non_idx, 3]
            w_min, w_max = np.min(w_non), np.max(w_non)
            colors = [f'rgba(0, 0, 255, {scale_opacity(w, w_min, w_max):.3f})' for w in w_non]
            scatter_non = go.Scatter3d(
                x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
                mode='markers',
                marker=dict(size=4, color=colors),
                name='Particles'
            )
            frame_data.append(scatter_non)
        
        # Best particle
        pos_best = particles[t, max_index, :3]
        scatter_best = go.Scatter3d(
            x=[pos_best[0]], y=[pos_best[1]], z=[pos_best[2]],
            mode='markers',
            marker=dict(size=6, color='red', opacity=1),
            name='Best Particle'
        )
        frame_data.append(scatter_best)
        
        # GPS
        pos_gps = gps[t]
        scatter_gps = go.Scatter3d(
            x=[pos_gps[0]], y=[pos_gps[1]], z=[pos_gps[2]],
            mode='markers',
            marker=dict(size=6, color='green', opacity=1),
            name='GPS'
        )
        frame_data.append(scatter_gps)
        
        # --- Fading trails for Best Particle and GPS ---
        # Best particle trail (red)
        start = max(0, t - trail_length + 1)
        trail_points = particles[start:t+1, max_index, :3]
        num_points = len(trail_points)
        if num_points >= 2:
            alphas = np.linspace(0.1, 1, num_points)
            colors = [f'rgba(255, 0, 0, {alpha:.3f})' for alpha in alphas]
            trail_trace = go.Scatter3d(
                x=trail_points[:, 0],
                y=trail_points[:, 1],
                z=trail_points[:, 2],
                mode='lines',
                line=dict(color=colors, width=4),
                showlegend=False
            )
            frame_data.append(trail_trace)
        
        # GPS trail (green)
        trail_gps = gps[start:t+1]
        num_points_gps = len(trail_gps)
        if num_points_gps >= 2:
            alphas_gps = np.linspace(0.1, 1, num_points_gps)
            colors_gps = [f'rgba(0, 255, 0, {alpha:.3f})' for alpha in alphas_gps]
            trail_gps_trace = go.Scatter3d(
                x=trail_gps[:, 0],
                y=trail_gps[:, 1],
                z=trail_gps[:, 2],
                mode='lines',
                line=dict(color=colors_gps, width=4),
                showlegend=False
            )
            frame_data.append(trail_gps_trace)
        
        frames.append(go.Frame(data=frame_data, name=str(t)))
    
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        scene=fixed_scene,
        title="3D Particle Filter Visualization",
        uirevision='constant',
        updatemenus=[dict(  # Separate updatemenus list
            type="buttons",
            showactive=False,
            buttons=[dict(label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 100, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0}}]
                        )]
        )],
        sliders=[{  # Separate sliders list (not nested inside updatemenus)
            "currentvalue": {"prefix": "Time step: "},
            "pad": {"t": 50},
            "steps": [{
                "args": [[str(k)], {"frame": {"duration": 100, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
                "label": str(k),
                "method": "animate"
            } for k in range(T)]
        }]
    )
    fig.show()
# ----- Example: Generate sample data for testing -----
def generate_sample_data(T=150, N=300, radius=50, sigma=4.0):
    """
    Generate sample data:
      - particles: follow a Gaussian distribution around the GPS position
                   at each time step, with random weights.
      - gps: positions following a large circular trajectory.

    Parameters:
      - T: int, number of time steps
      - N: int, number of particles
      - radius: float, radius of the circular GPS trajectory
      - sigma: float, standard deviation of the Gaussian distribution
               around the GPS position
    """
    # Generate GPS data following a circular trajectory
    theta = np.linspace(0, 2 * np.pi, T, endpoint=False)  # Angle parameter
    gps = np.zeros((T, 3))
    gps[:, 0] = ( radius * np.cos(theta)) # X-coordinates
    gps[:, 1] = ( radius * np.sin(theta)) # Y-coordinates
    gps[:, 2] = 0                         # Z-coordinates (assuming a flat circle in the XY plane)

    # Initialize particles array
    particles = np.zeros((T, N, 4))
    for t in range(T):
        # Particles' positions follow a Gaussian distribution around the GPS position
        particles[t, :, :3] = gps[t] + np.random.normal(0, sigma, (N, 3))
        # Assign random weights to particles
        particles[t, :, 3] = np.random.uniform(0, 1, N)
    
    return particles, gps

# ----- Main entry point -----
if __name__ == "__main__":
    # Replace generate_sample_data() with your actual data arrays if needed.
    particles, gps = generate_sample_data(T=100, N=40)
    animate_particle_filter(particles, gps, trail_length=20)
