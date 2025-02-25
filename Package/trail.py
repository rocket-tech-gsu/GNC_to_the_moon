import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import time

# Generate sample motion data (helix motion)
t = np.linspace(0, 4 * np.pi, 100)  # Time steps
x = np.cos(t)
y = np.sin(t)
z = t / (4 * np.pi)  # Normalized height

# Number of trail points to keep visible
trail_length = 15  

# Create the figure
fig = go.Figure()

# Initialize the moving point trace
fig.add_trace(go.Scatter3d(
    x=[x[0]], y=[y[0]], z=[z[0]],
    mode='markers',
    marker=dict(size=6, color='red'),
    name='Moving Point'
))

# Initialize the trailing line trace
fig.add_trace(go.Scatter3d(
    x=[], y=[], z=[],
    mode='lines',
    line=dict(color='blue', width=4),
    name='Trailing Edge'
))

# Layout settings
fig.update_layout(
    title="Animated 3D Line with Fading Trailing Edge",
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis'
    ),
    updatemenus=[dict(type="buttons",
                      showactive=False,
                      buttons=[dict(label="Play",
                                    method="animate",
                                    args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)])])]
)

# Frames for animation
frames = []
for i in range(len(t)):
    trail_start = max(0, i - trail_length)  # Keep a limited trail length
    
    frames.append(go.Frame(
        data=[
            go.Scatter3d(
                x=[x[i]], y=[y[i]], z=[z[i]],
                mode='markers',
                marker=dict(size=6, color='red')
            ),
            go.Scatter3d(
                x=x[trail_start:i+1], 
                y=y[trail_start:i+1], 
                z=z[trail_start:i+1],
                mode='lines',
                line=dict(
                    width=4, 
                    color=[f'rgba(0, 0, 255, {alpha})' for alpha in np.linspace(0.1, 1, i - trail_start + 1)]
                )
            )
        ]
    ))

# Add animation settings
fig.update(frames=frames)

# Show the animated plot
pio.show(fig)
