import numpy as np
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class GPS:
    def __init__(self, apogee, total_steps, accuracy_radius):
        """
        All units are in metric - SI(standard international)
        """
        self.apogee = apogee
        self.n_steps = total_steps
        self.accuracy_radius = 5
        self.path = None

    def _parabolic_path(self):
        """
        Populates the self.path variable.
        So that you can then simulate GPS output on it.
        """
        for t, coordinate in enumerate(self.path):
            coordinate[0] = t
            coordinate[1] = (-0.001) * (t - 2000)**2 + self.apogee
            coordinate[2] = 0.0
        return self.path

    def simulate_gps(self, current_time):
        """
        Simulates coordinates from the GPS output, when the rocket is on `self.path`.
        Args:
        - current_time: needs to be between [1, total_steps]
        """
        if self.path is None:
            self.path = np.zeros((self.n_steps, 3))
            self._parabolic_path()

        rocket_coordinates = self.path[current_time-1]
        print(random.gauss(0, self.accuracy_radius))
        gps_coordinates = np.array([
            (rocket_coordinates[0] + random.gauss(0, self.accuracy_radius)),
            (rocket_coordinates[1] + random.gauss(0, self.accuracy_radius)),
            (rocket_coordinates[2] + random.gauss(0, self.accuracy_radius))
        ])
        return gps_coordinates

def main():
    total_time_steps = 4000
    apogee = 4000
    gps_sensor = GPS(apogee, total_time_steps, 5)
    
    # Simulate GPS coordinates over time
    gps_coordinates = []
    for t in range(total_time_steps):
        gps_coordinates.append(gps_sensor.simulate_gps(t))
    gps_coordinates = np.array(gps_coordinates)

    # Create the 3D plot using Plotly
    fig = go.Figure()

    # Add the true path
    fig.add_trace(go.Scatter3d(
        x=gps_sensor.path[:, 0],
        y=gps_sensor.path[:, 1],
        z=gps_sensor.path[:, 2],
        mode='lines',
        name='True Path',
        line=dict(color='red', width=4),
    ))

    # Add the GPS simulated path
    fig.add_trace(go.Scatter3d(
        x=gps_coordinates[:, 0],
        y=gps_coordinates[:, 1],
        z=gps_coordinates[:, 2],
        mode='markers',
        name='GPS Path with Noise',
        marker=dict(
            size=2,
            color='blue',
            opacity=0.6
        )
    ))

    # Update the layout with improved styling
    fig.update_layout(
        title={
            'text': "3D Rocket Path: True vs GPS Simulated",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        scene=dict(
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate",
            zaxis_title="Z Coordinate",
            aspectmode='cube',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            xaxis=dict(gridcolor='lightgray'),
            yaxis=dict(gridcolor='lightgray'),
            zaxis=dict(gridcolor='lightgray')
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        template="plotly_dark"
    )

    # Show the interactive plot
    fig.show()

if __name__ == "__main__":
    main()