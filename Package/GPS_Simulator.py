import numpy as np
import random


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
            coordinate[0] = 0.0008 * t
            coordinate[1] = (-0.01) * (t - 632.47)**2 + 4000
            coordinate[2] = 0.0008 * t
        return self.path
    
    def simulate_gps(self, current_time):
        """
        Simulates coordinates from the GPS output, when the rocket is on `self.path`.
        Args:
            - current_time: needs to be between [1, total_steps]
        Working:
            - returns self.path but with added gasussian noise
            - if not self.path then runs _parabolic_path() to populate self.path
        """
        if not self.path:
            self.path = np.zeros((self.n_steps, 3))
            self._parabolic_path()

        rocket_coordinates = self.path[current_time-1]
        gps_coordinates = np.array( (rocket_coordinates[0] + random.gauss(0, self.accuracy_radius)),
                                    (rocket_coordinates[1] + random.gauss(0, self.accuracy_radius)),    
                                    (rocket_coordinates[2] + random.gauss(0, self.accuracy_radius))
                                   )
        # we'll use self.path as a lookup table
        return gps_coordinates
