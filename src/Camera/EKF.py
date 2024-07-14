import numpy as np

class EKF:
    def __init__(self, dt, a, b, Q, R, initial_state, initial_covariance):
        self.dt = dt
        self.a = a
        self.b = b
        self.Q = Q
        self.R = R
        self.X = initial_state
        self.P = initial_covariance

    def wrap_to_pi(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def predict(self, v, delta):
        theta = self.X[2]
        alpha = np.arctan(self.a * np.tan(delta) / self.b)

        self.X[0] += v * np.cos(alpha + theta) * self.dt
        self.X[1] += v * np.sin(alpha + theta) * self.dt
        self.X[2] += (v / self.b) * np.tan(delta) * self.dt
        self.X[2] = self.wrap_to_pi(self.X[2])

        Fx = np.array([
            [1, 0, -v * np.sin(alpha + theta) * self.dt],
            [0, 1,  v * np.cos(alpha + theta) * self.dt],
            [0, 0,  1]
        ])

        Fw = np.array([
            [np.cos(alpha + theta) * self.dt, 0],
            [np.sin(alpha + theta) * self.dt, 0],
            [0, self.dt]
        ])

        self.P = Fx @ self.P @ Fx.T + Fw @ self.Q @ Fw.T

    def update(self, z):
        x, y, theta = self.X

        h = np.array([
            x,
            y,
            theta
        ])

        Hx = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        y = z - h

        y[2] = self.wrap_to_pi(y[2])

        S = Hx @ self.P @ Hx.T + self.R
        K = self.P @ Hx.T @ np.linalg.inv(S)

        self.X = self.X + K @ y

        self.P = (np.eye(3) - K @ Hx) @ self.P

    def get_state(self):
        return self.X