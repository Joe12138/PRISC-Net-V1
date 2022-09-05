import numpy as np


class QuinticPolynomial:
    """
    Quintic Polynomial class
    """

    def __init__(self, x0, v0, a0, x1, v1, a1, time):
        # Given the starting and end points information, and variable t,
        # calcuate the coefficients of quintic polynomial
        self.a0 = x0
        self.a1 = v0
        self.a2 = a0 / 2.0

        A = np.array([[time ** 3, time ** 4, time ** 5],
                      [3 * time ** 2, 4 * time ** 3, 5 * time ** 4],
                      [6 * time, 12 * time ** 2, 20 * time ** 3]])
        b = np.array([x1 - self.a0 - self.a1 * time - self.a2 * time ** 2,
                      v1 - self.a1 - 2 * self.a2 * time,
                      a1 - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        x = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5
        return x

    def calc_first_derivative(self, t):
        x_d = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4
        return x_d

    def calc_second_derivative(self, t):
        x_dd = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3
        return x_dd

    def calc_third_derivative(self, t):
        x_ddd = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2
        return x_ddd


class QuarticPolynomial:
    """
    Quartic Polynomial class
    """
    def __init__(self, x0, v0, a0, v1, a1, time):
        # Given the starting point (x, v, a) and end point (a, v), and variable t,
        # calcuate the coefficients of quintic polynomial
        self.a0 = x0
        self.a1 = v0
        self.a2 = a0 / 2.0

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([v1 - self.a1 - 2 * self.a2 * time,
                      a1 - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        x = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4
        return x

    def calc_first_derivative(self, t):
        x_d = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3
        return x_d

    def calc_second_derivative(self, t):
        x_dd = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2
        return x_dd

    def calc_third_derivative(self, t):
        x_ddd = 6 * self.a3 + 24 * self.a4 * t
        return x_ddd