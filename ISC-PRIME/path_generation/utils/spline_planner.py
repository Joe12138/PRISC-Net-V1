import math
import numpy as np
import bisect


class CubicSpline:
    """
    Pairwise Cubic Splines, implemented by Numpy
    """

    def __init__(self, t_seq, x):
        self.t_seq = t_seq
        self.x = np.array(x)
        self.a, self.b, self.c, self.d = np.array(x), None, None, None

        self.nt = len(t_seq)  # dimension of input
        h = np.diff(t_seq)

        # calc spline coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h)
        self.c = np.linalg.solve(A, B)

        # calc spline coefficient b, d
        self.d = np.diff(self.c) / (3.0 * h)
        self.b = np.diff(self.a) / h - h * (self.c[1:] + 2 * self.c[:-1]) / 3.0

    def __calc_A(self, h):
        """
        calc matrix A for spline coefficient c
        """
        A = np.zeros((self.nt, self.nt))
        A[0, 0] = 1.0
        for i in range(self.nt - 1):
            if i != (self.nt - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nt - 1, self.nt - 2] = 0.0
        A[self.nt - 1, self.nt - 1] = 1.0
        return A

    def __calc_B(self, h):
        """
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nt)
        for i in range(self.nt - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / \
                       h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
        return B

    def __search_index(self, t):
        """
        search data segment index
        """
        return bisect.bisect(self.t_seq, t) - 1

    def calc_interval_indice(self, t):
        if t.size > 1:
            return [self.__search_index(ele) for ele in t]
        else:
            return self.__search_index(t)

    def calc_point(self, t, given_indice = None):
        """
        Calc position
        """
        # Mark: now the arc length range should be check outside this function.
        # # If t is outside of the input x, return None
        # if t < self.t_seq[0] or t > self.t_seq[-1]:
        #     return None

        # Otherwise, calculate the position using the corresponding spline piece.
        indice = self.calc_interval_indice(t) if given_indice is None else given_indice
        dx = t - self.t_seq[indice]

        if np.any(np.asarray(indice).astype(int) >= len(self.b)):
            indice_array = np.asarray(indice).astype(int)
            indice_array[indice_array >= len(self.b)] = len(self.b)-1
            b_indice = list(indice_array)
        else:
            b_indice = indice
        result = self.a[indice] + self.b[b_indice] * dx + \
                     self.c[indice] * dx ** 2.0 + self.d[b_indice] * dx ** 3.0
        return result

    def calc_first_derivative(self, t, given_indice = None):
        """
        Calc first derivative
        """

        # # If t is outside of the input x, return None
        # if t < self.t_seq[0] or t > self.t_seq[-1]:
        #     return None

        indice = self.calc_interval_indice(t) if given_indice is None else given_indice
        dx = t - self.t_seq[indice]
        if np.any(np.asarray(indice).astype(int) >= len(self.b)):
            indice_array = np.asarray(indice).astype(int)
            indice_array[indice_array >= len(self.b)] = len(self.b)-1
            b_indice = list(indice_array)
        else:
            b_indice = indice
        result = self.b[b_indice] + 2.0 * self.c[indice] * dx + 3.0 * self.d[b_indice] * dx ** 2.0
        return result

    def calc_second_derivative(self, t, given_indice = None):
        """
        Calc second derivative
        """

        # # If t is outside of the input x, return None
        # if t < self.t_seq[0] or t > self.t_seq[-1]:
        #     return None

        indice = self.calc_interval_indice(t) if given_indice is None else given_indice
        dx = t - self.t_seq[indice]
        if np.any(np.asarray(indice).astype(int) >= len(self.b)):
            indice_array = np.asarray(indice).astype(int)
            indice_array[indice_array >= len(self.b)] = len(self.b)-1
            b_indice = list(indice_array)
        else:
            b_indice = indice
        result = 2.0 * self.c[indice] + 6.0 * self.d[b_indice] * dx
        return result

    def calc_third_derivative(self, t, given_indice = None):
        """
        Calc third derivative
        """

        # # If t is outside of the input x, return None
        # if t < self.t_seq[0] or t > self.t_seq[-1]:
        #     return None

        indice = self.calc_interval_indice(t) if given_indice is None else given_indice
        if np.any(np.asarray(indice).astype(int) >= len(self.b)):
            indice_array = np.asarray(indice).astype(int)
            indice_array[indice_array >= len(self.b)] = len(self.b)-1
            b_indice = list(indice_array)
        else:
            b_indice = indice
        result = 6.0 * self.d[b_indice]
        return result


class CubicSpline2D:
    """
    2D Cubic Spline class
    """

    def __init__(self, x, y):
        # treat x, y as function of arc length s independently.
        self.s = self.__calc_s(x, y)
        self.sx = CubicSpline(self.s, x)
        self.sy = CubicSpline(self.s, y)

    def __calc_s(self, x, y):
        # calc the arc length along the lane
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = np.append(0.0, np.cumsum(self.ds))
        return s

    def calc_position(self, s):
        """
        calc position
        """
        x = self.sx.calc_point(s)
        y = self.sy.calc_point(s)
        return x, y

    def calc_yaw(self, s):
        """
        calc yaw
        """
        x_d = self.sx.calc_first_derivative(s)
        y_d = self.sy.calc_first_derivative(s)
        yaw = np.arctan2(y_d, x_d)
        return yaw

    def calc_curvature(self, s):
        """
        calc curvature
        """
        x_d = self.sx.calc_first_derivative(s)
        x_dd = self.sx.calc_second_derivative(s)
        y_d = self.sy.calc_first_derivative(s)
        y_dd = self.sy.calc_second_derivative(s)
        k = (y_dd * x_d - x_dd * y_d) / ((x_d ** 2 + y_d ** 2)**(3 / 2))
        return k

    def calc_curvature_deritative(self, s):
        """
        calc the first derivative of the curvature
        """
        x_d = self.sx.calc_first_derivative(s)
        x_dd = self.sx.calc_second_derivative(s)
        x_ddd = self.sx.calc_third_derivative(s)
        y_d = self.sy.calc_first_derivative(s)
        y_dd = self.sy.calc_second_derivative(s)
        y_ddd = self.sy.calc_third_derivative(s)
        k_prime = ((x_d*y_ddd-x_ddd*y_d) * (x_d**2+y_d**2) - 3 * (x_d*x_dd+y_d*y_dd) * (x_d*y_dd-x_dd*y_d)) \
                  * (x_d**2+y_d**2)**(-5/2)
        # MARK set rkappa_prime to be zero
        k_prime = np.zeros_like(k_prime)
        return k_prime


    def calc_all_in_single_forward(self, s):
        """
        calc (x, y, yaw, kappa, kappa_prime) by a single forward.
        Args:
            s - float / np.ndarray: the arc length along the centerline
        Return:
            all info exactly on the location of s.
        """

        # The t_seq is the same for sx and sy, it could be reused for all the following calculation.
        indices_of_s = self.sx.calc_interval_indice(s)

        x = self.sx.calc_point(s, indices_of_s)
        x_d = self.sx.calc_first_derivative(s, indices_of_s)
        x_dd = self.sx.calc_second_derivative(s, indices_of_s)
        x_ddd = self.sx.calc_third_derivative(s, indices_of_s)

        y = self.sy.calc_point(s, indices_of_s)
        y_d = self.sy.calc_first_derivative(s, indices_of_s)
        y_dd = self.sy.calc_second_derivative(s, indices_of_s)
        y_ddd = self.sy.calc_third_derivative(s, indices_of_s)

        yaw = np.arctan2(y_d, x_d)
        kappa = (y_dd * x_d - x_dd * y_d) / ((x_d ** 2 + y_d ** 2) ** (3 / 2))
        kappa_prime = ((x_d*y_ddd-x_ddd*y_d) * (x_d**2+y_d**2) - 3 * (x_d*x_dd+y_d*y_dd) * (x_d*y_dd-x_dd*y_d)) \
                  * (x_d**2+y_d**2) ** (-5/2)
        # MARK set rkappa_prime to be zero
        kappa_prime = np.zeros_like(kappa_prime)
        return x, y, yaw, kappa, kappa_prime



def calc_spline_course(x, y, ds=0.1):
    """
    calc the x, y, yaw, curvature
    alone the road where the arc length is discretized by ds
    """
    course = CubicSpline2D(x, y)
    rs = np.arange(0, course.s[-1], ds)
    rx, ry = course.calc_position(rs)
    ryaw = course.calc_yaw(rs)
    rk = course.calc_curvature(rs)
    rkprime = course.calc_curvature_deritative(rs)

    return rx, ry, ryaw, rk, rkprime