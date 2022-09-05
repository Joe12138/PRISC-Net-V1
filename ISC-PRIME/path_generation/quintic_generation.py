import math
import numpy as np
import sys
sys.path.append("/home/joe/Desktop/Rule-PRIME/RulePRIME")
from path_generation.utils.polynomial_planner import QuinticPolynomial


def quintic_polynomial_planner(sx: float, sy: float, syaw: float, sv: float, sa: float,
                               gx: float, gy: float, gyaw: float, gv: float, ga: float,
                               dt: float = 0.1, T: float = 3):
    vxs = sv * math.cos(syaw)
    vys = sv * math.sin(syaw)
    vxg = gv * math.cos(gyaw)
    vyg = gv * math.sin(gyaw)

    axs = sa * math.cos(syaw)
    ays = sa * math.sin(syaw)
    axg = ga * math.cos(gyaw)
    ayg = ga * math.sin(gyaw)

    xqp = QuinticPolynomial(sx, vxs, axs, gx, vxg, axg, T)
    yqp = QuinticPolynomial(sy, vys, ays, gy, vyg, ayg, T)
    rx, ry = [], []

    for t in np.arange(dt, T + dt, dt):
        rx.append(xqp.calc_point(t))
        ry.append(yqp.calc_point(t))

    return rx, ry


def quintic_generation(sx: float, sy: float, svx: float, svy: float, sax: float, say: float,
                       gx: float, gy: float, gvx: float, gvy: float, gax: float, gay: float,
                       dt: float = 0.1, T: float = 3):
    xqp = QuinticPolynomial(sx, svx, sax, gx, gvx, gax, T)
    yqp = QuinticPolynomial(sy, svy, say, gy, gvy, gay, T)

    rx, ry = [], []

    for t in np.arange(dt, T + dt, dt):
        rx.append(xqp.calc_point(t))
        ry.append(yqp.calc_point(t))

    return rx, ry


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    sx, sy = 0.0, 0.0
    gx, gy = 10.0, 1.0
    
    sxv, syv = 1, 1
    sxa, sya = 1, 1
    
    gxv, gyv = 23.25, 0.75
    gxa, gya = 29.33, -0.6667
    
    xqp = QuinticPolynomial(sx, sxv, sxa, gx, gxv, gxa, 1)
    yqp = QuinticPolynomial(sy, syv, sya, gy, gyv, gya, 1)
    rx, ry = [], []

    for t in np.arange(0.1, 1.1, 0.1):
        rx.append(xqp.calc_point(t))
        ry.append(yqp.calc_point(t))
    
    plt.plot(rx, ry, color="red", marker="o")
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()
    