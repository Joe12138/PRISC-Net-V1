import math

import numpy as np


def compute_path_profile(xy_point: np.ndarray):
    heading = []
    accumulated_s = []
    kappas = []
    dkappas = []

    if xy_point.shape[0] < 2:
        raise Exception("This reference points too less!")

    dxs, dys = [], []
    y_over_s_first_derivatives = []
    x_over_s_first_derivatives = []
    y_over_s_second_derivatives = []
    x_over_s_second_derivatives = []

    point_size = xy_point.shape[0]

    for i in range(point_size):
        if i == 0:
            x_delta = xy_point[i+1][0]-xy_point[i][0]
            y_delta = xy_point[i+1][1]-xy_point[i][1]
        elif i == point_size-1:
            x_delta = xy_point[i][0]-xy_point[i-1][0]
            y_delta = xy_point[i][1]-xy_point[i-1][1]
        else:
            x_delta = 0.5*(xy_point[i+1][0]-xy_point[i-1][0])
            y_delta = 0.5*(xy_point[i+1][1]-xy_point[i-1][1])

        dxs.append(x_delta)
        dys.append(y_delta)

    # Heading calculation
    for i in range(point_size):
        heading.append(math.atan2(dys[i], dxs[i]))

    # Get linear interpolated s for dkappa calculation
    distance = 0
    accumulated_s.append(distance)
    fx = xy_point[0][0]
    fy = xy_point[0][1]

    for i in range(1, point_size):
        nx = xy_point[i][0]
        ny = xy_point[i][1]

        end_segment_s = math.sqrt((fx-nx)*(fx-nx)+(fy-ny)*(fy-ny))
        accumulated_s.append(end_segment_s+distance)
        distance += end_segment_s

        fx = nx
        fy = ny

    # Get finite difference approximated first derivative of y and x respective to s for kappa calculation
    for i in range(point_size):
        if i == 0:
            divisor = 1e-12 if abs(accumulated_s[i+1]-accumulated_s[i]) < 1e-12 else accumulated_s[i+1]-accumulated_s[i]
            xds = (xy_point[i+1][0]-xy_point[i][0])/divisor
            yds = (xy_point[i+1][1]-xy_point[i][1])/divisor
        elif i == point_size-1:
            divisor = 1e-12 if abs(accumulated_s[i]-accumulated_s[i-1]) < 1e-12 else accumulated_s[i]-accumulated_s[i-1]
            xds = (xy_point[i][0]-xy_point[i-1][0])/divisor
            yds = (xy_point[i][1]-xy_point[i-1][1])/divisor
        else:
            divisor = 1e-12 if abs(accumulated_s[i+1]-accumulated_s[i-1]) < 1e-12 else accumulated_s[i+1]-accumulated_s[i-1]
            xds = (xy_point[i+1][0]-xy_point[i-1][0])/divisor
            yds = (xy_point[i+1][1]-xy_point[i-1][1])/divisor

        x_over_s_first_derivatives.append(xds)
        y_over_s_first_derivatives.append(yds)

    # Get finite difference approximated second derivative of y and x respective to s for kappa calculation
    for i in range(point_size):
        if i == 0:
            divisor = 1e-12 if abs(accumulated_s[i+1]-accumulated_s[i]) < 1e-12 else accumulated_s[i+1]-accumulated_s[i]
            xdds = (x_over_s_first_derivatives[i+1]-x_over_s_first_derivatives[i])/divisor
            ydds = (y_over_s_first_derivatives[i+1]-y_over_s_first_derivatives[i])/divisor
        elif i == point_size-1:
            divisor = 1e-12 if abs(accumulated_s[i]-accumulated_s[i-1]) < 1e-12 else accumulated_s[i]-accumulated_s[i-1]
            xdds = (x_over_s_first_derivatives[i]-x_over_s_first_derivatives[i-1])/divisor
            ydds = (y_over_s_first_derivatives[i]-y_over_s_first_derivatives[i-1])/divisor
        else:
            divisor = 1e-12 if abs(accumulated_s[i+1]-accumulated_s[i-1]) < 1e-12 else accumulated_s[i+1]-accumulated_s[i-1]
            xdds = (x_over_s_first_derivatives[i+1]-x_over_s_first_derivatives[i-1])/divisor
            ydds = (y_over_s_first_derivatives[i+1]-y_over_s_first_derivatives[i-1])/divisor

        x_over_s_second_derivatives.append(xdds)
        y_over_s_second_derivatives.append(ydds)

    for i in range(point_size):
        xds = x_over_s_first_derivatives[i]
        yds = y_over_s_first_derivatives[i]
        xdds = x_over_s_second_derivatives[i]
        ydds = y_over_s_second_derivatives[i]

        kappa = (xds*ydds-yds*xdds)/(math.sqrt(xds*xds+yds*yds)*(xds*xds+yds*yds)+1e-6)
        kappas.append(kappa)

    # Dkappa calculation
    for i in range(point_size):
        if i == 0:
            divisor = 1e-12 if abs(accumulated_s[i+1]-accumulated_s[i]) < 1e-12 else accumulated_s[i+1]-accumulated_s[i]
            dkappa = (kappas[i+1]-kappas[i])/divisor
        elif i == point_size-1:
            divisor = 1e-12 if abs(accumulated_s[i]-accumulated_s[i-1]) < 1e-12 else accumulated_s[i]-accumulated_s[i-1]
            dkappa = (kappas[i]-kappas[i-1])/divisor
        else:
            divisor = 1e-12 if abs(accumulated_s[i+1]-accumulated_s[i-1]) < 1e-12 else accumulated_s[i+1]-accumulated_s[i-1]
            dkappa = (kappas[i+1]-kappas[i-1])/divisor

        dkappas.append(dkappa)

    return heading, accumulated_s, kappas, dkappas