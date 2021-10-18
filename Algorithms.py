import copy

from Point import Point
from ExecutionSummary import ExecutionSummary
from tabulate import tabulate
import numpy as np


def print_summary(*args):
    headers = ["Name", "Found solution", "Solution", "Value", "Steps (N)", "f calls", "df calls", "ddf calls"]
    if len(args) < 1:
        print(tabulate([], headers=headers))
        return
    # Use a breakpoint in the code line below to debug your script.
    data = []
    print('Name, Solution, f(x)')
    for argument in args:
        data.append([argument.name,
                     argument.done,
                     argument.solution,
                     argument.value,
                     argument.steps,
                     argument.function_called_times,
                     argument.dfx_times,
                     argument.ddfx_times])
        print(f'{argument.name}, {argument.solution}, {argument.value}')
    print('\n\n')
    print(tabulate(data, headers=headers))


def dividing_into_halves(left_coord, right_coord, process_function, length_boundary):
    left = Point(process_function, left_coord, record_calls=False)
    right = Point(process_function, right_coord, record_calls=False)

    # check user error
    if left.coord > right.coord:
        temp = left.coord
        left.coord = right.coord
        right.coord = temp

    center = Point(process_function, record_calls=True)
    x_left = Point(process_function, record_calls=False)
    x_right = Point(process_function, record_calls=False)

    center.coord = (left.coord + right.coord) / 2
    center.calculate()
    length = right.coord - left.coord
    step_count = 0

    interval_history = []
    while length >= length_boundary:
        x_left.coord = left.coord + length/4
        x_right.coord = right.coord - length/4
        x_left.calculate()
        x_right.calculate()
        if x_left.value < center.value:
            right.set_as(center)
            center.set_as(x_left)
        elif x_right.value < center.value:
            left.set_as(center)
            center.set_as(x_right)
        else:
            left.set_as(x_left)
            right.set_as(x_right)
        length = right.coord - left.coord
        interval_history.append([left.coord, right.coord])
        step_count += 1
        # jump back to while

    return ExecutionSummary(center.coord, center.value, step_count, 'Intervalo pusiau dalijimas', interval_history).\
        collect_data_from_points(left, right, center, x_left, x_right)


def goldy_cutting(left_coord, right_coord, process_function, length_boundary, step_limit=9999999999):
    left = Point(process_function, left_coord, record_calls=False)
    right = Point(process_function, right_coord, record_calls=False)
    constant = (-1 + 5**0.5)/2

    # check user error
    if left.coord > right.coord:
        temp = left.coord
        left.coord = right.coord
        right.coord = temp

    length = right.coord - left.coord
    x_left = Point(process_function, right.coord - constant * length, record_calls=True)
    x_right = Point(process_function, left.coord + constant * length, record_calls=False)

    interval_history = []
    step_count = 0
    while length >= length_boundary and step_count < step_limit:
        x_left.calculate()
        x_right.calculate()

        if x_right.value < x_left.value:
            left.set_as(x_left)
            length = right.coord - left.coord
            x_left.set_as(x_right)
            x_right.coord = left.coord + constant * length
        else:
            right.set_as(x_right)
            length = right.coord - left.coord
            x_right.set_as(x_left)
            x_left.coord = right.coord - constant * length
        step_count += 1
        interval_history.append([left.coord, right.coord])
        # while

    return ExecutionSummary(x_left.coord, x_left.value, step_count, 'Auksinis pjūvis', interval_history).\
        collect_data_from_points(left, right, x_left, x_right)


def newton(process_function, process_function_derivative, process_function_derivative2, x0, step_length_boundary):
    xi = Point(process_function_derivative, x0, record_calls=True)
    xi1 = Point(process_function_derivative, xi.coord, record_calls=True)
    steps = 0
    calls_derivative2 = 0
    xi.calculate()
    length = abs(xi.value)
    while length > step_length_boundary:
        xi1.coord = xi.coord - (xi.value / process_function_derivative2(xi.coord))
        calls_derivative2 += 1
        steps += 1
        xi1.calculate()
        xi.set_as(xi1)
        length = abs(xi.value)

    done = True if process_function_derivative2(xi.coord) > 0 else False

    summary = ExecutionSummary(xi.coord,
                               process_function(xi.coord),
                               steps,
                               name="Newton").collect_data_from_points(xi, xi1)
    summary.dfx_times = summary.function_called_times
    summary.function_called_times = 1
    summary.ddfx_times = calls_derivative2
    summary.done = done
    return summary


def gradient_descend(process_function, process_function_gradient, x0, gradient_norm_epsilon, gamma_step):
    x = copy.deepcopy(x0)
    if isinstance(x, list):
        x = np.array(x)
    if not isinstance(x, np.ndarray):
        raise TypeError("Duotas x0 nebuvo list arba numpy.ndarray tipo")
    steps = 0
    while True:
        steps += 1
        si = process_function_gradient(x)  # n df calls
        x_next = x - gamma_step * si
        x = x_next.copy()
        if np.linalg.norm(si, ord=None) < gradient_norm_epsilon:
            # finished
            break
        continue
    return x, process_function(*x), steps


def the_fastest_descend(process_function, process_function_gradient, x0, gradient_norm_epsilon):
    x = copy.deepcopy(x0)
    if isinstance(x, list):
        x = np.array(x)
    if not isinstance(x, np.ndarray):
        raise TypeError("Duotas x0 nebuvo list arba numpy.ndarray tipo")
    steps = 0

    while True:
        steps += 1
        si = process_function_gradient(x)

        def func_with_step(gamma):
            return process_function(*(x - gamma * si))

        # with right_boundary = 10 or = 2 we would jump too far
        goldy_summary = goldy_cutting(0, steps, func_with_step, length_boundary=1e-5, step_limit=111)
        gamma_step = goldy_summary.solution

        x = x - gamma_step * si

        if np.linalg.norm(si, ord=None) < gradient_norm_epsilon:
            break
        continue
    return x, process_function(*x), steps


def deformed_simplex(process_function, x0, start_length, stop_length,
                     extend_coef=2, normal_contract_coef=0.5, big_contract_coef=-0.5, step_limit=11111):
    # form a simplex figure
    x = copy.deepcopy(x0)
    if isinstance(x, list):
        x = np.array(x)
    if not isinstance(x, np.ndarray):
        raise TypeError("Duotas x0 nebuvo list arba numpy.ndarray tipo")

    n = len(x0)  # number of axis (x, y) = 2
    vertices_c = n + 1  # number of vertices: triangle for 2 axis
    x_high_i = 0
    x_g_i = 0
    x_low_i = 0

    vertices = [np.zeros(n) for i in range(vertices_c)]
    values = [i for i in range(vertices_c)]
    needs_recalculate = np.zeros(vertices_c) == 0
    used_vertices = {}

    def _get_xcenter(regarding_vertice_i):
        the_center = np.zeros(n)
        for i in range(vertices_c):
            if i == regarding_vertice_i:
                continue
            the_center += vertices[i]
        return the_center / n

    # locate vertices
    delta1 = ((n + 1) ** 0.5 + n - 1) / (n * 2 ** 0.5) * start_length
    delta2 = ((n + 1) ** 0.5 - 1) / (n * 2 ** 0.5) * start_length
    for vertice_i in range(n):
        for axis_i in range(n):
            delta = delta1 if vertice_i != axis_i else delta2
            vertices[vertice_i][axis_i] = x[axis_i] + delta
    vertices[vertices_c - 1] = x  # x0 is also a vertice

    def _calculate_values():
        for i in range(vertices_c):
            if needs_recalculate[i]:
                values[i] = process_function(*vertices[i])
                needs_recalculate[i] = False
            identifier = 'z'.join([f'{v}' for v in vertices[i]])
            used_vertices[identifier] = True

    def _get_high_low_g():
        # at first the_max/g/min will hold values
        # then we will switch their values to indexes

        sorted_values = values.copy()
        sorted_values.sort()
        the_min = sorted_values[0]
        the_max = sorted_values[vertices_c-1]
        the_2nd_high = sorted_values[vertices_c-2]

        unfilled = [True, True, True]
        for i in range(vertices_c):
            if unfilled[0] and values[i] == the_max:
                the_max = i
                unfilled[0] = False
            elif unfilled[1] and values[i] == the_2nd_high:
                the_2nd_high = i
                unfilled[1] = False
            elif unfilled[2] and values[i] == the_min:
                the_min = i
                unfilled[2] = False

        if the_max == the_2nd_high or the_min == the_max:

            raise NotImplementedError(f"xh, xg, xl had a collision.\n"
                                      f"all_vertices =\n"
                                      f"{[(vertices[i], values[i]) for i in range(len(vertices))]}\n"
                                      f"the_max = {the_max}\n"
                                      f"the_2nd_high = {the_2nd_high}\n"
                                      f"the_min = {the_min}")
        return the_max, the_2nd_high, the_min

    def _get_x_new(the_x_center):
        x_high = vertices[x_high_i]
        x_direction_to_center = the_x_center - x_high
        x_temp = x_high + x_direction_to_center * 2
        x_temp_value = process_function(*x_temp)

        # extend, reduce, negative reduce
        multiplier = 1
        if values[x_low_i] < x_temp_value < values[x_g_i]:
            multiplier = 1
        elif x_temp_value < values[x_low_i]:
            multiplier = extend_coef
        elif x_temp_value > values[x_high_i]:
            multiplier = big_contract_coef
        elif values[x_g_i] < x_temp_value < values[x_high_i]:
            multiplier = normal_contract_coef
        multiplier += 1

        x_new = x_high + multiplier * x_direction_to_center
        value = x_temp_value if multiplier == 2 else process_function(*x_new)

        # if multiplier == 1, calls = 1 ; else calls = 2
        return x_new, value

    # let us iterate
    _calculate_values()
    x_high_i, x_g_i, x_low_i = _get_high_low_g()
    steps = 0
    while True:
        steps += 1
        x_center = _get_xcenter(x_high_i)
        vertices[x_high_i], values[x_high_i] = _get_x_new(x_center)
        x_high_i, x_g_i, x_low_i = _get_high_low_g()
        if np.linalg.norm(vertices[x_high_i] - vertices[x_low_i], ord=None) < stop_length \
                or steps >= step_limit:
            return vertices[x_low_i], values[x_low_i], steps
