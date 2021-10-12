from Point import Point
from ExecutionSummary import ExecutionSummary
from tabulate import tabulate


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


def goldy_cutting(left_coord, right_coord, process_function, length_boundary):
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
    while length >= length_boundary:
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
