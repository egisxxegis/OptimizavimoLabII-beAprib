import sys

import numpy as np
import matplotlib.pyplot as plot
import pylab
from sympy import Symbol, lambdify, solve
from sympy.parsing.sympy_parser import parse_expr

from Algorithms import *


def prepare_contour(levels=100, show_labels=False):
    the_xs = []
    the_ys = []
    coord_minimum = -0.5
    coord_maximum = 1.3

    for val in np.arange(coord_minimum, coord_maximum*1.05 + 1e-100, 0.1):
        the_xs.append(val)
        the_ys.append(val)

    # Z values as a matrix
    the_values = np.ndarray((len(the_xs), len(the_ys)))

    # Populate values
    for the_x in range(0, len(the_xs)):
        for the_y in range(0, len(the_ys)):
            the_values[the_x][the_y] = fx(the_xs[the_x], the_ys[the_y])

    # axis limits
    pylab.xlim([coord_minimum, coord_maximum])
    pylab.ylim([coord_minimum, coord_maximum])

    plot.title('Contour plot')
    plot.xlabel('x')
    plot.ylabel('y')
    contours = plot.contour(the_xs, the_ys, the_values, levels=levels)

    # show values on contour
    if show_labels:
        plot.clabel(contours, inline=1, fontsize=10)
    plot.rcParams.update({'font.size': 12})


def summary_to_graph(s1, the_x_vector: [float, float], number_early_attempts=False, color="y-"):
    the_xs = []
    the_ys = []

    first_iteration = True
    i = -1
    for gamma, the_args, the_value in s1.gamma_x_value_history:
        i += 1
        if first_iteration:
            first_iteration = False

        the_xs.append(the_args[0])
        the_ys.append(the_args[1])

        if number_early_attempts:
            if 1 <= i+1 <= 4 and i > 0:
                plot.annotate(f'{i}', [the_xs[i], the_ys[i]])

    if number_early_attempts and [the_xs[1], the_ys[1]] != [the_x_vector[0], the_x_vector[1]]:
        plot.annotate(f'X', [the_x_vector[0], the_x_vector[1]])
    if len(the_xs) > 2 or number_early_attempts == False:
        plot.annotate("*", [the_xs[-1], the_ys[-1]])

    plot.plot(the_xs, the_ys, color, label=s1.name)


def summary_simplex_to_graph(s1, number_early_attempts=False, color="r-", limit_steps=11111):
    vertices = []
    x_high_i = 0

    starting = True
    i = -1
    for gamma, the_args, the_value in s1.gamma_x_value_history:
        i += 1

        if starting:
            vertices.append(the_args)
            if i == 2:
                starting = False
            else:
                continue
        else:  # first occur with i == 3
            x_high_i = s1.simplex_high_history_indexes[i-3]
            vertices[x_high_i] = the_args

        # listened to change -> form triangle -> draw
        # form triangle
        triangle_xs = []
        triangle_ys = []
        for vertice in vertices:
            triangle_xs.append(vertice[0])
            triangle_ys.append(vertice[1])

        # triangle touches its start, doesn't it?
        triangle_xs.append(triangle_xs[0])
        triangle_ys.append(triangle_ys[0])

        # number
        if number_early_attempts:
            if 1 <= i+1 <= 7:
                if i == 2:
                    for iii in range(3):
                        plot.annotate(f'{iii+1}', [triangle_xs[iii], triangle_ys[iii]])
                else:
                    plot.annotate(f'{i+1}', [triangle_xs[x_high_i], triangle_ys[x_high_i]])

        # draw
        plot.plot(triangle_xs, triangle_ys, color, label=s1.name)
        if limit_steps == i:
            break
    plot.annotate("*", s1.gamma_x_value_history[-1][1])


def gamma_to_graph(the_summary: ExecutionSummary, color="r-"):
    the_ys = []
    for the_gamma, the_x, the_value in the_summary.gamma_x_value_history:
        the_ys.append(the_gamma)
    the_xs = range(len(the_ys))
    the_suffix = '.\nMetodas: ' + the_summary.name
    plot.title('Naudota gama reikšmė žingsnių juostoje' + the_suffix)
    plot.xlabel('žingsnis')
    plot.ylabel('gama')
    pylab.xlim(the_xs[0], the_xs[-1])
    the_max_y = max(*the_ys)
    the_max_y = the_max_y*1.1 if the_max_y*1.1 > 31 else 31
    pylab.ylim(0, the_max_y)
    plot.plot(the_xs, the_ys, color, label=the_summary.name + '. gamma')


if __name__ == '__main__':

    LSP = '1813056'
    LSP_a = int(LSP[5])
    LSP_b = int(LSP[6])

    variables = {'x': Symbol('x', real=True),
                 'y': Symbol('y', real=True),
                 'z': Symbol('z', real=True)}
    form_fx = parse_expr('-(x * y * z / (2**3))', variables)
    form_fx_restriction = parse_expr('x + y + z - 1', variables)

    form_z = solve(form_fx_restriction, variables['z'])[0]
    form_fx = form_fx.subs(variables['z'], form_z)
    variables.pop('z')

    fx = lambdify(variables.values(), form_fx, "numpy")
    form_dfx = [form_fx.diff(the_symbol) for the_symbol in variables.values()]
    dfx = [lambdify([v for v in variables.values()],
                    the_dfx,
                    "numpy")
           for the_dfx in form_dfx]

    def gradient_fx(values):
        return np.array([de_fx(*values) for de_fx in dfx])

    def two_arguments_to_edges(the_x, the_y):
        the_z = form_z.subs([(variables['x'], the_x), (variables['y'], the_y)])
        the_mult = the_z * the_x * the_y
        if the_mult == 0:
            return 0, 0, 0
        the_length = (the_x * the_z / the_y / 2)**0.5
        the_width = (the_y * the_z / the_x / 2)**0.5
        the_height = (the_x * the_y / the_z / 2)**0.5
        return the_length, the_width, the_height

    print(f'LSP: {LSP}')
    print(f'a={LSP_a} ; b={LSP_b}')
    print('\n\n')

    near_zero = sys.float_info.min * sys.float_info.epsilon
    near_zero2 = 1e-16
    near_zero3 = 1e-6
    start_length = 0.09
    gamma_step = 3.8  # 3.8
    gradient_stop = near_zero2
    stop_line = 1e-8  # 1e-8

    x0 = [0, 0]
    x1 = [1, 1]
    xm = [LSP_a/10, LSP_b/10]
    xs = [x0, x1, xm]

    print("X, f(X), gradf(X)")
    for x in xs:
        print(f"{x}, "
              f"{fx(*x)}, "
              f"{gradient_fx(x)}")

    xs_summary = []
    for x in xs:
        xs_summary.append(
            [
                gradient_descend(fx, gradient_fx, x, stop_line, gamma_step=gamma_step),
                the_fastest_descend(fx, gradient_fx, x, stop_line),
                deformed_simplex(fx, x, start_length, stop_line, step_limit=189999)
            ]
        )

    x_experiments = [(xs[ii], xs_summary[ii]) for ii in range(len(xs))]

    for start_point, summaries in x_experiments:
        for summary in summaries:
            summary.translated = two_arguments_to_edges(*summary.solution)
            volume = 1
            for edge in summary.translated:
                volume *= edge
            summary.translated_fx = volume
        print(f"\n-------------Summary---------\n"
              f"Starting point = {start_point}")
        print_summary(*summaries)
        print(f'-----------------------------\n')

    # for ii in np.arange(0.4, 10.1, 0.2):
    #     gamma_step = ii
    #     for jj in range(3):
    #         x_experiments[jj][1][0] = gradient_descend(fx,
    #                                                    gradient_fx,
    #                                                    x_experiments[jj][0],
    #                                                    near_zero2,
    #                                                    gamma_step=gamma_step)
    #     print(f'\ngradientinis taškuose {x0}, {x1}, {xm}. gammma = {gamma_step}, epsilon = {gradient_stop}')
    #     print(f'solution, fx, steps')
    #     for start_point, summaries in x_experiments:
    #         print(f'{summaries[0].solution}, {summaries[0].value}, {summaries[0].steps}')

    # for ii in np.arange(0.01, 0.141, 0.01):
    #     alfa = ii
    #     for jj in range(3):
    #         x_experiments[jj][1][2] = deformed_simplex(fx,
    #                                                    x_experiments[jj][0],
    #                                                    alfa,
    #                                                    stop_line,
    #                                                    step_limit=1000)
    #     print(f'\nNelder-Mead taškuose {x0}, {x1}, {xm}. alpha = {alfa}, epsilon = {stop_line}')
    #     print(f'solution, fx, steps')
    #     for start_point, summaries in x_experiments:
    #         print(f'{summaries[2].solution}, {summaries[2].value}, {summaries[2].steps}')

    # # # # visualisation

    to_graph = True
    if to_graph:
        for start_point, summaries in x_experiments:
            colors = ['m-', 'g-', 'c-', 'r-', 'b-', 'y-']
            title = f'Eksperimento pradinis taškas X = ({start_point[0]}, {start_point[1]})'
            # gradient and steepest
            for ii in range(2):
                prepare_contour()
                summary_to_graph(summaries[ii], start_point, True, color=colors[ii])
                plot.title(title + f"\nmetodas: {summaries[ii].name}")
                plot.show()
                gamma_to_graph(summaries[ii])
                plot.show()
            prepare_contour()
            # simplex
            summary_simplex_to_graph(summaries[2], True, limit_steps=100, color=colors[3])
            plot.title(title + f"\nmetodas: {summaries[2].name}")
            plot.show()
            # all of them
            prepare_contour()
            summary_to_graph(summaries[0], start_point, False, color=colors[0])
            summary_to_graph(summaries[1], start_point, False, color=colors[1])
            summary_simplex_to_graph(summaries[2], False, limit_steps=100, color=colors[3])
            plot.title(title + f"\nVisi metodai")
            plot.show()
