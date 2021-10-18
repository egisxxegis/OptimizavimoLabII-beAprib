import sys

import matplotlib.pyplot as plt
import numpy as np
from sympy import Symbol, lambdify, solve
from sympy.parsing.sympy_parser import parse_expr

from Algorithms import *


# def my_function(argument):
#     value = (argument ** 2 - 5)**2 / 6 - 1
#     return value
#
#
# def my_function_der1(argument):
#     value = 2*argument * (argument**2 - 5) / 3
#     return value
#
#
# def my_function_der2(argument):
#     value = (6*argument**2 - 10) / 3
#     return value

def graph_intervals(interval_array, function, name, line_count=6):
    increase = 0.1
    reset = True
    offset = 0
    colors = ['c-', 'g-', 'm-', 'r-', 'b-', 'y-']
    step_for_length = lambda x: x / 240
    suffix = ''
    for graph_index in range(0, len(interval_array)):

        start = interval_array[graph_index][0]
        end = interval_array[graph_index][1]
        length = end - start

        if graph_index - offset >= line_count:
            reset = True
            plt.title(f'{name}{suffix}')
            plt.grid()
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.legend()
            plt.show()
            if offset == 0:
                suffix = '; zoomed in'
            else:
                suffix = suffix + '+'
            offset = graph_index

        if reset:
            # plot basic function
            if offset == 0:
                # add tails
                start0 = start - length * increase
                end0 = end + length * increase
            else:
                # continue from last shown interval
                start0 = interval_array[graph_index-1][0]
                end0 = interval_array[graph_index-1][1]
            x_arr = np.arange(start0, end0, step_for_length(length))
            y_arr = function(x_arr)
            plt.plot(x_arr, y_arr, 'k-', label='f(x)')
            reset = False

        # plot interval
        x_arr = np.arange(start, end, step_for_length(end - start))
        plt.plot(x_arr,
                 function(x_arr),
                 colors[graph_index - offset],
                 label=f'interval at N={graph_index + 1}',
                 linewidth=((graph_index - offset+1)*2))

    # flush if needed
    if graph_index % line_count != 0:
        plt.title(f'{name}{suffix}')
        plt.grid()
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.show()
    plt.clf()


if __name__ == '__main__':

    LSP = '9999956'
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

    print(f'LSP: {LSP}')
    print(f'a={LSP_a} ; b={LSP_b}')
    print('\n\n')

    print_summary()

    near_zero = sys.float_info.min * sys.float_info.epsilon
    near_zero2 = 1e-16

    x0 = [0, 0]
    x1 = [1, 1]
    xm = [LSP_a/10, LSP_b/10]

    print(f'\ngradientinis taškuose {x0}, {x1}, {xm}')
    print(gradient_descend(fx, gradient_fx, x0, near_zero2, 1))
    print(gradient_descend(fx, gradient_fx, x1, near_zero2, 1))
    print(gradient_descend(fx, gradient_fx, xm, near_zero2, 1))

    print(f'\nGreičiausiasis taškuose {x0}, {x1}, {xm}')
    print(the_fastest_descend(fx, gradient_fx, x0, near_zero2))
    print(the_fastest_descend(fx, gradient_fx, x1, near_zero2))
    print(the_fastest_descend(fx, gradient_fx, xm, near_zero2))

    print(f'\nDeformuojamas taškuose {x0}, {x1}, {xm}')
    print(deformed_simplex(fx, x0, 0.25, near_zero2))
    print(deformed_simplex(fx, x1, 0.25, near_zero2))
    print(deformed_simplex(fx, xm, 0.25, near_zero2*1e13))

    # # # # visualisation
    #
    # # # plot basic function
    #
    # # 1st picture. [-10; 10] and our task at [0; 10]
    # xs = np.arange(-10., 10., 0.05)
    # plt.plot(xs, fx(xs), 'r-', label='f(x)')
    # xs = np.arange(0., 10., 0.05)
    # plt.plot(xs, fx(xs), 'b-', label='f(x) intervale [0; 10]', linewidth=3)
    # plt.title(
    #     fr'f(x) = ${str(form_fx).replace("**", "^")}$ ' + '\ngrafiškai išskiriant mums svarbią sritį: x ∈ [0; 10]')
    # plt.grid()
    # plt.xlabel('x')
    # plt.ylabel('f(x)')
    # plt.legend()
    # plt.show()
    #
    # # 2nd picture. [-5; 5] and [0; 5]
    # xs = np.arange(-5., 5., 0.05)
    # plt.plot(xs, fx(xs), 'r-', label='f(x)')
    # xs = np.arange(0., 5., 0.05)
    # plt.plot(xs, fx(xs), 'b-', label='f(x) intervale [0; 5]')
    # plt.title(fr'f(x) = ${str(form_fx).replace("**", "^")}$ ' + '\ngrafikas, kai x ∈ [-5; 5]')
    # plt.grid()
    # plt.xlabel('x')
    # plt.ylabel('f(x)')
    # plt.legend()
    # plt.show()
    #
    # # 3rd picture. [0; 10]
    # xs = np.arange(0., 10., 0.01)
    # plt.plot(xs, fx(xs), 'b-', label='f(x)')
    # plt.title(fr'f(x) = ${str(form_fx).replace("**", "^")}$ ' + '\ngrafikas, kai x ∈ [0; 10]')
    # plt.grid()
    # plt.xlabel('x')
    # plt.ylabel('f(x)')
    # plt.legend()
    # plt.show()
    #
    # # # plot interval methods
    # graph_intervals(summary_divide.interval_history, fx, summary_divide.name)
    # graph_intervals(summary_goldy.interval_history, fx, summary_goldy.name)
    #
    # # # plot scatter type methods
    #
    # #  plot basic function
    # xs = np.arange(-0., 6., 0.001)
    # plt.plot(xs, fx(xs), 'r-', label='f(x)')
    #
    # # plot summary newton
    # xs = []
    # ys = []
    # counter = 0
    # for k, v in summary_newton.history.items():
    #     xs.append(k)
    #     ys.append(fx(k))
    #     plt.annotate(f'#{counter}', [k, fx(k)])
    #     counter += 1
    # plt.scatter(xs, ys, c='m', label='newton')
    # plt.annotate(f'Solution at x = {xs[-1]: <#01.5f}',
    #              xy=(xs[-1], ys[-1]),
    #              xytext=(xs[-1], ys[0]),
    #              arrowprops=dict(
    #                  facecolor='black',
    #                  shrink=0.05))
    # plt.title(f'{summary_newton.name} in {summary_newton.steps} steps\n(x0 (at #0) is not counted as a step)')
    # plt.grid()
    # plt.xlabel('x')
    # plt.ylabel('f(x)')
    # plt.legend()
    # plt.show()
