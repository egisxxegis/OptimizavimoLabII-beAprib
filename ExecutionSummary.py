class ExecutionSummary:
    def __init__(self, solution=0, value=0, steps=0, name='unnamed', interval_history=[]):
        self.name = name
        self.solution = 0
        self.value = 0
        self.steps = 0
        self.set_results(solution, value, steps)
        self.fx_times = 0
        self.dfx_times = 0
        self.ddfx_times = 0
        self.history = {}
        self.interval_history = interval_history
        self.gamma_x_value_history = []
        self.simplex_high_history_indexes = []
        self.done = True

    def set_results(self, solution, value, steps):
        self.solution = solution
        self.value = value
        self.steps = steps

    def collect_data_from_points(self, *args):
        calls = 0
        history = {}
        for point in args:
            calls += point.fx_times
            if point.collect_function_call_history:
                history.update(point.history)
        self.fx_times = calls
        self.history = history
        return self
