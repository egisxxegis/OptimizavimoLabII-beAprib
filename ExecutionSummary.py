class ExecutionSummary:
    def __init__(self, solution, value, steps, name='unnamed', interval_history=[]):
        self.name = name
        self.solution = solution
        self.value = value
        self.steps = steps
        self.function_called_times = 0
        self.dfx_times = 0
        self.ddfx_times = 0
        self.history = {}
        self.interval_history = interval_history
        self.done = True

    def collect_data_from_points(self, *args):
        calls = 0
        history = {}
        for point in args:
            calls += point.function_called_times
            if point.collect_function_call_history:
                history.update(point.history)
        self.function_called_times = calls
        self.history = history
        return self
