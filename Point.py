class Point:
    def __init__(self, process_function, coord=0, record_calls=False):
        self.process_function = process_function
        self.coord = coord
        self.value = 0
        self.needsRefresh = True
        self.function_called_times = 0
        self.collect_function_call_history = record_calls
        if self.collect_function_call_history:
            self.history = {}

    @property
    def coord(self):
        return self._coord

    @coord.setter
    def coord(self, number):
        self._coord = number
        self.needsRefresh = True

    def set_value_and_mark(self, value):
        self.value = value
        self.needsRefresh = False

    def calculate(self):
        if self.needsRefresh:
            self.set_value_and_mark(self.process_function(self.coord))
            self.function_called_times += 1
            if self.collect_function_call_history:
                self.history[self.coord] = self.value

    def set_as(self, given_point):
        self.coord = given_point.coord
        self.value = given_point.value
        self.process_function = given_point.process_function
        self.needsRefresh = given_point.needsRefresh
        if given_point.collect_function_call_history:
            return
        if self.collect_function_call_history:
            self.history[self.coord] = self.value
