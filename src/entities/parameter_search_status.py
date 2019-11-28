

class ParameterSearchStatus:

    UNDER_EXECUTION = 0
    CONTINUE_PREVIOUS_FAIL = 100
    DONE = 999

    def __init__(self):
        self.current_parameter = ''
        self.current_kfold = 0
        self.current_test = 0
        self.continue_parameter = ''
        self.continue_kfold = 0
        self.status = self.UNDER_EXECUTION

    def set_current_parameter(self, parameter_combination):
        self.current_parameter = parameter_combination

    def get_current_parameter(self):
        return self.current_parameter

    def set_current_kfold(self, kfold):
        self.current_kfold = kfold

    def get_current_kfold(self):
        return self.current_kfold

    def set_current_test(self, test):
        self.current_test = test

    def get_current_test(self):
        return self.current_test

    def set_continue_parameter(self, parameter_combination):
        self.continue_parameter = parameter_combination

    def get_continue_parameter(self):
        return self.continue_parameter

    def set_continue_kfold(self, kfold):
        self.continue_kfold = kfold

    def get_continue_kfold(self):
        return self.continue_kfold

    def set_status(self, new_status):
        if new_status == self.UNDER_EXECUTION:
            self.status = self.UNDER_EXECUTION
        elif new_status == self.CONTINUE_PREVIOUS_FAIL:
            self.status = self.CONTINUE_PREVIOUS_FAIL
        elif new_status == self.DONE:
            self.status = self.DONE

    def get_status(self):
        return self.status