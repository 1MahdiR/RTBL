
class InferenceTask:
    count = 0
    def __init__(self, size:float, cycles:int):
        self.size = size
        self.cycles = cycles
        self.id = InferenceTask.count
        InferenceTask.count += 1

class DNNModel:
    def __init__(self, accuracy:float):
        self.accuracy = accuracy
