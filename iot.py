# IoT device

from random import uniform

from dnn import DNNModel, InferenceTask

class IoTDevice:
    def __init__(self, edge_servers:list, dnn_model:DNNModel, k:float):
        self.edge_servers = edge_servers
        self.dnn_model = dnn_model
        self.k = k
        self.reliability = uniform(0.8, 0.999)
        self.h_observations = []
        self.observations = []
        self.mean_r_j = 0
        self.emp_r_j = 0

    def run_inference_task(self, task:InferenceTask):
        energy = self.k * task.cycles
        print("Inference Task #%s Executed on <%s> (Accuracy: %d, Energy: %f)" % (task.id, self, self.dnn_model.accuracy, energy))
        return (self.dnn_model.accuracy, energy)

    def calculate_energy_consumption(self, task:InferenceTask):
        return self.k * task.cycles
    
    def __str__(self):
        return "IoT Device"
    
    def __repr__(self):
        return self.__str__()

class EdgeServer:
    count = 0
    def __init__(self, iot_device:IoTDevice, dnn_model:DNNModel, G_t:float):
        self.iot_device = iot_device
        self.dnn_model = dnn_model
        self.G_t = G_t
        self.reliability = uniform(0.6, 0.9)
        self.h_observations = []
        self.observations = []
        self.id = EdgeServer.count
        self.mean_r_j = 0
        self.emp_r_j = 0
        EdgeServer.count += 1

    def run_inference_task(self, task:InferenceTask):
        energy = self.G_t * task.size
        print("Inference Task #%s Executed on <%s> (Accuracy: %d, Energy: %f)" % (task.id, self, self.dnn_model.accuracy, energy))
        return (self.dnn_model.accuracy, self.G_t * task.size)
    
    def calculate_energy_consumption(self, task:InferenceTask):
        return self.G_t * task.size
    
    def __str__(self):
        return "Edge Device #%d" % self.id
    
    def __repr__(self):
        return self.__str__()