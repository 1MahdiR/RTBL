
from random import randint

from iot import IoTDevice, EdgeServer
from dnn import InferenceTask, DNNModel

MAX_TASK_SIZE = 10
MIN_TASK_SIZE = 5
MAX_TASK_CYCLE = 10
MIN_TASK_CYCLE = 5

class System:
    iot_device = None
    time_slots = None

    V = None

    E_bug = None

    R_h = list()
    H = list()

    def run_system_local():
        E_ = 0
        A_ = 0
        for i in range(System.time_slots):
            inference_task = InferenceTask(randint(MIN_TASK_SIZE, MAX_TASK_SIZE), randint(MIN_TASK_CYCLE, MAX_TASK_CYCLE))

            result = System.iot_device.run_inference_task(inference_task)

            A_ += result[0]
            E_ += result[1]
        
        acc = A_/System.time_slots
        print((acc, E_))
        return (acc, E_)
    
    def run_system_random():
        E_ = 0
        A_ = 0
        for i in range(System.time_slots):
            inference_task = InferenceTask(randint(MIN_TASK_SIZE, MAX_TASK_SIZE), randint(MIN_TASK_CYCLE, MAX_TASK_CYCLE))
            device = randint(0, len(System.iot_device.edge_servers))

            if device == 0:
                result = System.iot_device.run_inference_task(inference_task)
            else:
                result = System.iot_device.edge_servers[device-1].run_inference_task(inference_task)
            
            A_ += result[0]
            E_ += result[1]
        
        acc = A_/System.time_slots
        print((acc, E_))
        return (acc, E_)

            

if __name__ == "__main__":

    iot_dnn = DNNModel(60)
    edge_dnn_1 = DNNModel(90)
    edge_dnn_2 = DNNModel(85)
    edge_dnn_3 = DNNModel(88)

    es1 = EdgeServer(None, edge_dnn_1, 10)
    es2 = EdgeServer(None, edge_dnn_1, 5)
    es3 = EdgeServer(None, edge_dnn_1, 9)

    System.iot_device = IoTDevice([es1, es2, es3], iot_dnn, 3)

    es1.iot_device = System.iot_device
    es2.iot_device = System.iot_device
    es3.iot_device = System.iot_device

    System.time_slots = 10

    V = 500

    E_bug = 60

    System.run_system_random()
