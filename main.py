
from random import randint, random

from iot import IoTDevice, EdgeServer
from dnn import InferenceTask, DNNModel

MAX_TASK_SIZE = 10000
MIN_TASK_SIZE = 5000
MAX_TASK_CYCLE = 0.2 * 10 ** 9
MIN_TASK_CYCLE = 0.15 * 10 ** 9

class System:
    iot_device = None
    time_slots = None

    V = None

    E_bug = None

    R_h = list()
    H = list()
    
    def run_system_random():
        E_ = 0
        A_ = 0
        for i in range(System.time_slots):
            inference_task = InferenceTask(randint(MIN_TASK_SIZE, MAX_TASK_SIZE), randint(MIN_TASK_CYCLE, MAX_TASK_CYCLE))

            num_of_devices = randint(1, 1 + len(System.iot_device.edge_servers))
            #num_of_devices = randint(1, 2)

            devices = []
            used_devices = []
            for i in range(num_of_devices):
                device = randint(0, len(System.iot_device.edge_servers))
                while device in devices:
                    device = randint(0, len(System.iot_device.edge_servers))
                
                used_devices.append(device)
                if device == 0:
                    devices.append(System.iot_device)
                else:
                    devices.append(System.iot_device.edge_servers[device-1])
                
            e_sum = 0
            a_max = 0
            for device in devices:
                if isinstance(device, IoTDevice):
                    if random() <= device.reliability:
                        result = device.run_inference_task(inference_task)
                    else:
                        print("Inference Task #%s Failed on <%s>" % (inference_task.id, device))
                        result = (0, 0)
                else:
                    if random() <= device.reliability:
                        result = device.run_inference_task(inference_task)
                    else:
                        print("Inference Task #%s Failed on <%s>" % (inference_task.id, device))
                        result = (0, 0)
                e_sum += result[1]
                if result[0] > a_max:
                    a_max = result[0]
            
            A_ += a_max
            E_ += e_sum
        
        acc = A_/System.time_slots
        print("Workload finished!")
        print("Average accuracy: %f" % acc)
        print("Total energy consumption: %f" % E_)
        return (acc, E_)
    
    def run_system_rtbl():
        E_ = 0
        A_ = 0
        for i in range(System.time_slots):
            inference_task = InferenceTask(randint(MIN_TASK_SIZE, MAX_TASK_SIZE), randint(MIN_TASK_CYCLE, MAX_TASK_CYCLE))

            num_of_devices = randint(1, 1 + len(System.iot_device.edge_servers))
            #num_of_devices = randint(1, 2)

            devices = []
            used_devices = []
            for i in range(num_of_devices):
                device = randint(0, len(System.iot_device.edge_servers))
                while device in devices:
                    device = randint(0, len(System.iot_device.edge_servers))
                
                used_devices.append(device)
                if device == 0:
                    devices.append(System.iot_device)
                else:
                    devices.append(System.iot_device.edge_servers[device-1])
                
            e_sum = 0
            a_max = 0
            for device in devices:
                if isinstance(device, IoTDevice):
                    if random() <= device.reliability:
                        result = device.run_inference_task(inference_task)
                    else:
                        print("Inference Task #%s Failed on <%s>" % (inference_task.id, device))
                        result = (0, 0)
                else:
                    if random() <= device.reliability:
                        result = device.run_inference_task(inference_task)
                    else:
                        print("Inference Task #%s Failed on <%s>" % (inference_task.id, device))
                        result = (0, 0)
                e_sum += result[1]
                if result[0] > a_max:
                    a_max = result[0]
            
            A_ += a_max
            E_ += e_sum
        
        acc = A_/System.time_slots
        print("Workload finished!")
        print("Average accuracy: %f" % acc)
        print("Total energy consumption: %f" % E_)
        return (acc, E_)

if __name__ == "__main__":

    iot_dnn = DNNModel(60)
    edge_dnn_1 = DNNModel(90)
    edge_dnn_2 = DNNModel(85)
    edge_dnn_3 = DNNModel(88)

    es1 = EdgeServer(None, edge_dnn_1, 10)
    es2 = EdgeServer(None, edge_dnn_2, 5)
    es3 = EdgeServer(None, edge_dnn_3, 9)

    System.iot_device = IoTDevice([es1, es2, es3], iot_dnn, 3)

    es1.iot_device = System.iot_device
    es2.iot_device = System.iot_device
    es3.iot_device = System.iot_device

    for i in range(10):
        if random() <= es1.reliability:
            es1.h_observations.append(1)
        else:
            es1.h_observations.append(0)

        if random() <= es2.reliability:
            es2.h_observations.append(1)
        else:
            es2.h_observations.append(0)

        if random() <= es3.reliability:
            es3.h_observations.append(1)
        else:
            es3.h_observations.append(0)

    print(es1.h_observations, es1.reliability)
    print(es2.h_observations, es2.reliability)
    print(es3.h_observations, es3.reliability)

    System.time_slots = 10

    V = 500

    E_bug = 60

    System.run_system_random()
