
from random import randint, random, shuffle
from math import log10

from iot import IoTDevice, EdgeServer
from dnn import InferenceTask, DNNModel

MAX_TASK_SIZE = 100
MIN_TASK_SIZE = 50
MAX_TASK_CYCLE = 0.2 * 10 ** 6
MIN_TASK_CYCLE = 0.15 * 10 ** 6

class System:
    iot_device = None
    time_slots = None

    V = 1000000000

    E_bug = 100000

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
            _devices = list(range(1 + len(System.iot_device.edge_servers)))
            shuffle(_devices)
            for j in range(num_of_devices):
                device = _devices.pop()

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
                        device.observations.append(1)
                    else:
                        print("Inference Task #%s Failed on <%s>" % (inference_task.id, device))
                        device.observations.append(0)
                        result = (0, 0)
                else:
                    if random() <= device.reliability:
                        result = device.run_inference_task(inference_task)
                        device.observations.append(1)
                    else:
                        print("Inference Task #%s Failed on <%s>" % (inference_task.id, device))
                        device.observations.append(0)
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
        V_Q = [1]

        E_ = 0
        A_ = 0
        System.iot_device.mean_r_j = sum(System.iot_device.h_observations) / len(System.iot_device.h_observations)
        #print(System.iot_device.mean_r_j)
        for device in System.iot_device.edge_servers:
            device.mean_r_j = sum(device.h_observations) / len(device.h_observations)
            #print(device.mean_r_j)

        for i in range(System.time_slots):

            for device in [System.iot_device] + System.iot_device.edge_servers:
                r_1 = device.mean_r_j - ((2 * log10(i+1))/(len(device.h_observations)+len([ x for x in device.observations if x == 1])))
                r_2 = 0
                device.emp_r_j = max(r_1, r_2)
                #print(device.emp_r_j)

            inference_task = InferenceTask(randint(MIN_TASK_SIZE, MAX_TASK_SIZE), randint(MIN_TASK_CYCLE, MAX_TASK_CYCLE))

            devices = []
            for j in range(1 + len(System.iot_device.edge_servers)):
                devices_temp = devices.copy()
                arg_max = 0
                device_max = None
                for device in [System.iot_device] + System.iot_device.edge_servers:
                    if device not in devices_temp:
                        devices_temp_temp = devices_temp.copy()
                        devices_temp_temp.append(device)

                        max_acc_temp = 0
                        max_acc_temp_temp = 0

                        prod_rel_temp = 1
                        prod_rel_temp_temp = 1

                        for k in devices_temp:
                            acc = k.dnn_model.accuracy * k.emp_r_j
                            if max_acc_temp < acc:
                                max_acc_temp = acc
                            
                            prod_rel_temp *= 1 - k.emp_r_j

                        for k in devices_temp_temp:
                            acc = k.dnn_model.accuracy * k.emp_r_j
                            if max_acc_temp_temp < acc:
                                max_acc_temp_temp = acc
                            
                            prod_rel_temp_temp *= 1 - k.emp_r_j

                        e_sum_temp = 0
                        for k in devices_temp_temp:
                            e_sum_temp += V_Q[-1] * k.calculate_energy_consumption(inference_task)

                        f1_1 = System.V * (max_acc_temp_temp + 1 - prod_rel_temp_temp)
                        f1_2 = System.V * (max_acc_temp + 1 - prod_rel_temp)
                        f2 = e_sum_temp

                        arg = (f1_1 - f1_2) / f2
                        if arg > arg_max:
                            arg_max = arg
                            device_max = device

                if device_max:
                    devices_temp_temp = devices_temp + [device_max]
                max_acc_temp = 0
                max_acc_temp_temp = 0

                prod_rel_temp = 1
                prod_rel_temp_temp = 1

                for k in devices_temp:
                    acc = k.dnn_model.accuracy * k.emp_r_j
                    if max_acc_temp < acc:
                        max_acc_temp = acc
                    
                    prod_rel_temp *= 1 - k.emp_r_j

                for k in devices_temp_temp:
                    acc = k.dnn_model.accuracy * k.emp_r_j
                    if max_acc_temp_temp < acc:
                        max_acc_temp_temp = acc
                    
                    prod_rel_temp_temp *= 1 - k.emp_r_j

                    e_sum_temp += V_Q[-1] * k.calculate_energy_consumption(inference_task)


                
                f1_1 = System.V * (max_acc_temp_temp + 1 - prod_rel_temp_temp)
                f1_2 = System.V * (max_acc_temp + 1 - prod_rel_temp)
                f2 = e_sum_temp
                #print(device_max)

                if f1_1 - f1_2 - f2 >= 0:
                    devices.append(device_max)
                else:
                    #print(f1_1 - f1_2 - f2)
                    break
                
            e_sum = 0
            a_max = 0
            failed_devices = []
            for device in devices:
                if isinstance(device, IoTDevice):
                    if random() <= device.reliability:
                        result = device.run_inference_task(inference_task)
                        device.observations.append(1)
                    else:
                        print("Inference Task #%s Failed on <%s>" % (inference_task.id, device))
                        device.observations.append(0)
                        failed_devices.append(device)
                        result = (0, 0)
                else:
                    if random() <= device.reliability:
                        result = device.run_inference_task(inference_task)
                        device.observations.append(1)
                    else:
                        print("Inference Task #%s Failed on <%s>" % (inference_task.id, device))
                        device.observations.append(0)
                        failed_devices.append(device)
                        result = (0, 0)
                e_sum += result[1]
                if result[0] > a_max:
                    a_max = result[0]
            
            for device in devices:
                expr_1 = ((len([x for x in device.observations if x == 1]) + len(device.h_observations)) / (len([x for x in device.observations if x == 1]) + 1 + len(device.h_observations))) * device.mean_r_j
                
                if device not in failed_devices:
                    expr_2 = 1 / (len([x for x in device.observations if x == 1]) + 1 + len(device.h_observations))
                else:
                    expr_2 = 0
                
                device.mean_r_j = expr_1 + expr_2
                

            V_Q.append(max(V_Q[-1] - System.E_bug + e_sum, 1))
            
            A_ += a_max
            E_ += e_sum
        
        acc = A_/System.time_slots
        print("Workload finished!")
        print("Average accuracy: %f" % acc)
        print("Total energy consumption: %f" % E_)
        print("Virtual Queue:", V_Q)
        return (acc, E_)

if __name__ == "__main__":

    iot_dnn = DNNModel(60)
    edge_dnn_1 = DNNModel(90)
    edge_dnn_2 = DNNModel(85)
    edge_dnn_3 = DNNModel(88)

    es1 = EdgeServer(None, edge_dnn_1, 10)
    es2 = EdgeServer(None, edge_dnn_2, 5)
    es3 = EdgeServer(None, edge_dnn_3, 9)

    System.iot_device = IoTDevice([es1, es2, es3], iot_dnn, 1)

    es1.iot_device = System.iot_device
    es2.iot_device = System.iot_device
    es3.iot_device = System.iot_device

    for i in range(10):
        if random() <= System.iot_device.reliability:
            System.iot_device.h_observations.append(1)
        else:
            System.iot_device.h_observations.append(0)

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

    print(System.iot_device.h_observations, System.iot_device.reliability)
    print(es1.h_observations, es1.reliability)
    print(es2.h_observations, es2.reliability)
    print(es3.h_observations, es3.reliability)

    System.time_slots = 20

    System.run_system_rtbl()

    print(System.iot_device.observations, System.iot_device.reliability)
    print(es1.observations, es1.reliability)
    print(es2.observations, es2.reliability)
    print(es3.observations, es3.reliability)