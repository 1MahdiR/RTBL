
from random import randint, random, shuffle, uniform
from math import log10

from iot import IoTDevice, EdgeServer
from dnn import InferenceTask, DNNModel
from bcolor import *
from config import *

class System:
    iot_device = None
    time_slots = NUMBER_OF_TIME_SLOTS

    V = V_CONSTANT

    E_bug = ENERGY_BUDGET

    R_h = list()
    H = list()
    
    def run_system_random():
        E_ = 0
        A_ = 0
        for i in range(System.time_slots):
            inference_task = InferenceTask(randint(MIN_TASK_SIZE, MAX_TASK_SIZE), randint(MIN_TASK_CYCLE, MAX_TASK_CYCLE))

            num_of_devices = randint(1, 1 + len(System.iot_device.edge_servers))

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
                        print((FAIL + "Inference Task #%s Failed on <%s>" + ENDC) % (inference_task.id, device))
                        device.observations.append(0)
                        result = (0, 0)
                else:
                    if random() <= device.reliability:
                        result = device.run_inference_task(inference_task)
                        device.observations.append(1)
                    else:
                        print((FAIL + "Inference Task #%s Failed on <%s>" + ENDC) % (inference_task.id, device))
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
        for device in System.iot_device.edge_servers:
            device.mean_r_j = sum(device.h_observations) / len(device.h_observations)

        for i in range(System.time_slots):

            for device in [System.iot_device] + System.iot_device.edge_servers:
                r_1 = device.mean_r_j - ((2 * log10(i+1))/(len(device.h_observations)+len([ x for x in device.observations if x == 1])))
                r_2 = 0
                device.emp_r_j = max(r_1, r_2)

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

                if f1_1 - f1_2 - f2 >= 0:
                    devices.append(device_max)
                else:
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
                        print((FAIL + "Inference Task #%s Failed on <%s>" + ENDC) % (inference_task.id, device))
                        device.observations.append(0)
                        failed_devices.append(device)
                        result = (0, 0)
                else:
                    if random() <= device.reliability:
                        result = device.run_inference_task(inference_task)
                        device.observations.append(1)
                    else:
                        print((FAIL + "Inference Task #%s Failed on <%s>" + ENDC) % (inference_task.id, device))
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
        print(OKGREEN + BOLD + "Workload finished!" + ENDC)
        print("\n---\n")
        print(OKGREEN + BOLD + "Stats:" + ENDC)
        print("Average accuracy: %f" % acc)
        print("Total energy consumption: %f" % E_)
        print("Virtual queue:", "[ " + ", ".join([ OKGREEN + str(x) + ENDC if x == 1 else FAIL + str(x) + ENDC for x in V_Q ]) + " ]")
        print()
        return (acc, E_)

if __name__ == "__main__":

    iot_dnn = DNNModel(randint(MIN_DNN_ACCURACY_IOT, MAX_DNN_ACCURACY_IOT))
    ls_edge_dnn = []
    for i in range(NUMBER_OF_EDGE_SERVERS):
        ls_edge_dnn.append(DNNModel(randint(MIN_DNN_ACCURACY_EDGE, MAX_DNN_ACCURACY_EDGE)))

    ls_edge = []
    for i in range(NUMBER_OF_EDGE_SERVERS):
        ls_edge.append(EdgeServer(None, ls_edge_dnn[i], uniform(MIN_EDGE_G, MAX_EDGE_G)))
    
    System.iot_device = IoTDevice(ls_edge, iot_dnn, uniform(MIN_IOT_K, MAX_IOT_K))

    for es in ls_edge:
        es.iot_device = System.iot_device

    for i in range(NUMBER_OF_HISTORICAL_OBSERVATIONS):
        if random() <= System.iot_device.reliability:
            System.iot_device.h_observations.append(1)
        else:
            System.iot_device.h_observations.append(0)

        for es in ls_edge:
            if random() <= es.reliability:
                es.h_observations.append(1)
            else:
                es.h_observations.append(0)

    print(OKBLUE + BOLD + "Devices stats:" + ENDC)
    print(System.iot_device, "(Accuracay: %d, Energy constant: %.2f)" % (System.iot_device.dnn_model.accuracy, System.iot_device.k))
    for es in ls_edge:
        print(es, "(Accuracay: %d, Energy constant: %.2f)" % (es.dnn_model.accuracy, es.G_t))
    print("\n Enter to begin simulation...\r", end="")
    input()
    print("\n---\n")

    print(OKBLUE + BOLD + "Historical observations:" + ENDC)
    print(OKBLUE + str(System.iot_device) + ":" + ENDC, System.iot_device.h_observations)
    for es in ls_edge:
        print(OKBLUE + str(es) + ":" + ENDC, es.h_observations)
    print("\n---\n")

    print(WARNING + BOLD + "Simulation starts:" + ENDC)
    System.run_system_rtbl()

    print(OKBLUE + BOLD + "Observations:" + ENDC)
    print(OKBLUE + str(System.iot_device) + " (Real reliability:{:.4f}):".format(System.iot_device.reliability) + ENDC, System.iot_device.observations)
    for es in ls_edge:
        print(OKBLUE + str(es) + " (Real reliability:{:.4f}):".format(es.reliability) + ENDC, es.observations)

    print(WARNING + BOLD + "Simulation ends!" + ENDC)