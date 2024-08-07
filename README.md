# Reliability-Aware Task Scheduling with Bandit Learning (RTBL)
This is an implementation of the proposed method in the paper ["Reliability-Aware Online Scheduling for DNN Inference Tasks in Mobile-Edge Computing"](https://doi.org/10.1109/JIOT.2023.3243266).

> **Abstract:**
> Mobile-edge computing (MEC) is widely envisioned as a promising technique for provisioning artificial intelligence (AI) capability for resource-limited Internet of Things (IoT) devices by leveraging edge servers (ESs) for executing deep neural network (DNN) inference tasks in proximity. However, scheduling DNN inference tasks at the network edge under unknown system dynamics (e.g., uncertain availability of ESs) may suffer from failures, making it difficult to guarantee reliable services for the IoT device. To overcome this challenge, we propose a reliability-aware online scheduling scheme for DNN inference tasks in MEC by leveraging both online feedback and offline data to learn the uncertain availability of ESs to maximize both the inference accuracy and service reliability of DNN inference tasks (i.e., the number of DNN inference tasks processed during the system span). We first formulate the reliability-aware DNN inference tasks scheduling problem as a novel constrained combinatorial multiarmed bandit (CMAB) problem. Then by integrating the Lyapunov optimization technique, bandit learning, approximated submodular maximization, and historical data organically, we design a reliability-aware task scheduling scheme with a bandit learning (RTBL) algorithm to solve this problem. Unfortunately, even with an accurate prediction of the system uncertainties, the task scheduling problem is still NP-hard. To deal with it, we, therefore, design an advanced approximation algorithm based on the submodularity of the scheduling problem which obtains a near-optimal solution and provides a satisfactory performance guarantee. Finally, we conduct rigorous theoretical analysis and race-driven simulations to show RTBL’s brilliant performance.

## Running the simulation
Simply clone the repository and execute the 'main.py' file:

``` $ python3 main.py ```

The script will generate the simulation environment based on the configuration and it will run the simulation and finally gives a brief stats about the simulation results.

## Configuration
You can change the configuration of the simulation in the 'config.py' script.

This configuration file includes these parameters:
- number of edge servers
- number of time slots
- number of historical observations
- the V constant
- energy budget
- maximum and minimum task size
- maximum and minimum cycles required for a task
- maximum and minimum reliability of edge or IoT device
- maximum and minimum accuracy of dnn models in edge or IoT devices
- maximum and minimum constant of energy consumption rate

## Last but not least...
I hope you'll find this project useful.

Any comments or contribution to this project would be appreciated.

---

Happy Hacking... :)