import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import pandas as pd
from pylab import rcParams


size = np.arange(2, 18, 2)
run_times_default = genfromtxt('Data/run_times_default.txt')
run_times_pinn_on_cpu = genfromtxt('Data/run_times_pinn_cpu.txt')  # Fine on CPU, PINN on CPU
run_times_pinn_on_gpu = genfromtxt('Data/run_times_pinn_gpu.txt')  # Fine on CPU, PINN on GPU (increased by number of cores)
run_times_pinn_numba = genfromtxt('Data/run_times_pinn_numba.txt')  # Fine on GPU, PINN on GPU (increased by time interval)

run_times_serial = np.repeat(1.254250020980835, len(size))
run_times_serial_cpu = np.repeat(1.054250020980835, len(size))
run_times_serial_gpu = np.repeat(0.811250020980835, len(size))


fs = 8
rcParams['figure.figsize'] = 2.5, 2.5


fig = plt.figure()
plt.semilogy(size, run_times_default, 'k-o', label='Numeric', markersize=fs/2)
plt.semilogy(size, run_times_serial, 'k-', label='Serial-Numeric', markersize=fs/2)
plt.semilogy(size, run_times_pinn_on_gpu, 'g-o', label='PINN-GPU', markersize=fs/2)
plt.semilogy(size, run_times_serial_gpu, 'g-', label='Serial-PINN-GPU', markersize=fs/2)
plt.xlabel('cores', fontsize=fs, labelpad=0.5)
plt.ylabel('Runtimes [sec]', fontsize=fs, labelpad=0.5)
plt.legend(loc='upper right', fontsize=fs, prop={'size': fs-2})
plt.xticks(size, fontsize=fs)
plt.yticks(fontsize=fs)
bottom, top = plt.ylim()
plt.gcf().savefig("runtime_pinn_gpu.pdf", bbox_inches='tight')


fig = plt.figure()
plt.semilogy(size, run_times_default, 'k-o', label='Numeric', markersize=fs/2)
plt.semilogy(size, run_times_serial, 'k-', label='Serial-Numeric', markersize=fs/2)
plt.semilogy(size, run_times_pinn_on_cpu, 'b-o', label='PINN-CPU', markersize=fs/2)
plt.semilogy(size, run_times_serial_cpu, 'b-', label='Serial-PINN-CPU', markersize=fs/2)
plt.ylim(bottom, top)
plt.xlabel('cores', fontsize=fs, labelpad=0.5)
plt.ylabel('Runtimes [sec]', fontsize=fs, labelpad=0.5)
plt.legend(loc='upper right', fontsize=fs, prop={'size': fs-2})
plt.xticks(size, fontsize=fs)
plt.yticks(fontsize=fs)
plt.gcf().savefig("runtime_pinn_cpu.pdf", bbox_inches='tight')


def speed_up(n_iteration, cost_coarse, cost_fine):
    data = []
    for size in np.arange(2, 18, 2):
        speed_up = 1 / ((1 + n_iteration / size) * (cost_coarse / cost_fine) + (n_iteration / size))
        efficiency = speed_up / size
        data.append([size, speed_up, efficiency])
        df = pd.DataFrame(data, columns=['Size', 'Speed-Up', 'Efficiency'])
    return df