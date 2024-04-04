import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np

fs = 8
rcParams['figure.figsize'] = 2.5, 2.5


coarse_time = 1.00
fine_time = 5.99
P_values = [2, 4, 6, 8, 10, 12, 14, 16]


print("........................GPU................................")

runtimes_numeric_gpu = np.genfromtxt('Parareal/Data/runtime_num_gpu.txt')
runtimes_pinn_gpu = np.genfromtxt('Parareal/Data/runtime_pinn_gpu.txt')
runtimes_pinn_gpu_serial = np.genfromtxt('Parareal/Data/runtime_serial_pinn_gpu.txt')
runtimes_num_gpu_serial = np.genfromtxt('Parareal/Data/runtime_serial_num_gpu.txt')

fig = plt.figure()
plt.semilogy(P_values, runtimes_num_gpu_serial, 'k-', label='Serial-Numeric', markersize=fs/2)
plt.semilogy(P_values, runtimes_numeric_gpu[::-1], 'k-o', label='Numeric', markersize=fs/2)
plt.semilogy(P_values, runtimes_pinn_gpu_serial, 'g-', label='Serial-PINN-GPU', markersize=fs/2)
plt.semilogy(P_values, runtimes_pinn_gpu[::-1],'g-o', label='PINN-GPU', markersize=fs/2)
plt.xlabel('cores', fontsize=fs, labelpad=0.5)
plt.ylabel('Runtimes [ms]', fontsize=fs, labelpad=0.5)
plt.legend(loc='upper right', fontsize=fs, prop={'size': fs-2})
plt.xticks(P_values, fontsize=fs)
plt.yticks(fontsize=fs)
bottom, top = plt.ylim()
plt.gcf().savefig("runtime_pinn_gpu.pdf", bbox_inches='tight')
plt.show()


print("........................CPU................................")

runtimes_numeric_cpu = np.genfromtxt('Parareal/Data/runtime_num_cpu.txt')
runtimes_pinn_cpu = np.genfromtxt('Parareal/Data/runtime_pinn_cpu.txt')
runtimes_pinn_cpu_serial = np.genfromtxt('Parareal/Data/runtime_serial_pinn_cpu.txt')
runtimes_num_cpu_serial = np.genfromtxt('Parareal/Data/runtime_serial_num_cpu.txt')

fig = plt.figure()
plt.semilogy(P_values, runtimes_num_cpu_serial, 'k-', label='Serial-Numeric', markersize=fs/2)
plt.semilogy(P_values, runtimes_numeric_cpu[::-1],'k-o', label='Numeric', markersize=fs/2)
plt.semilogy(P_values, runtimes_pinn_cpu_serial, 'b-', label='Serial-PINN-CPU', markersize=fs/2)
plt.semilogy(P_values, runtimes_pinn_cpu[::-1],'b-o', label='PINN-CPU', markersize=fs/2)
plt.xlabel('cores', fontsize=fs, labelpad=0.5)
plt.ylabel('Runtimes [ms]', fontsize=fs, labelpad=0.5)
plt.legend(loc='upper right', fontsize=fs, prop={'size': fs-2})
plt.xticks(P_values, fontsize=fs)
plt.yticks(fontsize=fs)
plt.ylim(bottom)
plt.gcf().savefig("runtime_pinn_cpu.pdf", bbox_inches='tight')
plt.show()


print("........................Speed Up................................")

def speed_up(n_iteration, cost_coarse, cost_fine):
    data = []
    for size in np.arange(2, 18, 2):
        speed_up = 1 / ((1 + n_iteration / size) * (cost_coarse / cost_fine) + (n_iteration / size))
        efficiency = speed_up / size
        data.append([size, speed_up, efficiency])
        df = pd.DataFrame(data, columns=['Size', 'Speed-Up', 'Efficiency'])
    return df


n_iteration = 1

#  Numeric Coarse Propagator
cost_fine_num = 5.32456
cost_coarse_num = 3.01543
cost_coarse_theory_num = 9.035665
cost_fine_theory_num = 18.624556
df_theory_numeric = speed_up(n_iteration, cost_coarse_theory_num, cost_fine_theory_num)
df_numeric = speed_up(n_iteration, cost_coarse_num, cost_fine_num)


#  PINN Coarse Propagator (CPU)
cost_fine_cpu = 3.6745656
cost_coarse_cpu = 1.35554
cost_coarse_theory_cpu = 5.0534554
cost_fine_theory_cpu = 17.734545
df_theory_pinn_cpu = speed_up(n_iteration, cost_coarse_theory_cpu, cost_fine_theory_cpu)
df_pinn_cpu = speed_up(n_iteration, cost_coarse_cpu, cost_fine_cpu)


#  PINN Coarse Propagator (GPU)
cost_fine_gpu = 4.7897
cost_coarse_gpu = 1.00
cost_coarse_theory_gpu = 3.01896
cost_fine_theory_gpu = 16.73465
df_theory_pinn_gpu = speed_up(n_iteration, cost_coarse_theory_gpu, cost_fine_theory_gpu)
df_pinn_gpu = speed_up(n_iteration, cost_coarse_gpu, cost_fine_gpu)


fig = plt.figure()
plt.plot(df_theory_numeric['Size'], df_theory_numeric['Speed-Up'], 'k-',  label='Bound Numeric', markersize=fs/2, linewidth=0.3)
plt.plot(df_numeric['Size'], df_numeric['Speed-Up'], 'k-o', label='Numeric', markersize=fs/2)
plt.plot(df_theory_pinn_gpu['Size'], df_theory_pinn_gpu['Speed-Up'], 'g-', label='Bound-PINN-GPU', markersize=fs/2, linewidth=0.3)
plt.plot(df_pinn_gpu['Size'], df_pinn_gpu['Speed-Up'], 'g-o', label='PINN-GPU', markersize=fs/2)
plt.plot(df_theory_pinn_cpu['Size'], df_theory_pinn_cpu['Speed-Up'], 'b-', label='Bound-PINN-CPU', markersize=fs/2, linewidth=0.3)
plt.plot(df_pinn_cpu['Size'], df_pinn_cpu['Speed-Up'], 'b-o', label='PINN-CPU', markersize=fs/2)
plt.xlabel('cores', fontsize=fs, labelpad=0.5)
plt.ylabel('speedup', fontsize=fs, labelpad=0.5)
plt.legend(loc='upper left', fontsize=fs, prop={'size': fs-2})
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.gcf().savefig("speedup_pinn_both_k_1.pdf", bbox_inches='tight')
plt.show()


fig = plt.figure()
plt.plot(df_numeric['Size'], df_numeric['Efficiency'], 'k-o', label='Numeric', markersize=fs/2)
plt.plot(df_pinn_gpu['Size'], df_pinn_gpu['Efficiency'], 'g-o', label='PINN-GPU', markersize=fs/2)
plt.plot(df_pinn_cpu['Size'], df_pinn_cpu['Efficiency'], 'b-o', label='PINN-CPU', markersize=fs/2)
plt.xlabel('cores', fontsize=fs, labelpad=0.5)
plt.ylabel('efficiency', fontsize=fs, labelpad=0.5)
plt.legend(loc='upper left', fontsize=fs, prop={'size': fs-2})
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.gcf().savefig("speedup_efficiency.pdf", bbox_inches='tight')
# plt.show()