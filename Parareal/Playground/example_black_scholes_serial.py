# import matplotlib.pyplot as plt
import numpy as np
# from Numerics.BlackScholes.AmericanOptions.CNEu import CNEu
from Numerics.BlackScholes.AmericanOptions.ImplicitEu import ImplicitEu
# import torch
# from mpi4py import MPI
# import time

"""
    dV/dt + r dV/dx + 1/2 sigma^2 d^V/dx^2 -rv = 0
    
    Implementation of the parareal algorithm for the Black-Scholes Equations (BSE) with an application
    on the pricing of different types of derivatives (barrier options, digital options, 
    the American European {call/put} option etc with small modifications.
    For the discretization of the BSE, we use Crank-Nicolson
    A common practice is to choose the computational region between 3K and K/3. 
"""

# Black Scholes Parameters
S0 = 100  # initial stock price
exercise_price = 20
sigma = 0.4  # volatility
r = 0.9  # interest rate 0.03
dividend = 0.00
is_call = True
Smax = 100
time_slice = [0, 1]  # start time and expiration times

# Propagator Parameters
M = 500  # number of stock slices
N = 10  # number of time slices
N_coarse = 50  # number of coarse time steps
N_fine = 100
u0 = 0
n_iterations = 10

# time slices for parareal iteration
time_steps = np.round(np.linspace(time_slice[0], time_slice[1], N + 1), 2)
time_array = []
for first, second in zip(time_steps, time_steps[1:]):
    time_slice = [first, second]
    time_array.append(time_slice)
time_array = np.flip(time_array, 0)


initial_guess = ImplicitEu(S0, exercise_price, r, time_slice[0], time_slice[1], sigma, Smax, M, N, u0, is_call)  # initial Guess
initial_guess.price()
initial_guess_ = initial_guess.grid

print(initial_guess_[:, 0])
# # Matrix to store parareal iterations
# U_big = np.zeros(shape=(n_iterations + 1, M + 1, N + 1))
# U_big[0, :, :] = initial_guess_[:, :]
# U_big[:, :, 0] = initial_guess_[:, -1]
#
# start_time = time.time()
#
# for k in range(n_iterations):
#     # print('iter', k)
#     for i in range(N):
#         # fine propagator
#         fine = CNEu(S0, exercise_price, r, time_array[i][1], sigma, Smax, M, N_fine, U_big[k, :, i], is_call)
#         fine.price()
#         fine_guess = fine.grid
#
#         # coarse propagator (CN)
#         coarse = CNEu(S0, exercise_price, r, time_array[i][1], sigma, Smax, M, N_coarse, U_big[k + 1, :, i], is_call)
#         coarse.price()
#         coarse_guess = coarse.grid
#
#         # coarse propagator (second)
#         coarse_alt = CNEu(S0, exercise_price, r, time_array[i][1], sigma, Smax, M, N_coarse, U_big[k, :, i], is_call)
#         coarse_alt.price()
#         coarse_guess_alt = coarse_alt.grid
#
#         # correction step
#         U_big[k + 1, :, i + 1] = coarse_guess[:, 0] + fine_guess[:, 0] - coarse_guess_alt[:, 0]
# #
# errors_cn = np.zeros(n_iterations)
#
# for k in range(n_iterations):
#     errors_cn[k] = np.sqrt(np.sum((U_big[k + 1, :, -1] - fine_guess[:, 0]) ** 2)) / np.sqrt(
#         np.sum(fine_guess[:, 0] ** 2))
#
# # np.savetxt("error_serial.csv", errors_cn, delimiter=",")
# # Plot the parareal iteration errors
# # k_array = np.linspace(1, n_iterations, n_iterations)
# # subcount = 1
# # plt.figure(figsize=(15, 6), dpi=100)
# # plt.semilogy(k_array, errors_cn, '-o', color='red', label='CN')
# # # plt.semilogy(k_array, errors_pinn, '-o', color='blue', label='PINN')
# # plt.xticks(k_array)
# # plt.xlabel('K')
# # plt.ylabel('Normalized error')
# # plt.title('Logarithmic scale, M(Space) = %s, N(Time) = %s, N_fine = %s, N_coarse = %s' % (M, N, N_fine, N_coarse))
# # plt.legend(loc='upper left', prop={'size': 15})
# # plt.show()
#
# #
# end_time = time.time()
# local_time = end_time - start_time
# # print("Average time for all processes - Serial: ", local_time)
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
# if rank == 0:
#     with open("run_times_serial.txt", "a") as myfile:
#         myfile.write(str(local_time))