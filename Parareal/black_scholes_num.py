from Numerics.BlackScholes.EuropeanOptions.CNEu import CNEu
from Numerics.BlackScholes.EuropeanOptions.ImplicitEu import ImplicitEu
import torch
from mpi4py import MPI
import time
import numpy as np
from pylab import rcParams
import matplotlib.pyplot as plt

S0 = 100  # initial stock price
exercise_price = 20
sigma = 0.4  # volatility
r = 0.9  # interest rate 0.03
dividend = 0.00
is_call = True
Smax = 5000
time_slice = [0, 1]  # start time and expiration times

# Propagator Parameters
M = 5000  # number of stock slices
N = 10  # number of time slices
N_coarse = 100  # number of coarse time steps
N_fine = 200
u0 = 0
n_iterations = 3

# time slices for parareal iteration
time_slice = 16

# Load the trained Model
pinn_model = torch.load("black_scholes_pnn.pt")
pinn_model.eval()

nn_model = torch.load("black_scholes_nn.pt")
nn_model.eval()


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define time array
t_start = 0
t_end = 1
t_step = (t_end - t_start) / time_slice
time_array = []
for i in range(time_slice):
    t_slice_start = t_start + i * t_step
    t_slice_end = t_slice_start + t_step
    time_array.append((t_slice_start, t_slice_end))

# Initialize U_big_imp array
U_big_imp = np.zeros((n_iterations + 1, M + 1, N_fine + 1))

# Initialize nn_guess array
nn_guess = np.zeros((M + 1, 1))

# Start timer
start_time = time.time()

for k in range(n_iterations):
    for i in range(time_slice):

        # Compute fine propagator
        if rank == i:
            fine = CNEu(S0, exercise_price, r, time_array[i][1], sigma, Smax, M, N_fine, U_big_imp[k, :, i], is_call)
            fine.price()
            fine_guess = fine.grid
        else:
            fine_guess = np.zeros((M + 1, 1))

        # Send fine_guess to next processor
        if rank < size - 1:
            comm.send(fine_guess, dest=rank+1)

        # Receive fine_guess from previous processor
        if rank > 0:
            fine_guess = comm.recv(source=rank-1)

        # Compute coarse propagator (Implicit)
        if rank == i:
            coarse_imp = ImplicitEu(S0, exercise_price, r, time_array[i][0], time_array[i][1], sigma, Smax, M, N_coarse, U_big_imp[k + 1, :,i], is_call)
            coarse_imp.price()
            coarse_guess_imp = coarse_imp.grid
        else:
            coarse_guess_imp = np.zeros((M + 1, 1))

        # Send coarse_guess_imp to next processor
        if rank < size - 1:
            comm.send(coarse_guess_imp, dest=rank+1)

        # Receive coarse_guess_alt_imp from previous processor
        if rank > 0:
            coarse_guess_alt_imp = comm.recv(source=rank-1)

        # Compute coarse propagator (second)
        if rank == i:
            coarse_alt_imp = ImplicitEu(S0, exercise_price, r, time_array[i][0], time_array[i][1], sigma, Smax, M, N_coarse, U_big_imp[k , :, i], is_call)
            coarse_alt_imp.price()
            coarse_guess_alt_imp = coarse_alt_imp.grid

        # Send coarse_guess_alt_imp to previous processor
        if rank > 0:
            comm.send(coarse_guess_alt_imp, dest=rank-1)

        # Receive nn_guess from next processor
        if rank < size - 1:
            nn_guess = comm.recv(source=rank+1)

        # Compute U_big_imp
        U_big_imp[k + 1, :, i + 1] = nn_guess[:, 0] + fine_guess[:, 0] - coarse_guess_alt_imp[:, 0]

        # Send nn_guess to next processor
        if rank < size - 1:
            comm.send(nn_guess, dest=rank+1)

        # Receive U_big_imp from previous processor
        if rank > 0:
            U_big_imp[k + 1, :, i] = comm.recv(source=rank-1)

        elapsed_time = time.time() - start_time

        if rank == 0:
            print(f"Elapsed time: {elapsed_time:.2f} seconds")

