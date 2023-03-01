from Numerics.BlackScholes.EuropeanOptions.CNEu import CNEu
from Numerics.BlackScholes.EuropeanOptions.ImplicitEu import ImplicitEu
import torch
from mpi4py import MPI
import time
import numpy as np

S0 = 100  # initial stock price
exercise_price = 20
sigma = 0.4  # volatility
r = 0.9  # interest rate 0.03
dividend = 0.00
is_call = True
Smax = 100
time_slice = [0, 1]  # start time and expiration times

# Propagator Parameters
M = 5000  # number of stock slices
N = 10  # number of time slices
N_coarse = 100  # number of coarse time steps
N_fine = 200
u0 = 0
n_iterations = 10

# time slices for parareal iteration
time_steps = np.round(np.linspace(time_slice[0], time_slice[1], N + 1), 2)
time_array = []
for first, second in zip(time_steps, time_steps[1:]):
    time_slice = [first, second]
    time_array.append(time_slice)
time_array = np.flip(time_array, 0)


# initial guess
initial_guess = ImplicitEu(S0, exercise_price, r, time_slice[1], sigma, Smax, M, N, u0, is_call)  # initial Guess
initial_guess.price()
initial_guess_ = initial_guess.grid

# Load the trained Model
pinn_model = torch.load("black_scholes_pnn.pt")
pinn_model.eval()

nn_model = torch.load("black_scholes_nn.pt")
nn_model.eval()

# # # Matrix to store parareal iterations
U_big = np.zeros(shape=(n_iterations + 1, M + 1, N + 1))
U_big[0, :, :] = initial_guess_[:, :]
U_big[:, :, 0] = initial_guess_[:, -1]

U_pinn = np.zeros(shape=(n_iterations + 1, M + 1, N + 1))
U_pinn[0, :, :] = initial_guess_[:, :]
U_pinn[:, :, 0] = initial_guess_[:, -1]

U_nn = np.zeros(shape=(n_iterations + 1, M + 1, N + 1))
U_nn[0, :, :] = initial_guess_[:, :]
U_nn[:, :, 0] = initial_guess_[:, -1]

stock_price_collocation = torch.randint(low=0, high=M + 1, size=(M + 1, 1)).type(torch.FloatTensor)


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start_time = time.time()

    for k in range(n_iterations):
        for i in range(N):
            # fine propagator
            fine = CNEu(S0, exercise_price, r, time_array[i][1], sigma, Smax, M, N_fine, U_big[k, :, i], is_call)
            fine.price()
            fine_guess = fine.grid

            # Pinn Coarse Propagator
            time_slice = torch.tensor(time_array[i][1]).float()
            time_tensor = time_slice.repeat(M + 1, 1)
            X_f = torch.cat((time_tensor, stock_price_collocation), 1)
            pinn_guess = pinn_model(time_array[i][1], X_f)

            # Pinn Coarse Propagator
            time_slice = torch.tensor(time_array[i][1]).float()
            time_tensor = time_slice.repeat(M + 1, 1)
            X_f = torch.cat((time_tensor, stock_price_collocation), 1)
            nn_guess = nn_model(time_array[i][1], X_f)

            time_slice = torch.tensor(time_array[i][1]).float()
            time_tensor = time_slice.repeat(M + 1, 1)
            X_f = torch.cat((time_tensor, stock_price_collocation), 1)
            pinn_alt = pinn_model(time_array[i][1], X_f)

            time_slice = torch.tensor(time_array[i][1]).float()
            time_tensor = time_slice.repeat(M + 1, 1)
            X_f = torch.cat((time_tensor, stock_price_collocation), 1)
            nn_alt = nn_model(time_array[i][1], X_f)

            comm.Allreduce(MPI.IN_PLACE, pinn_guess.detach().numpy(), op=MPI.SUM)
            comm.Allreduce(MPI.IN_PLACE, pinn_alt.detach().numpy(), op=MPI.SUM)
            comm.Allreduce(MPI.IN_PLACE, fine_guess, op=MPI.SUM)
            # correction step
            U_pinn[k + 1, :, i + 1] = pinn_guess.detach().numpy()[:, 0] + fine_guess[:, 0] - pinn_alt.detach().numpy()[:, 0]

    errors_pinn = np.zeros(n_iterations)
    errors_fine = np.zeros(n_iterations)
    errors_nn = np.zeros(n_iterations)

    for k in range(n_iterations):
        errors_pinn[k] = np.sqrt(np.sum((U_pinn[k + 1, :, -1] - fine_guess[:, 0]) ** 2)) / np.sqrt(
            np.sum(fine_guess[:, 0] ** 2))
        errors_nn[k] = np.sqrt(np.sum((U_nn[k + 1, :, -1] - fine_guess[:, 0]) ** 2)) / np.sqrt(
            np.sum(fine_guess[:, 0] ** 2))
        errors_fine[k] = np.sqrt(np.sum((U_big[k + 1, :, -1] - fine_guess[:, 0]) ** 2)) / np.sqrt(
            np.sum(fine_guess[:, 0] ** 2))

    end_time = time.time()
    local_time = end_time - start_time

    # Collect the local time on each process
    local_times = comm.gather(local_time, root=0)

    if rank == 0:
        # Compute the average time across all processes
        avg_time = sum(local_times) / size
        print("Average time for all processes - ML: ", avg_time)

        with open("run_times_pinn_cpu.txt", "a") as myfile:
            myfile.write(str(avg_time))


