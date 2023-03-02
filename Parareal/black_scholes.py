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
U_big_imp = np.zeros(shape=(n_iterations + 1, M + 1, N + 1))
U_big_imp[0, :, :] = initial_guess_[:, :]
U_big_imp[:, :, 0] = initial_guess_[:, -1]

U_pinn = np.zeros(shape=(n_iterations + 1, M + 1, N + 1))
U_pinn[0, :, :] = initial_guess_[:, :]
U_pinn[:, :, 0] = initial_guess_[:, -1]

U_nn = np.zeros(shape=(n_iterations + 1, M + 1, N + 1))
U_nn[0, :, :] = initial_guess_[:, :]
U_nn[:, :, 0] = initial_guess_[:, -1]

stock_price_collocation = torch.randint(low=0, high=M + 1, size=(M + 1, 1)).type(torch.FloatTensor)


if __name__ == '__main__':
    for k in range(n_iterations):
        for i in range(N):
            # fine propagator
            fine = CNEu(S0, exercise_price, r, time_array[i][1], sigma, Smax, M, N_fine, U_big_imp[k, :, i], is_call)
            fine.price()
            fine_guess = fine.grid

            # coarse propagator (Implicit)
            coarse_imp = coarse = ImplicitEu(S0, exercise_price, r, time_array[i][0], time_array[i][1], sigma, Smax, M, N_coarse, U_big_imp[k + 1, :,i], is_call)
            coarse_imp.price()
            coarse_guess_imp = coarse_imp.grid

            # Pinn Coarse Propagator
            time_slice = torch.tensor(time_array[i][1]).float()
            time_tensor = time_slice.repeat(M + 1, 1)
            X_f = torch.cat((time_tensor, stock_price_collocation), 1)
            pinn_guess = pinn_model(time_array[i][1], X_f)

            # NN Coarse Propagator
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

            # coarse propagator (second)
            coarse_alt_imp = ImplicitEu(S0, exercise_price, r, time_array[i][0], time_array[i][1], sigma, Smax, M, N_coarse, U_big_imp[k , :, i], is_call)
            coarse_alt_imp.price()
            coarse_guess_alt_imp = coarse_alt_imp.grid

            # correction step
            U_pinn[k + 1, :, i + 1] = pinn_guess.detach().numpy()[:, 0] + fine_guess[:, 0] - pinn_alt.detach().numpy()[:, 0]
            U_nn[k + 1, :, i + 1] = nn_guess.detach().numpy()[:, 0] + fine_guess[:, 0] - nn_alt.detach().numpy()[:, 0]
            U_big_imp[k + 1, :, i + 1] = nn_guess.detach().numpy()[:, 0] + fine_guess[:, 0] - coarse_guess_alt_imp.detach().numpy()[:, 0]


    errors_pinn = np.zeros(n_iterations)
    errors_num = np.zeros(n_iterations)
    errors_nn = np.zeros(n_iterations)

    for k in range(n_iterations):
        errors_pinn[k] = np.sqrt(np.sum((U_pinn[k + 1, :, -1] - fine_guess[:, 0]) ** 2)) / np.sqrt(
            np.sum(fine_guess[:, 0] ** 2))
        errors_nn[k] = np.sqrt(np.sum((U_nn[k + 1, :, -1] - fine_guess[:, 0]) ** 2)) / np.sqrt(
            np.sum(fine_guess[:, 0] ** 2))
        errors_num[k] = np.sqrt(np.sum((U_big_imp[k + 1, :, -1] - fine_guess[:, 0]) ** 2)) / np.sqrt(
            np.sum(fine_guess[:, 0] ** 2))


fs = 8
rcParams['figure.figsize'] = 2.5, 2.5
fig = plt.figure()
k_array = np.linspace(0, iter, iter+1)
subcount = 1
plt.semilogy(k_array, errors_pinn, 'b-o',  label='PINN', markersize=fs/2)
plt.semilogy(k_array, errors_nn, 'k-s', label='Numeric', markersize=fs/2)
plt.semilogy(k_array, errors_num, 'g-v', label='NN', markersize=fs/2)
plt.xticks(k_array, fontsize=fs)
plt.xlabel('K', fontsize=fs, labelpad=0.5)
plt.ylabel('Normalized error', fontsize=fs, labelpad=0.5)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.legend(loc='center right', fontsize=fs, prop={'size': fs-2})
plt.gcf().savefig("parareal_pinn_nn.pdf", bbox_inches='tight')
plt.show()
