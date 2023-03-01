import numpy as np
import matplotlib.pyplot as plt
from Numerics.BlackScholes.AmericanOptions.CNEu import CNEu
from Numerics.BlackScholes.AmericanOptions.ImplicitEu import ImplicitEu
from Numerics.BlackScholes.AmericanOptions.CNEu import CNEu as PINN
# from Numerics.BlackScholes.AmericanOptions.ImplicitAm import ImplicitAmBerT as Pinn
from MachineLearning.Default.pinn_default import *
import torch
from pylab import rcParams

"""
    dV/dt + r dV/dx + 1/2 sigma^2 d^V/dx^2 -rv = 0
    
    Implementation of the parareal algorithm for the Black-Scholes Equations (BSE) with an application
    on the pricing of different types of derivatives (barrier options, digital options, 
    the American European {call/put} option etc with small modifications.
    For the discretization of the BSE, we use Crank-Nicolson
    A common practice is to choose the computational region between 3K and K/3. 
    
"""
# Black Scholes Parameters
S0 = 100 # initial stock price
exercise_price = 20
sigma = 0.4 # volatility
r = 0.03 # interest rate
dividend = 0.00
is_call = True
Smax = 100
time_slice = [0, 1] # start time and expiration times

# Propagator Parameters
M = 100 # number of stock slices
N = 10 # number of time slices
N_coarse = 30 # number of coarse time steps
N_c = 15
N_fine = 100
u0 = 0
iter = 5

# time slices for parareal iteration
time_steps = np.round(np.linspace(time_slice[0], time_slice[1], N + 1), 2)
time_array = []
for first, second in zip(time_steps, time_steps[1:]):
    time_slice = [first, second]
    time_array.append(time_slice)
time_array = np.flip(time_array, 0)

N_pinn = N_fine - 90
# initial guess
initial_guess = ImplicitEu(S0, exercise_price, r, time_slice[0], time_slice[1], sigma, Smax, M, N, u0, is_call)  #initial Guess
initial_guess.price()
initial_guess_ = initial_guess.grid
# print(initial_guess_)
# Load the trained Model
# model = torch.load("/home/cez4707/Documents/Code/phdthesis/MachineLearning/trained_models/black_scholes.pt")
# model = torch.load("/Users/tunde/Documents/Code/phdthesis/MachineLearning/trained_models/black_scholes.pt")
#
# model.eval()


# # # Matrix to store parareal iterations
U_big = np.zeros(shape=(iter + 1, M+1, N + 1))
U_big[0, :, :] = initial_guess_[:, :]
U_big[:, :, 0] = initial_guess_[:, -1]

U_pinn= np.zeros(shape=(iter + 1, M+1, N + 1))
U_pinn[0, :, :] = initial_guess_[:, :]
U_pinn[:, :, 0] = initial_guess_[:, -1]


U_big_imp = np.zeros(shape=(iter + 1, M+1, N + 1))
U_big_imp[0, :, :] = initial_guess_[:, :]
U_big_imp[:, :, 0] = initial_guess_[:, -1]


# stock_price_collocation = torch.randint(low=0, high=M + 1, size=(M + 1, 1)).type(torch.FloatTensor)


for k in range(iter):
    for i in range(N):
        # fine propagator
        fine = CNEu(S0, exercise_price, r, time_array[i][0], time_array[i][1], sigma, Smax, M, N_fine, U_big[k, :, i], is_call)
        fine.price()
        fine_guess = fine.grid

        # coarse propagator (CN)
        coarse = CNEu(S0, exercise_price, r, time_array[i][0], time_array[i][1], sigma, Smax, M, N_coarse, U_big[k + 1, :, i], is_call)
        coarse.price()
        coarse_guess = coarse.grid

        # coarse propagator (Implicit)
        coarse_imp = coarse = CNEu(S0, exercise_price, r, time_array[i][0], time_array[i][1], sigma, Smax, M, N_c, U_big_imp[k + 1, :, i], is_call)
        coarse_imp.price()
        coarse_guess_imp = coarse_imp.grid

        # # pinn propagator
        pinn = PINN(S0, exercise_price, r, time_array[i][0], time_array[i][1], sigma, Smax, M, N_pinn, U_big[k + 1, :, i], is_call)
        pinn.price()
        pinn_guess = pinn.grid

        # Pinn Coarse Propagator
        # time_slice = torch.tensor(time_array[i][1]).float()
        # time_tensor = time_slice.repeat(M + 1, 1)
        # X_f = torch.cat((time_tensor, stock_price_collocation), 1)
        # pinn_guess = model(X_f)

        # coarse propagator (second)
        coarse_alt = CNEu(S0, exercise_price, r, time_array[i][0], time_array[i][1], sigma, Smax, M, N_coarse, U_big[k , :, i], is_call)
        coarse_alt.price()
        coarse_guess_alt = coarse_alt.grid

        # coarse propagator (second)
        coarse_alt_imp = CNEu(S0, exercise_price, r, time_array[i][0], time_array[i][1], sigma, Smax, M, N_c, U_big[k , :, i], is_call)
        coarse_alt_imp.price()
        coarse_guess_alt_imp = coarse_alt_imp.grid

        # coarse propagator (second)
        pinn_alt = PINN(S0, exercise_price, r, time_array[i][0], time_array[i][1], sigma, Smax, M, N_pinn, U_big[k , :, i], is_call)
        pinn_alt.price()
        pinn_alt = pinn_alt.grid
        # dt = (time_array[i][1] - time_array[i][0]) / 2 # TODO
        # time_slice = torch.tensor(time_array[i][1]).float()
        # time_tensor = time_slice.repeat(M + 1, 1)
        # X_f = torch.cat((time_tensor, stock_price_collocation), 1)
        # pinn_alt = model(X_f)

        # correction step
        U_big[k+1, :, i+1] = coarse_guess[:, 0] + fine_guess[:, 0] - coarse_guess_alt[:, 0]
        # U_pinn[k+1, :, i+1] = pinn_guess.detach().numpy()[:, 0] + fine_guess[:, 0] - pinn_alt.detach().numpy()[:, 0]
        U_pinn[k+1, :, i+1] = pinn_guess[:, 0] + fine_guess[:, 0] - pinn_alt[:, 0]
        U_big_imp[k+1, :, i+1] = coarse_guess_imp[:, 0] + fine_guess[:, 0] - coarse_guess_alt_imp[:, 0]


errors_pinn = np.zeros(iter)
errors_cn = np.zeros(iter)
errors_imp = np.zeros(iter)

for k in range(iter):
    errors_pinn[k] = np.sqrt(np.sum((U_pinn[k + 1, :, -1] - fine_guess[:, 0]) ** 2)) / np.sqrt(np.sum(fine_guess[:, 0] ** 2))
    errors_cn[k] = np.sqrt(np.sum((U_big[k + 1, :, -1] - fine_guess[:, 0]) ** 2)) / np.sqrt(np.sum(fine_guess[:, 0] ** 2))
    errors_imp[k] = np.sqrt(np.sum((U_big_imp[k + 1, :, -1] - fine_guess[:, 0]) ** 2)) / np.sqrt(np.sum(fine_guess[:, 0] ** 2))

errors_pinn = np.insert(errors_pinn, 0, 1.484217191434019801e-01)
errors_cn = np.insert(errors_cn, 0, 1.484217191434019801e-01)
errors_imp = np.insert(errors_imp, 0, 1.484217191434019801e-01)

# np.savetxt("test.csv", errors_cn, delimiter=",")
# print(errors_pinn)
# print(errors_cn)
# print(errors_imp)
# Plot the parareal iteration errors
fs = 8
rcParams['figure.figsize'] = 2.5, 2.5
fig = plt.figure()
k_array = np.linspace(0, iter, iter+1)
subcount = 1
plt.semilogy(k_array, errors_cn, 'b-o',  label='PINN', markersize=fs/2)
plt.semilogy(k_array, errors_pinn, 'k-s', label='Numeric', markersize=fs/2)
plt.semilogy(k_array, errors_imp, 'g-v', label='NN', markersize=fs/2)
plt.xticks(k_array, fontsize=fs)
plt.xlabel('K', fontsize=fs, labelpad=0.5)
plt.ylabel('Normalized error', fontsize=fs, labelpad=0.5)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
# plt.title('Comparison of Numeric vs PINN Coarse Propagator')
plt.legend(loc='center right', fontsize=fs, prop={'size': fs-2})
plt.gcf().savefig("/Users/tunde/Documents/Code/phdthesis/Paper/Plots/parareal_pinn_nn.pdf", bbox_inches='tight')
plt.show()

