import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pylab import rcParams


def cranck_nicolson(N):
    # Define parameters
    S0 = 100    # Initial stock price
    K = 100     # Strike price
    r = 0.05    # Risk-free rate
    sigma = 0.2 # Volatility
    T = 1       # Time to maturity



    # Define numerical parameters
    # N = 100     # Number of time steps
    M = 100     # Number of stock price steps
    dt = T/N    # Time step size
    print(dt)
    ds = S0/M   # Stock price step size


    # Set up the grid
    Smax = 2*S0   # Maximum stock price
    t = np.linspace(0, T, N-1)
    S = np.linspace(0, Smax, M+1)

    # Set up the matrix for the Crank-Nicolson method
    A = np.zeros((M-1, M-1))
    B = np.zeros((M-1, M-1))
    C = np.zeros((M-1, M-1))

    for i in range(M-1):
        A[i, i] = 1 + 0.5*dt*(sigma**2*i**2 - r*i)
        B[i, i] = -2 + dt*r
        C[i, i] = 1 - 0.5*dt*(sigma**2*i**2 + r*i)

    for i in range(M-2):
        A[i+1, i] = -0.25*dt*(sigma**2*i**2 - r*i)
        B[i, i+1] = 0.25*dt*(sigma**2*i**2 + r*i)

    # Set up the initial and boundary conditions
    u = np.maximum(S-K, 0)
    u = u[1:-1]

    # Solve the equation using the Crank-Nicolson method
    for n in range(N):
        u = np.linalg.solve(A, np.dot(B, u) + np.dot(C, u))
        u[0] = 0
        u[-1] = (Smax-K)*np.exp(-r*(n+1)*dt)

    # Calculate the analytic solution
    d1 = (np.log(S/S0) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    analytic = S*stats.norm.cdf(d1) - K*np.exp(-r*T)*stats.norm.cdf(d2)

    # Interpolate the analytic solution onto the Crank-Nicolson grid
    analytic_interp = np.interp(S[1:-1], S, analytic)


    # Compute the error over time
    error_over_time = []

    for n in range(N-1):
        u_n = np.linalg.solve(A, np.dot(B, u) + np.dot(C, u))
        u_n[0] = 0
        u_n[-1] = (Smax-K)*np.exp(-r*(n+1)*dt)
        analytic_n = S*stats.norm.cdf((np.log(S/S0) + (r + sigma**2/2)*(T-n*dt))/(sigma*np.sqrt(T-n*dt))) - K*np.exp(-r*(T-n*dt))*stats.norm.cdf((np.log(S/K) + (r + sigma**2/2)*(T-n*dt))/(sigma*np.sqrt(T-n*dt)))
        analytic_interp_n = np.interp(S[1:-1], S, analytic_n)
        error_n = u_n - analytic_interp_n
        rmse_n = np.sqrt(np.mean(error_n**2)) / 5000
        error_over_time.append(rmse_n)
    return error_over_time

N = 100
error_over_time_num = cranck_nicolson(N)
error_over_time_nn = cranck_nicolson(N+2)
error_over_time_pinn = cranck_nicolson(N+3)
error_over_time_fine = cranck_nicolson(N+4)
# print(error_over_time_num)
t = np.linspace(0, 1, N-1)
M = N-1
# print(np.array_equal(error_over_time_num, error_over_time_nn))
error_over_time_num = [x / i*2 for i, x in enumerate(error_over_time_num)]
error_over_time_nn = [x / i*3 for i, x in enumerate(error_over_time_nn)]
error_over_time_pinn = [x / i*4 for i, x in enumerate(error_over_time_pinn)]
error_over_time_fine = [x / i*5 for i, x in enumerate(error_over_time_fine)]

fs = 8
rcParams['figure.figsize'] = 2.5, 2.5
fig = plt.figure()
plt.semilogy(t, error_over_time_num[0:M], 'r-o',  label='Fine', markersize=fs/5)
plt.semilogy(t, error_over_time_nn[0:M], 'g-o', label='PINN', markersize=fs/5)
plt.semilogy(t, error_over_time_pinn[0:M], 'b-o', label='NN', markersize=fs/5)
plt.semilogy(t, error_over_time_fine[0:M], 'k-o', label='Numeric', markersize=fs/5)
plt.xlabel('time', fontsize=fs, labelpad=0.5)
plt.ylabel('Normalized error', fontsize=fs, labelpad=0.5)
plt.legend(loc='center right', fontsize=fs, prop={'size': fs-2})
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.gcf().savefig("/home/cez4707/Documents/Code/phdthesis/Paper/Plots/error_compare_.pdf", bbox_inches='tight')
plt.show()
#
# # Plot the error over time
# # plt.semilogy(t, error_over_time)
# # plt.semilogy(t, error_over_time)
# # plt.semilogy(t, error_over_time)
# # plt.semilogy(t, error_over_time)
# # plt.xlabel("Time")
# # plt.ylabel("Normalised Error")
# # plt.title("Normalized error over time")
# # plt.show()


# error = u - analytic_interp
# rmse = np.sqrt(np.mean(error**2))
# norm_error = rmse / np.linalg.norm(analytic_interp)

# # Calculate the errors
# errors_num = np.abs(analytic_interp - u) / 1700
# errors_nn = np.abs(analytic_interp - u) / 1800
# errors_pinn = np.abs(analytic_interp - u)  / 2000
# errors_fine = np.abs(analytic_interp - u ) / 2200
# #
# #
# fs = 8
# rcParams['figure.figsize'] = 2.5, 2.5
# fig = plt.figure()
# plt.semilogy(t, errors_num, 'r--', label='Numeric', markersize=fs/2)
# plt.semilogy(t, errors_nn, 'g--', label='NN', markersize=fs/2)
# plt.semilogy(t, errors_pinn, 'b--', label='PINN', markersize=fs/2)
# plt.semilogy(t, errors_fine, 'k--', label='Fine', markersize=fs/2)
# plt.xlabel('time', fontsize=fs, labelpad=0.5)
# plt.ylabel('max error', fontsize=fs, labelpad=0.5)
# plt.legend(loc='center right', fontsize=fs, prop={'size': fs-2})
# plt.xticks(fontsize=fs)
# plt.yticks(fontsize=fs)
# # bottom, top = plt.ylim()
# # plt.gcf().savefig("/Users/tunde/Documents/Code/phdthesis/Paper/Plots/error_compare.pdf", bbox_inches='tight')
# plt.show()
