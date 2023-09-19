import numpy as np
from scipy.optimize import fsolve
from mpi4py import MPI


def f(q, t):
    return np.sin(q) * t

def coarse(q, t0, t1, h=0.1):
    t = np.arange(t0, t1+h, h)
    for i in range(1, len(t)):
        q = q + h * f(q, t[i-1])
    return q


def fine(q, t0, t1, h=0.00001):
    t = np.arange(t0, t1+h, h)
    q_old = q
    for i in range(1, len(t)):
        q = fsolve(lambda q: q - (q_old + h * f(q, t[i])), q)
        q_old = q
    return q

# def f(q, t, S, r, sigma, dt):
#     dS = r * S * dt + sigma * S * np.sqrt(dt) * np.random.normal()
#     return max(0, S - q)
#
#
# def coarse(q, t0, t1, S, r, sigma, h=0.1):
#     dt = h
#     S = S + r * S * dt + sigma * S * np.sqrt(dt) * np.random.normal()
#     return max(0, S - q)
#
#
# def fine(q, t0, t1, S, r, sigma, h=0.001):
#     dt = t1 - t0
#     t = np.arange(t0, t1 + h, h)
#     q_old = q
#     for i in range(1, len(t)):
#         dS = r * S * dt + sigma * S * np.sqrt(dt) * np.random.normal()
#         S = S + dS
#         func = lambda q: q - q_old - 0.5 * h * (f(q, t[i], S, r, sigma, dt) + f(q_old, t[i-1], S, r, sigma, dt))
#         q = fsolve(func, q)
#         q_old = q
#     return max(0, S - q)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    q_o = 120
    q = q_o

    time_interval = [0, 1]
    segment_size = (time_interval[1] - time_interval[0]) / (size + 1)
    time_segment = [time_interval[0] + rank * segment_size, time_interval[0] + (rank + 1) * segment_size]

    # print("rank =", rank, "  time_segment = ", time_segment)

    K = 10
    S = 100
    r  = 0.4
    sigma = 0.4

    start_time = MPI.Wtime()

    # if rank == 0:
    q = coarse(q, time_interval[0], time_segment[1] - segment_size)

    q_c = coarse(q, time_segment[0] + segment_size, time_segment[1] + segment_size)
    # print("rank =", rank, " ", time_segment[0] + segment_size, time_segment[1] + segment_size)

    for _ in range(K):
        q = fine(q, time_segment[0] * segment_size, time_segment[1] * segment_size)
        d_c = q - q_c
        q = comm.recv(source=rank - 1) if rank != 0 else q_o
        q_c = coarse(q, time_segment[0] + segment_size, time_segment[1] + segment_size)
        q = q_c - d_c
        if rank != size - 1:
            comm.send(q, dest=rank + 1)

        end_time = MPI.Wtime()
    if rank == 0:
        print(f"Initialize a time: {str(end_time - start_time)}")
        with open("run_times_default.txt", "a") as myfile:
            myfile.write("\n")
            myfile.write(str(end_time-start_time))


if __name__ == '__main__':
    main()


