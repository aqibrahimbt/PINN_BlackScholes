import time
import numpy as np
from mpi4py import MPI

# Define the number of runs
n_runs = 10

# Define the range of processor counts to test
n_procs_range = [1, 2, 4, 8, 10, 12, 14, 18]

# Initialize MPI and get the rank and size
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define a function to run the parareal with pipelining implementation and return the elapsed time
def run_parareal():
    start_time = time.time()

    # Call the file that contains the implementation code
    # specify the file to run here
    exec(open('black_scholes_num.py').read())

    elapsed_time = time.time() - start_time
    return elapsed_time

# Initialize an array to hold the elapsed times for each run
elapsed_times = np.zeros((len(n_procs_range), n_runs))

# Loop over the processor counts to test
for i, n_procs in enumerate(n_procs_range):
    # Set the number of processors to use
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Loop over the number of runs
    for j in range(n_runs):
        # Set the number of processors to use
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Set the number of time slices to use
        time_slices = size

        # Run the parareal with pipelining implementation and record the elapsed time
        elapsed_times[i, j] = run_parareal()

    # Print the average elapsed time and standard deviation for this number of processors
    if rank == 0:
        print(f"Number of processors: {n_procs}")
        print(f"Average elapsed time: {np.mean(elapsed_times[i]):.2f} seconds")
        print(f"Standard deviation: {np.std(elapsed_times[i]):.2f} seconds")
        print()
