import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def run_mpi_command(nprocs, command):
    cmd = "mpirun -np {} {}".format(nprocs, command)
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.decode()


def parse_output(output):
    for line in output.split("\n"):
        if "Initialize a time: " in line:
            return float(line.strip().split(" ")[-1])


def main(command, processor_numbers, n_repeats=5):
    runtimes = []
    for nprocs in processor_numbers:
        run_times = []
        for i in range(n_repeats):
            output = run_mpi_command(nprocs, command)
            runtime = parse_output(output)
            run_times.append(runtime)
        avg_runtime = np.mean(run_times)
        sd_runtime = np.sd(runtimes)
        runtimes.append((nprocs, avg_runtime, sd_runtime))
    df = pd.DataFrame(runtimes, columns=["Number of Processors", "Average Runtime",
                                         'Standard Deviation'])
    return df


if __name__ == "__main__":
    command = "python3"
    processor_numbers = [2, 4, 6, 8, 10, 12, 14, 16]
    df = main(command, processor_numbers)
    plt.plot(df["Number of Processors"], df["Average Runtime"], marker="o")
    plt.xlabel("Number of Processors")
    plt.ylabel("Avg. Runtime")
    plt.title("Parareal MPI Runtimes")
    plt.show()
    df.to_csv("runtimes.csv")
