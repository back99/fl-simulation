import json
import matplotlib.pyplot as plt
import numpy as np

clients_list = [10, 20, 50]
workers_list = [2, 4, 8]

serial_times = {}
for n in clients_list:
    with open(f"results/serial_clients{n}.json") as f:
        serial_times[n] = json.load(f)["total_time"]

parallel_times = {}
for n in clients_list:
    parallel_times[n] = {}
    for w in workers_list:
        with open(f"results/parallel_clients{n}_workers{w}.json") as f:
            parallel_times[n][w] = json.load(f)["total_time"]

speedups = {}
for n in clients_list:
    speedups[n] = [serial_times[n] / parallel_times[n][w] for w in workers_list]

# Plot 1: Speedup vs Workers
plt.figure(figsize=(8, 5))
for n in clients_list:
    plt.plot(workers_list, speedups[n], marker='o', label=f'{n} clients')
plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Serial baseline')
plt.xticks([2, 4, 8])
plt.xlabel("Number of Workers")
plt.ylabel("Speedup (Serial / Parallel)")
plt.title("Speedup vs Number of Workers")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/speedup_vs_workers.png", dpi=150)
plt.show()

# Plot 2: Total Time vs Workers
plt.figure(figsize=(8, 5))
for n in clients_list:
    times = [parallel_times[n][w] for w in workers_list]
    plt.plot(workers_list, times, marker='s', label=f'{n} clients (parallel)')
    plt.axhline(y=serial_times[n], linestyle='--', alpha=0.5, label=f'{n} clients (serial)')
plt.xticks([2, 4, 8])
plt.xlabel("Number of Workers")
plt.ylabel("Total Time (s)")
plt.title("Total Execution Time vs Number of Workers")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/time_vs_workers.png", dpi=150)
plt.show()

print("Graphs saved to results/")
