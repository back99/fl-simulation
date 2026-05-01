import json
import matplotlib.pyplot as plt
import numpy as np

# Amdahl's Law: S(p) = 1 / ((1 - f) + f/p)
# f = parallel fraction, p = number of processors
# We estimate f from our experimental data

clients_list = [50, 100, 200]
workers_list = [2, 4, 8, 16]

# Load serial baseline times
serial_times = {}
for n in clients_list:
    with open(f"results/serial_clients{n}.json") as f:
        serial_times[n] = json.load(f)["total_time"]

# Load parallel times
parallel_times = {}
for n in clients_list:
    parallel_times[n] = {}
    for w in workers_list:
        with open(f"results/parallel_clients{n}_workers{w}.json") as f:
            parallel_times[n][w] = json.load(f)["total_time"]

# Calculate actual speedups
actual_speedups = {}
for n in clients_list:
    actual_speedups[n] = [serial_times[n] / parallel_times[n][w] for w in workers_list]

# Estimate parallel fraction f using Amdahl's Law
# S = 1 / ((1-f) + f/p) -> solve for f using workers=8 data
def estimate_f(speedup, p):
    # f = (1/S - 1) / (1/p - 1)
    return (1/speedup - 1) / (1/p - 1)

# Use workers=8 speedup to estimate f for each client count
f_estimates = {}
for n in clients_list:
    s8 = actual_speedups[n][2]  # workers=8
    f_estimates[n] = estimate_f(s8, 8)

# Generate Amdahl's Law theoretical curve
p_range = np.linspace(1, 16, 100)

# Plot: Amdahl's Law vs Actual Speedup
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Amdahl's Law (Theoretical) vs Actual Speedup", fontsize=13)

for idx, n in enumerate(clients_list):
    ax = axes[idx]
    f = f_estimates[n]

    # Theoretical curve
    theoretical = 1 / ((1 - f) + f / p_range)
    ax.plot(p_range, theoretical, 'r--', label=f"Amdahl's Law (f={f:.2f})", linewidth=2)

    # Actual speedup points
    ax.plot(workers_list, actual_speedups[n], 'bo-', label='Actual speedup', linewidth=2, markersize=8)

    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Serial baseline')
    ax.set_xlabel("Number of Workers")
    ax.set_ylabel("Speedup")
    ax.set_title(f"{n} Clients")
    ax.set_xticks([2, 4, 8, 16])
    ax.legend(fontsize=8)
    ax.grid(True)

plt.tight_layout()
plt.savefig("results/amdahl_vs_actual.png", dpi=150)
plt.show()

# Print estimated parallel fractions
print("\nEstimated parallel fraction (f) per client count:")
for n in clients_list:
    print(f"  clients={n}: f = {f_estimates[n]:.4f} ({f_estimates[n]*100:.1f}% parallelizable)")

print("\nTheoretical max speedup (p=infinity):")
for n in clients_list:
    f = f_estimates[n]
    print(f"  clients={n}: max speedup = {1/(1-f):.2f}x")