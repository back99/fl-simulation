import os
# Limit internal threads per process to avoid CPU contention between workers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from concurrent.futures import ProcessPoolExecutor
from model import SimpleCNN
from data import load_dataset, split_non_iid
from server import evaluate


def weights_to_numpy(state_dict):
    # Convert PyTorch tensors to NumPy arrays for safe inter-process transfer
    # PyTorch tensors cannot be pickled across processes (causes deadlock)
    return {k: v.numpy() for k, v in state_dict.items()}


def numpy_to_weights(numpy_dict):
    # Convert NumPy arrays back to PyTorch tensors inside each worker
    return {k: torch.from_numpy(v) for k, v in numpy_dict.items()}


def train_client_worker(args):
    # This function runs inside a separate worker process
    # Each worker trains one client independently

    # Limit threads inside each worker to prevent contention
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    client_id, numpy_weights, indices, epochs, lr = args

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset

    # Load dataset inside the worker process (avoids DataLoader pickle error)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST('./data', train=True, download=False, transform=transform)
    loader = DataLoader(Subset(train_data, indices), batch_size=32, shuffle=True)

    # Load the global model weights sent from the server
    model = SimpleCNN()
    model.load_state_dict(numpy_to_weights(numpy_weights))
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Train on local data for the given number of epochs
    for epoch in range(epochs):
        for images, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    print(f"  [Client {client_id}] done", flush=True)

    # Return weights as NumPy for safe transfer back to main process
    return client_id, weights_to_numpy(model.state_dict()), len(indices)


def fedavg_numpy(global_model, client_results):
    # FedAvg aggregation using NumPy arrays
    total_data = sum(size for _, _, size in client_results)
    avg_weights = {}
    keys = client_results[0][1].keys()
    for key in keys:
        # Weighted average based on each client's dataset size
        avg_weights[key] = sum(
            w[key] * (size / total_data)
            for _, w, size in client_results
        )
    global_model.load_state_dict(numpy_to_weights(avg_weights))
    return global_model


def run_parallel(num_clients=50, num_rounds=3, local_epochs=1, lr=0.01, num_workers=4):
    print(f"\n{'='*50}")
    print(f"[Parallel FL] clients={num_clients}, workers={num_workers}, rounds={num_rounds}")
    print(f"{'='*50}")

    train_data, test_data = load_dataset()
    client_indices = split_non_iid(train_data, num_clients)
    global_model = SimpleCNN()

    total_start = time.time()
    round_times = []
    # Track overhead breakdown for analysis
    serialization_times = []
    training_times = []
    aggregation_times = []
    acc = 0.0

    for r in range(num_rounds):
        round_start = time.time()
        print(f"\n[Round {r+1}/{num_rounds}]")

        # Step 1: Convert weights to NumPy (serialization overhead)
        t1 = time.time()
        numpy_weights = weights_to_numpy(global_model.state_dict())
        args_list = [
            (i, numpy_weights, client_indices[i], local_epochs, lr)
            for i in range(num_clients)
        ]
        t2 = time.time()
        serialization_times.append(t2 - t1)

        # Step 2: Run all client training in parallel using ProcessPoolExecutor
        # This replaces the sequential for-loop in serial_main.py
        t3 = time.time()
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            client_results = list(executor.map(train_client_worker, args_list))
        t4 = time.time()
        training_times.append(t4 - t3)

        # Step 3: Aggregate client weights on the server (sequential step)
        t5 = time.time()
        global_model = fedavg_numpy(global_model, client_results)
        t6 = time.time()
        aggregation_times.append(t6 - t5)

        round_time = time.time() - round_start
        round_times.append(round_time)

        acc = evaluate(global_model, test_data)
        print(f"  time: {round_time:.2f}s | accuracy: {acc:.4f}")
        print(f"  [overhead] serialization: {t2-t1:.3f}s | training: {t4-t3:.2f}s | aggregation: {t6-t5:.3f}s")

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Avg round:      {np.mean(round_times):.2f}s")
    print(f"Avg serialization: {np.mean(serialization_times):.3f}s")
    print(f"Avg training:      {np.mean(training_times):.2f}s")
    print(f"Avg aggregation:   {np.mean(aggregation_times):.3f}s")
    print(f"Final accuracy: {acc:.4f}")

    # Save all results including overhead breakdown
    results = {
        "mode": "parallel",
        "num_clients": num_clients,
        "num_workers": num_workers,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "total_time": total_time,
        "round_times": round_times,
        "serialization_times": serialization_times,
        "training_times": training_times,
        "aggregation_times": aggregation_times,
        "final_accuracy": acc
    }
    os.makedirs("results", exist_ok=True)
    filename = f"results/parallel_clients{num_clients}_workers{num_workers}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {filename}")

    return total_time, round_times


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    # Run experiments with different numbers of worker processes
    for workers in [2, 4, 8]:
        run_parallel(num_clients=50, num_rounds=3, num_workers=workers)