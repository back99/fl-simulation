import os
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
    return {k: v.numpy() for k, v in state_dict.items()}


def numpy_to_weights(numpy_dict):
    return {k: torch.from_numpy(v) for k, v in numpy_dict.items()}


def train_client_worker(args):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    client_id, numpy_weights, indices, epochs, lr = args

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST('./data', train=True, download=False, transform=transform)
    loader = DataLoader(Subset(train_data, indices), batch_size=32, shuffle=True)

    model = SimpleCNN()
    model.load_state_dict(numpy_to_weights(numpy_weights))
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for images, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    print(f"  [Client {client_id}] done", flush=True)
    return client_id, weights_to_numpy(model.state_dict()), len(indices)


def fedavg_numpy(global_model, client_results):
    total_data = sum(size for _, _, size in client_results)
    avg_weights = {}
    keys = client_results[0][1].keys()
    for key in keys:
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
    acc = 0.0

    for r in range(num_rounds):
        round_start = time.time()
        print(f"\n[Round {r+1}/{num_rounds}]")

        numpy_weights = weights_to_numpy(global_model.state_dict())
        args_list = [
            (i, numpy_weights, client_indices[i], local_epochs, lr)
            for i in range(num_clients)
        ]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            client_results = list(executor.map(train_client_worker, args_list))

        global_model = fedavg_numpy(global_model, client_results)

        round_time = time.time() - round_start
        round_times.append(round_time)

        acc = evaluate(global_model, test_data)
        print(f"  time: {round_time:.2f}s | accuracy: {acc:.4f}")

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Avg round:  {np.mean(round_times):.2f}s")
    print(f"Final accuracy: {acc:.4f}")

    results = {
        "mode": "parallel",
        "num_clients": num_clients,
        "num_workers": num_workers,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "total_time": total_time,
        "round_times": round_times,
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
    for workers in [2, 4, 8]:
        run_parallel(num_clients=50, num_rounds=3, num_workers=workers)
