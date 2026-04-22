import time
import json
import os
import numpy as np
import torch
from model import SimpleCNN
from data import load_dataset, split_non_iid, make_client_loaders
from client import local_train
from server import fedavg, evaluate

def run_serial(num_clients=10, num_rounds=5, local_epochs=1, lr=0.01):
    print(f"\n{'='*50}")
    print(f"[Serial FL] clients={num_clients}, rounds={num_rounds}")
    print(f"{'='*50}")

    train_data, test_data = load_dataset()
    client_indices = split_non_iid(train_data, num_clients)
    client_loaders = make_client_loaders(train_data, client_indices)
    global_model = SimpleCNN()

    total_start = time.time()
    round_times = []
    acc = 0.0

    for r in range(num_rounds):
        round_start = time.time()
        print(f"\n[Round {r+1}/{num_rounds}]")

        client_results = []
        for i, loader in enumerate(client_loaders):
            args = (i, global_model.state_dict(), loader, local_epochs, lr)
            result = local_train(args)
            client_results.append(result)

        global_model = fedavg(global_model, client_results)

        round_time = time.time() - round_start
        round_times.append(round_time)

        acc = evaluate(global_model, test_data)
        print(f"  time: {round_time:.2f}s | accuracy: {acc:.4f}")

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Avg round:  {np.mean(round_times):.2f}s")
    print(f"Final accuracy: {acc:.4f}")

    results = {
        "mode": "serial",
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "total_time": total_time,
        "round_times": round_times,
        "final_accuracy": acc
    }
    os.makedirs("results", exist_ok=True)
    filename = f"results/serial_clients{num_clients}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {filename}")

    return total_time, round_times

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    for n in [10, 20, 50]:
        run_serial(num_clients=n, num_rounds=3)
