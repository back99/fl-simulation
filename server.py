import copy
import torch
from torch.utils.data import DataLoader

def fedavg(global_model, client_results):
    # FedAvg algorithm: aggregate client model weights using weighted average
    # Clients with more local data have a stronger influence on the global model
    total_data = sum(size for _, _, size in client_results)

    # Start with a copy of the first client's weights as the base
    avg_weights = copy.deepcopy(client_results[0][1])

    # Compute weighted average for each parameter layer
    for key in avg_weights:
        avg_weights[key] = sum(
            weights[key] * (size / total_data)
            for _, weights, size in client_results
        )

    # Update the global model with the averaged weights
    global_model.load_state_dict(avg_weights)
    return global_model

def evaluate(model, test_data, batch_size=256):
    # Evaluate the global model accuracy on the test set
    # This is done only on the server side after each aggregation round
    loader = DataLoader(test_data, batch_size=batch_size)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total  # Return accuracy as a fraction (0.0 to 1.0)
