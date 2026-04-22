import copy
import torch
from torch.utils.data import DataLoader

def fedavg(global_model, client_results):
    total_data = sum(size for _, _, size in client_results)
    avg_weights = copy.deepcopy(client_results[0][1])
    for key in avg_weights:
        avg_weights[key] = sum(
            weights[key] * (size / total_data)
            for _, weights, size in client_results
        )
    global_model.load_state_dict(avg_weights)
    return global_model

def evaluate(model, test_data, batch_size=256):
    loader = DataLoader(test_data, batch_size=batch_size)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total
