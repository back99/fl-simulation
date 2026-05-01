import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def load_dataset():
    # Load MNIST dataset and apply standard normalization
    # Mean=0.1307 and Std=0.3081 are standard values for MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data  = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return train_data, test_data

def split_non_iid(dataset, num_clients, classes_per_client=2):
    # Split data across clients in a Non-IID way
    # Each client only gets data from 2 out of 10 digit classes
    # This simulates realistic FL where each device has different types of data
    labels = np.array(dataset.targets)

    # Group all data indices by their class label (0-9)
    class_indices = {c: np.where(labels == c)[0].tolist() for c in range(10)}
    for c in class_indices:
        np.random.shuffle(class_indices[c])  # Shuffle to randomize assignment

    client_indices = []
    for i in range(num_clients):
        # Assign 2 classes to each client, rotating across all 10 classes
        assigned_classes = [(i * classes_per_client + j) % 10 for j in range(classes_per_client)]
        indices = []
        for c in assigned_classes:
            # Give each client a portion of data from its assigned classes
            chunk = len(class_indices[c]) // (num_clients // 5 + 1)
            start = (i // 5) * chunk
            indices.extend(class_indices[c][start:start + chunk])
        client_indices.append(indices)

    return client_indices

def make_client_loaders(dataset, client_indices, batch_size=32):
    # Create a DataLoader for each client using only its assigned data indices
    return [
        DataLoader(Subset(dataset, idx), batch_size=batch_size, shuffle=True)
        for idx in client_indices
    ]
