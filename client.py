import copy
import torch.nn as nn
import torch.optim as optim
from model import SimpleCNN

def local_train(args):
    # Unpack arguments: client ID, global model weights, local data, training settings
    client_id, global_weights, dataloader, epochs, lr = args

    # Start from the global model sent by the server
    # Each client gets its own copy so training does not interfere with others
    model = SimpleCNN()
    model.load_state_dict(copy.deepcopy(global_weights))
    model.train()

    # Use SGD with momentum for local training
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Train on local data for the specified number of epochs
    # Raw data never leaves the client - only model weights are shared later
    for epoch in range(epochs):
        for images, labels in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    print(f"  [Client {client_id}] done")

    # Return only the updated model weights and dataset size
    # This is the core privacy principle of Federated Learning
    return client_id, model.state_dict(), len(dataloader.dataset)
