import copy
import torch.nn as nn
import torch.optim as optim
from model import SimpleCNN

def local_train(args):
    client_id, global_weights, dataloader, epochs, lr = args

    model = SimpleCNN()
    model.load_state_dict(copy.deepcopy(global_weights))
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for images, labels in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    print(f"  [Client {client_id}] done")
    return client_id, model.state_dict(), len(dataloader.dataset)
