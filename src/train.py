import torch
import torch.nn as nn


def train_model(model, X_train, y_train, epochs=100, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()

        predictions = model(X_train)
        loss = criterion(predictions, y_train)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return model, losses