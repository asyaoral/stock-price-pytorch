import copy
import torch
import torch.nn as nn


def train_model(
    model,
    train_loader,
    val_X,
    val_y,
    epochs=100,
    lr=0.001,
    patience=5,
):
    """
    Trains a PyTorch model with early stopping on validation loss
    and restores the true best validation checkpoint via copy.deepcopy.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_weights = None
    p_counter = 0

    train_losses = []
    val_losses = []

    total_train_samples = len(train_loader.dataset)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for bx, by in train_loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(bx)

        epoch_train_loss = running_loss / total_train_samples
        train_losses.append(epoch_train_loss)

        model.eval()
        with torch.no_grad():
            val_pred = model(val_X)
            val_loss = criterion(val_pred, val_y).item()
            val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            p_counter = 0
        else:
            p_counter += 1
            if p_counter >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, train_losses, val_losses