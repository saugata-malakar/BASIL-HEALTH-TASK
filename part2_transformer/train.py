"""
Shared training / evaluation harness. Keep runs short — the point is the idea,
not the hyperparameters.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def make_loaders(Xtr, ytr, Xte, yte, batch=64):
    tr = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    te = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
    return (DataLoader(tr, batch_size=batch, shuffle=True),
            DataLoader(te, batch_size=256))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / total


def train_model(model, Xtr, ytr, Xte, yte, *, epochs=40, lr=1e-3,
                weight_decay=1e-3, device="cpu", verbose=False):
    """Train and return (best_test_acc, final_test_acc). Early-stops on nothing;
    we report the peak so a short budget is forgiving."""
    model = model.to(device)
    tr, te = make_loaders(Xtr, ytr, Xte, yte)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        acc = evaluate(model, te, device)
        best = max(best, acc)
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:2d}  test_acc {acc:.3f}  (best {best:.3f})")
    return best, acc


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
