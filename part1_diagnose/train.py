"""Train and evaluate a hands-vs-feet decoder on unseen subjects.

Usage:
  python train.py --channels all --seed 0
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn

from data import build_dataset, SUBJECTS
from model import EEGNet, count_params

# Electrode groups for the channel-subset run (Task 3).
# Frontal electrodes: the visual cue (top=fists, bottom=feet) would drive
# frontal/occipital gaze artefacts. We pick a broad frontal strip to test
# whether the model exploits this non-brain signal.
CHANNEL_GROUPS = {
    "all": None,
    "frontal": ["Fp1", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1",
                "Fz", "F2", "F4", "F6", "F8", "FC5", "FC3", "FC1",
                "FCz", "FC2", "FC4", "FC6"],
}


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)


def evaluate(model, X, y, bs=256):
    model.eval(); correct = 0
    with torch.no_grad():
        for i in range(0, len(y), bs):
            xb = torch.from_numpy(X[i:i + bs]); yb = torch.from_numpy(y[i:i + bs])
            correct += (model(xb).argmax(1) == yb).sum().item()
    return correct / len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", choices=list(CHANNEL_GROUPS), default="all")
    ap.add_argument("--subjects", type=int, nargs="+", default=SUBJECTS)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)

    X, y, groups, ch_names = build_dataset(args.subjects)
    picks = CHANNEL_GROUPS[args.channels]
    if picks is not None:
        if not picks:
            raise ValueError(f"Fill in CHANNEL_GROUPS['{args.channels}'] first.")
        X = X[:, [ch_names.index(c) for c in picks]]
    print(f"data: {X.shape}, subjects={len(np.unique(groups))}, "
          f"class balance={np.bincount(y)}")

    # Subject-independent evaluation: hold out the last 20% of *subjects* for test,
    # then use the last 20% of remaining subjects for validation.
    # BUG WAS: train_test_split on trial indices (stratify=y) — this randomly mixes
    # the same subjects across train/val/test, giving inflated within-subject accuracy.
    # FIX: split by subject group so no subject appears in more than one split.
    unique_subjects = np.unique(groups)
    n = len(unique_subjects)
    # Deterministic subject split: last 20% → test, next 20% → val, rest → train.
    n_te = max(1, int(np.ceil(n * 0.2)))
    n_va = max(1, int(np.ceil((n - n_te) * 0.2)))
    rng_subj = np.random.default_rng(args.seed)
    shuffled = rng_subj.permutation(unique_subjects)
    te_subjects = set(shuffled[:n_te])
    va_subjects = set(shuffled[n_te:n_te + n_va])
    idx = np.arange(len(y))
    te = idx[np.isin(groups, list(te_subjects))]
    va = idx[np.isin(groups, list(va_subjects))]
    tr = idx[~np.isin(groups, list(te_subjects | va_subjects))]
    print(f"split: train subjects={sorted(set(groups[tr]))}, "
          f"val subjects={sorted(set(groups[va]))}, "
          f"test subjects={sorted(set(groups[te]))}")

    # Per-channel z-scoring — TRAIN indices only; apply the same stats to val/test
    # to avoid leaking test distribution into the normalisation.
    mean = X[tr].mean(axis=(0, 2), keepdims=True)
    std = X[tr].std(axis=(0, 2), keepdims=True) + 1e-6
    X = ((X - mean) / std).astype(np.float32)

    n_ch, n_t = X.shape[1], X.shape[2]
    model = EEGNet(n_ch, n_t)
    print(f"EEGNet: {count_params(model)} params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(args.seed)

    best_acc, best_state, best_ep = -1.0, None, -1
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(tr)
        for i in range(0, len(order), args.bs):
            b = order[i:i + args.bs]
            xb, yb = torch.from_numpy(X[b]), torch.from_numpy(y[b])
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
        va_acc = evaluate(model, X[va], y[va])
        te_acc = evaluate(model, X[te], y[te])
        if va_acc > best_acc:   # keep the best checkpoint (val, NOT test)
            best_acc, best_ep = va_acc, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep:3d}  loss {loss.item():.3f}  val {va_acc:.3f}  test {te_acc:.3f}")

    model.load_state_dict(best_state)
    result = dict(channels=args.channels, seed=args.seed, best_epoch=best_ep,
                  test_acc=evaluate(model, X[te], y[te]), n_test=len(te))
    print(json.dumps(result))
    with open("results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
