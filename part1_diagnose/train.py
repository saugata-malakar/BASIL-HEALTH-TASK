"""Train and evaluate a hands-vs-feet decoder on unseen subjects.

Usage:
  python train.py --channels all --seed 0
  python train.py --channels frontal --seed 0
  python train.py --channels frontal --seed 0 --l_freq 8.0  # EOG ablation
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn

from data import build_dataset, SUBJECTS
from model import EEGNet, count_params

# ---------------------------------------------------------------------------
# Electrode groups for the channel-subset ablation (Task 3).
#
# "frontal": pure prefrontal + frontal strip (17 channels), strictly anterior
#   to the premotor frontocentral (FC*) belt.  FC1-FC6 and FCz are excluded
#   because FCz overlies the Supplementary Motor Area (SMA) and FC3/FC4
#   overlie dorsal Premotor Cortex (PMd), which carry genuine motor planning
#   signals. Including them would conflate motor cortex volume conduction with
#   the ocular EOG artefact we are isolating.
# ---------------------------------------------------------------------------
CHANNEL_GROUPS = {
    "all": None,
    "frontal": [
        "Fp1", "Fpz", "Fp2",
        "AF7", "AF3", "AFz", "AF4", "AF8",
        "F7",  "F5",  "F3",  "F1",  "Fz",  "F2",  "F4",  "F6",  "F8",
    ],
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray, bs: int = 256) -> float:
    """Return accuracy on (X, y) without computing gradients."""
    model.eval()
    correct = 0
    for i in range(0, len(y), bs):
        xb = torch.from_numpy(X[i:i + bs])
        yb = torch.from_numpy(y[i:i + bs])
        correct += (model(xb).argmax(1) == yb).sum().item()
    return correct / len(y)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cross-subject EEGNet evaluation on hands-vs-feet motor imagery.")
    ap.add_argument("--channels", choices=list(CHANNEL_GROUPS), default="all",
                    help="Electrode subset (default: all 64 channels).")
    ap.add_argument("--subjects", type=int, nargs="+", default=SUBJECTS,
                    help="Subjects to include (default: 1-10).")
    ap.add_argument("--epochs",   type=int,   default=30,
                    help="Training epochs (default: 30).")
    ap.add_argument("--lr",       type=float, default=1e-3,
                    help="AdamW learning rate (default: 1e-3).")
    ap.add_argument("--bs",       type=int,   default=64,
                    help="Mini-batch size (default: 64).")
    ap.add_argument("--seed",     type=int,   default=0,
                    help="Random seed (default: 0).")
    # Frequency ablation flags — used for Task 3 EOG double-dissociation.
    ap.add_argument("--l_freq",   type=float, default=None,
                    help="Override low-pass frequency (default: inherits data.py L_FREQ=1.0 Hz).")
    ap.add_argument("--h_freq",   type=float, default=None,
                    help="Override high-pass frequency (default: inherits data.py H_FREQ=40.0 Hz).")
    args = ap.parse_args()
    set_seed(args.seed)

    # ------------------------------------------------------------------
    # Data loading — support optional frequency override for ablation runs.
    # When --l_freq / --h_freq are provided we re-filter inside build_dataset
    # by monkey-patching the module constants so erd.py's load_raw path is
    # reused unchanged.  This keeps the ablation transparent.
    # ------------------------------------------------------------------
    import data as _data
    l_freq_used = args.l_freq if args.l_freq is not None else _data.L_FREQ
    h_freq_used = args.h_freq if args.h_freq is not None else _data.H_FREQ
    if args.l_freq is not None or args.h_freq is not None:
        # Temporarily patch module-level constants used inside epoch_subject().
        _orig_l, _orig_h = _data.L_FREQ, _data.H_FREQ
        _data.L_FREQ, _data.H_FREQ = l_freq_used, h_freq_used
        print(f"[ABLATION] Re-filtering with {l_freq_used}–{h_freq_used} Hz "
              f"(default was {_orig_l}–{_orig_h} Hz)")

    X, y, groups, ch_names = build_dataset(args.subjects)

    if args.l_freq is not None or args.h_freq is not None:
        _data.L_FREQ, _data.H_FREQ = _orig_l, _orig_h  # restore

    # ------------------------------------------------------------------
    # Channel selection.
    # ------------------------------------------------------------------
    picks = CHANNEL_GROUPS[args.channels]
    if picks is not None:
        idx_ch = [ch_names.index(c) for c in picks]
        X = X[:, idx_ch]
    print(f"data shape : {X.shape}  |  subjects: {len(np.unique(groups))}"
          f"  |  class balance: {np.bincount(y)}")

    # ------------------------------------------------------------------
    # Subject-independent split.
    #
    # BUG (original): train_test_split(idx, stratify=y) scattered individual
    #   subjects' trials randomly across train/val/test, creating massive
    #   within-subject memorisation leakage and inflating accuracy.
    # FIX: Group split — each subject belongs exclusively to one partition.
    #   Seed-controlled shuffle → test (last 20%), val (next 20%), train (rest).
    # ------------------------------------------------------------------
    unique_subj = np.unique(groups)
    n = len(unique_subj)
    n_te = max(1, int(np.ceil(n * 0.2)))
    n_va = max(1, int(np.ceil((n - n_te) * 0.2)))
    rng_subj = np.random.default_rng(args.seed)
    shuffled  = rng_subj.permutation(unique_subj)
    te_subj   = set(shuffled[:n_te])
    va_subj   = set(shuffled[n_te:n_te + n_va])
    idx = np.arange(len(y))
    te = idx[np.isin(groups, list(te_subj))]
    va = idx[np.isin(groups, list(va_subj))]
    tr = idx[~np.isin(groups, list(te_subj | va_subj))]
    print(f"split      : train={sorted(set(groups[tr]))}  "
          f"val={sorted(set(groups[va]))}  test={sorted(set(groups[te]))}")

    # ------------------------------------------------------------------
    # Per-channel z-scoring on TRAIN statistics only.
    #
    # BUG (original): global z-score before splitting leaked test subjects'
    #   amplitude scale into normalisation.
    # FIX: compute mean/std exclusively on X[tr], broadcast to val/test.
    # ------------------------------------------------------------------
    mean = X[tr].mean(axis=(0, 2), keepdims=True)
    std  = X[tr].std(axis=(0, 2),  keepdims=True) + 1e-6
    X    = ((X - mean) / std).astype(np.float32)

    # ------------------------------------------------------------------
    # Model + optimiser.
    # ------------------------------------------------------------------
    n_ch, n_t = X.shape[1], X.shape[2]
    model = EEGNet(n_ch, n_t)
    print(f"EEGNet     : {count_params(model):,} params  |  input ({n_ch} ch × {n_t} time-steps)")

    opt     = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    rng     = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # Training loop.
    #
    # BUG 4 (original): if te_acc > best_acc  → direct test-set peeking.
    # FIX  : if va_acc > best_acc             → val-set checkpoint selection.
    #
    # BUG 5 (original): best_state = model.state_dict()
    #   PyTorch state_dict() returns shallow tensor references sharing memory
    #   with model.parameters().  Subsequent opt.step() mutates best_state
    #   in-place → load_state_dict at the end silently loads epoch 29.
    # FIX  : {k: v.clone() for k, v in model.state_dict().items()}
    # ------------------------------------------------------------------
    best_acc, best_state, best_ep = -1.0, None, -1
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(tr)
        for i in range(0, len(order), args.bs):
            b = order[i:i + args.bs]
            xb, yb = torch.from_numpy(X[b]), torch.from_numpy(y[b])
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        va_acc = evaluate(model, X[va], y[va])
        te_acc = evaluate(model, X[te], y[te])

        # Checkpoint on validation (never on test).
        if va_acc > best_acc:
            best_acc, best_ep = va_acc, ep
            # Deep-clone: break the shared-memory alias with model.parameters().
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"  ep {ep:3d}  loss {loss.item():.4f}  val {va_acc:.4f}  test {te_acc:.4f}"
              + (" *" if ep == best_ep else ""))

    # ------------------------------------------------------------------
    # Final evaluation on best checkpoint.
    # ------------------------------------------------------------------
    model.load_state_dict(best_state)
    final_te_acc = evaluate(model, X[te], y[te])
    result = dict(
        channels  = args.channels,
        seed      = args.seed,
        best_epoch= best_ep,
        test_acc  = final_te_acc,
        n_test    = len(te),
        l_freq    = l_freq_used,
        h_freq    = h_freq_used,
    )
    print("\n" + json.dumps(result))
    with open("results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
