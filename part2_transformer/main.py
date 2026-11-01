"""
EEG Transformer Sprint — entry point.
Run:  python main.py

This script:
  1. Loads the cross-subject split (train subjects 1-8, test subjects 9-10).
  2. Trains EEGNet and VanillaEEGTransformer as fixed baselines.
  3. Trains YourModel across multiple seeds and reports best, worst, and mean.
  4. Saves figure.png (t-SNE of CLS embeddings on held-out test subjects).

Inductive bias implemented in YourModel (below):
  Spatial channel tokens + AP-coordinate-initialised positional embeddings.
  See class docstring and part2_transformer/README.md for full hypothesis.

NOTE: models.py is intentionally left unchanged.  The whole point of the
experiment is the CONTRAST between vanilla attention (no EEG prior) and
YourModel (one targeted structural inductive bias).
"""
import argparse
import numpy as np
import torch
import torch.nn as nn

from data import load_data, get_channel_positions
from models import EEGNet, VanillaEEGTransformer, count_params
from train import train_model, set_seed


# ============================================================================
# YourModel — spatial channel tokens + AP-initialised positional encoding.
#
# Hypothesis (written before coding):
#   Per-timestep attention on 64-channel EEG wastes capacity learning that
#   adjacent samples are correlated — EEGNet's conv stem encodes this for
#   free.  For bilateral fists-vs-feet the discriminative axis is spatial
#   and anterior-posterior (AP), not temporal.  Tokenising by electrode
#   (one token per channel) collapses attention from O(T²) to O(C²) and,
#   seeded with real AP electrode coordinates, gives the model the correct
#   spatial prior from epoch one — without being told which channels to weight.
#
# The ONE inductive bias:
#   Channel tokens whose learned positional embeddings are warm-started from
#   the real Y (anterior-posterior) coordinate of each electrode.
# ============================================================================


class YourModel(nn.Module):
    """Spatial channel-token transformer with AP-initialised positional encoding.

    Architecture
    ------------
    1. Temporal summary (per channel, independently):
       Depthwise Conv1d (kernel=16, stride=8, groups=C) compresses each
       electrode's T-sample time series to ~60 values without mixing channels.
       A Linear projects each channel's summary vector to d_model.
       → Sequence of C spatial tokens, each of dimension d_model.

    2. CLS token + positional embedding:
       A learnable CLS token is prepended.  Positional embeddings for the C
       channel positions are warm-started from the real anterior-posterior (Y)
       electrode coordinate (from get_channel_positions()).
       Frontal electrodes (Fp1, Fz, FCz) and parietal electrodes (Cz, CPz, Pz)
       start with distinct positional vectors, biasing attention toward AP
       contrasts without hard-coding which electrodes to weight.

    3. Transformer encoder (depth=3, n_heads=4):
       Self-attention across the C+1 spatial tokens.
       Attention matrix: (C+1)² = 65² = 4,225 entries/head.
       Compare: VanillaEEGTransformer has (T+1)² = 482² = 232,324 entries/head.
       → 55× smaller attention surface for the same number of training trials.

    4. Classification head:
       LayerNorm(CLS output) → Linear → n_classes logits.
    """

    def __init__(self, n_chan: int, n_time: int, n_classes: int = 2,
                 d_model: int = 64, n_heads: int = 4, depth: int = 3,
                 p: float = 0.3, chan_positions=None):
        super().__init__()

        # ── 1. Temporal summary per channel ──────────────────────────────── #
        self.dw_conv = nn.Conv1d(
            n_chan, n_chan, kernel_size=16, stride=8, padding=4,
            groups=n_chan, bias=False)
        self.bn  = nn.BatchNorm1d(n_chan)
        self.act = nn.ELU()
        n_t_reduced = (n_time + 2 * 4 - 16) // 8 + 1   # output length of dw_conv
        self.ch_proj = nn.Linear(n_t_reduced, d_model)  # per-channel linear projection

        # ── 2. CLS token + positional embedding ──────────────────────────── #
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        # One positional embedding per channel position (index 0 = CLS token).
        pos_init = torch.zeros(1, n_chan + 1, d_model)
        if chan_positions is not None:          # shape (C, 3): x (L-R), y (AP), z (sup-inf)
            ap = torch.from_numpy(chan_positions[:, 1])   # Y = anterior-posterior axis
            ap = (ap - ap.mean()) / (ap.std() + 1e-6)    # z-score normalise
            # Broadcast AP scalar into d_model dims as a warm-start bias (std ≈ 0.02).
            pos_init[0, 1:, :] = ap.unsqueeze(1).expand(-1, d_model) * 0.02
        self.pos = nn.Parameter(pos_init)
        nn.init.trunc_normal_(self.cls, std=0.02)

        # ── 3. Transformer encoder ────────────────────────────────────────── #
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2,
            dropout=p, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d_model)

        # ── 4. Classification head ────────────────────────────────────────── #
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor, return_embed: bool = False):
        """
        Parameters
        ----------
        x            : (B, C, T) — batch of EEG trials.
        return_embed : if True, also return the CLS embedding for visualisation.

        Returns
        -------
        logits       : (B, n_classes)
        embed        : (B, d_model)  — only returned when return_embed=True.
        """
        B, C, T = x.shape

        # Temporal summary: compress each channel's time series independently.
        h   = self.act(self.bn(self.dw_conv(x)))   # (B, C, n_t_reduced)
        tok = self.ch_proj(h)                       # (B, C, d_model) — channel tokens

        # Prepend CLS, add positional embeddings.
        cls = self.cls.expand(B, -1, -1)
        z   = torch.cat([cls, tok], dim=1) + self.pos   # (B, C+1, d_model)

        # Transformer: self-attention across the C+1 spatial tokens.
        z     = self.encoder(z)
        embed = self.norm(z[:, 0])   # CLS token output
        logits = self.head(embed)

        return (logits, embed) if return_embed else logits


# ============================================================================


def run_across_seeds(build_fn, Xtr, ytr, Xte, yte, seeds, device, **kw):
    """Train a fresh model for each seed, collect and return test accuracies."""
    accs = []
    for s in seeds:
        set_seed(s)
        model = build_fn()
        best, _ = train_model(model, Xtr, ytr, Xte, yte, device=device, **kw)
        accs.append(best)
        print(f"    seed {s}: best test acc {best:.3f}")
    return np.array(accs)


def save_figure(model, Xte: np.ndarray, yte: np.ndarray,
                device: str, path: str = "figure.png") -> None:
    """t-SNE of the CLS embeddings on held-out test subjects, coloured by class.

    Fists (class 0) and feet (class 1) are plotted as separate scatter clouds.
    Cluster separation indicates cross-subject transfer of the learned spatial
    representation; overlap is an honest negative result on limited data.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(Xte).to(device), return_embed=True)
        emb = out[1].cpu().numpy() if isinstance(out, tuple) else None

    if emb is None:
        print("  (model did not return an embedding; skipping figure)")
        return

    n    = len(emb)
    perp = min(30, max(5, n // 4))
    emb2 = TSNE(n_components=2, perplexity=perp,
                init="pca", random_state=0).fit_transform(emb)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    colors  = {0: "#e05c5c", 1: "#4c7be0"}
    labels  = {0: "fists (T1)", 1: "feet (T2)"}
    for c in [0, 1]:
        m = yte == c
        ax.scatter(emb2[m, 0], emb2[m, 1], s=20, alpha=0.75,
                   color=colors[c], label=labels[c])
    ax.legend(fontsize=9)
    ax.set_title("t-SNE of test-subject CLS embeddings\n(YourModel, held-out subjects 9-10)",
                 fontsize=10)
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"  saved {path}")


def main():
    ap = argparse.ArgumentParser(
        description="EEG Transformer Sprint — run baselines and YourModel.")
    ap.add_argument("--epochs", type=int, default=40,
                    help="Training epochs per model/seed (default: 40).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="Random seeds to average over (default: 0 1 2).")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}\n")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("Loading data (first run downloads ~50 MB)...")
    Xtr, ytr, Xte, yte = load_data()
    C, T = Xtr.shape[1], Xtr.shape[2]
    print(f"  train {Xtr.shape}  test {Xte.shape}")
    print(f"  class balance — train: {ytr.mean():.2f}  test: {yte.mean():.2f}\n")

    print("Loading 3D electrode positions for AP positional encoding...")
    chan_pos = get_channel_positions()   # (C, 3)

    # ------------------------------------------------------------------
    # Baseline 1 — EEGNet
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Baseline 1: EEGNet (conv priors)")
    print("=" * 60)
    set_seed(0)
    net = EEGNet(C, T)
    print(f"  params: {count_params(net):,}")
    best_net, _ = train_model(net, Xtr, ytr, Xte, yte,
                              epochs=args.epochs, device=device, verbose=True)
    print(f"  EEGNet best test acc: {best_net:.3f}\n")

    # ------------------------------------------------------------------
    # Baseline 2 — Vanilla Transformer
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Baseline 2: VanillaEEGTransformer (no EEG prior)")
    print("=" * 60)
    set_seed(0)
    vit = VanillaEEGTransformer(C, T)
    print(f"  params: {count_params(vit):,}")
    print(f"  attention entries/head: {(T + 1) ** 2:,}  (T+1={T+1})")
    accs_v = run_across_seeds(
        lambda: VanillaEEGTransformer(C, T),
        Xtr, ytr, Xte, yte, args.seeds, device, epochs=args.epochs)
    print(f"\n  vanilla transformer — "
          f"best {accs_v.max():.3f}  "
          f"worst {accs_v.min():.3f}  "
          f"mean {accs_v.mean():.3f}  "
          f"std {accs_v.std():.3f}\n")

    # ------------------------------------------------------------------
    # YourModel — channel tokens + AP positional warm-start
    # ------------------------------------------------------------------
    print("=" * 60)
    print("YourModel: channel tokens + AP positional encoding")
    print("=" * 60)
    set_seed(0)
    ym = YourModel(C, T, chan_positions=chan_pos)
    print(f"  params: {count_params(ym):,}")
    print(f"  attention entries/head: {(C + 1) ** 2:,}  (C+1={C+1})")
    print(f"  attention reduction vs. vanilla: {(T + 1) ** 2 / (C + 1) ** 2:.1f}x\n")
    accs = run_across_seeds(
        lambda: YourModel(C, T, chan_positions=chan_pos),
        Xtr, ytr, Xte, yte, args.seeds, device, epochs=args.epochs)
    print(f"\n  YourModel — "
          f"best {accs.max():.3f}  "
          f"worst {accs.min():.3f}  "
          f"mean {accs.mean():.3f}  "
          f"std {accs.std():.3f}\n")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  EEGNet              : {best_net:.3f}")
    print(f"  VanillaTransformer  : {accs_v.mean():.3f} ± {accs_v.std():.3f}"
          f"  (range {accs_v.max() - accs_v.min():.3f})")
    print(f"  YourModel           : {accs.mean():.3f} ± {accs.std():.3f}"
          f"  (range {accs.max() - accs.min():.3f})")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    print("\nGenerating figure.png ...")
    set_seed(0)
    ym_fig = YourModel(C, T, chan_positions=chan_pos)
    train_model(ym_fig, Xtr, ytr, Xte, yte, epochs=args.epochs, device=device)
    save_figure(ym_fig, Xte, yte, device)

    print("\nDone. Report best AND worst in the README — no cherry-picking.\n"
          "See part2_transformer/README.md for the full write-up.")


if __name__ == "__main__":
    main()
