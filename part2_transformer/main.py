"""
EEG Transformer Sprint — entry point.
Run:  python main.py

This script:
  1. loads the cross-subject split (train subjects 1-8, test 9-10),
  2. trains the two baselines (EEGNet, vanilla transformer),
  3. trains YOUR model across a few seeds and reports best + worst,
  4. saves one figure (figure.png).

>>> YOUR JOB <<<
Implement ONE inductive bias in `YourModel` below (or replace it entirely).
State your hypothesis in the README. Do not quietly edit the vanilla
transformer in models.py — the contrast is the point.
"""
import argparse
import numpy as np
import torch

from data import load_data, get_channel_positions
from models import EEGNet, VanillaEEGTransformer, count_params
from train import train_model, set_seed


# =========================================================================== #
#  YOUR MODEL  — spatial channel tokens + AP-initialised positional encoding.
#
#  Hypothesis: the bilateral fists-vs-feet task is anterior-posterior (AP), not
#  left-right. A transformer that tokenises by CHANNEL (one token per electrode)
#  and whose positional embedding is seeded from the real AP coordinate of each
#  electrode has an explicit spatial prior aligned to the discriminative axis —
#  without hard-coding any left/right symmetry.
#
#  The ONE inductive bias: channel tokens whose learned positional embeddings are
#  initialised from real 3-D electrode coordinates (specifically the Y/AP axis).
#  Attention can therefore learn "attend more to frontal-vs-central differences"
#  from a warm start, rather than discovering electrode layout from scratch.
# =========================================================================== #
import torch.nn as nn


class YourModel(nn.Module):
    """Spatial channel-token transformer with AP-initialised positional encoding.

    Each of the C electrodes becomes one token by summarising its T-sample time
    series through a depthwise-then-pointwise Conv1d (i.e., a lightweight
    temporal summary).  The C positional embeddings are initialised from the
    real anterior-posterior (Y) coordinate of each electrode so the transformer
    starts 'knowing' which channels sit at the front vs. back of the scalp.

    This gives the model two advantages over the vanilla per-timestep baseline:
      1. The sequence length is C=64 instead of T=481 → O(C²) attention, far
         less overfit surface for 8 training subjects.
      2. The spatial embedding warm-start biases attention toward AP contrasts
         (Fz/FCz frontal vs. Cz/CPz central-parietal) without hardcoding them.
    """

    def __init__(self, n_chan, n_time, n_classes=2,
                 d_model=64, n_heads=4, depth=3, p=0.3,
                 chan_positions=None):
        super().__init__()
        # ── temporal summary per channel ──────────────────────────────────── #
        # Depthwise conv (groups=n_chan) compresses T → T//8 time-steps,
        # then a pointwise conv projects to d_model. Each channel stays
        # independent during the temporal summary stage.
        self.dw_conv = nn.Conv1d(n_chan, n_chan, kernel_size=16,
                                 stride=8, padding=4, groups=n_chan, bias=False)
        self.bn = nn.BatchNorm1d(n_chan)
        self.act = nn.ELU()
        n_t_reduced = (n_time + 2 * 4 - 16) // 8 + 1   # after dw_conv
        # Per-channel projection: each channel's compressed time vector → d_model.
        self.ch_proj = nn.Linear(n_t_reduced, d_model)

        # ── CLS token + positional embedding (C channel positions) ────────── #
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        # Positional embeddings: one per channel + one for CLS.
        # Initialised from real electrode Y (anterior-posterior) coordinate.
        pos_init = torch.zeros(1, n_chan + 1, d_model)
        if chan_positions is not None:               # shape (C, 3): x,y,z
            ap = torch.from_numpy(chan_positions[:, 1])   # Y = anterior-posterior
            ap = (ap - ap.mean()) / (ap.std() + 1e-6)     # normalise
            # Broadcast AP scalar into d_model dims as a warm-start bias.
            pos_init[0, 1:, :] = ap.unsqueeze(1).expand(-1, d_model) * 0.02
        self.pos = nn.Parameter(pos_init)
        nn.init.trunc_normal_(self.cls, std=0.02)

        # ── transformer encoder ───────────────────────────────────────────── #
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2,
            dropout=p, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, return_embed=False):   # x: (B, C, T)
        B, C, T = x.shape
        # Temporal summary: compress each channel's time series.
        h = self.act(self.bn(self.dw_conv(x)))   # (B, C, n_t_reduced)
        # Project each channel to d_model → (B, C, d_model) as channel tokens.
        tok = self.ch_proj(h)                    # (B, C, d_model)
        # Prepend CLS token and add positional embedding.
        cls = self.cls.expand(B, -1, -1)
        z = torch.cat([cls, tok], dim=1) + self.pos   # (B, C+1, d_model)
        z = self.encoder(z)
        embed = self.norm(z[:, 0])               # CLS output
        logits = self.head(embed)
        return (logits, embed) if return_embed else logits


# =========================================================================== #


def run_across_seeds(build_fn, Xtr, ytr, Xte, yte, seeds, device, **kw):
    accs = []
    for s in seeds:
        set_seed(s)
        model = build_fn()
        best, _ = train_model(model, Xtr, ytr, Xte, yte, device=device, **kw)
        accs.append(best)
        print(f"  seed {s}: best test acc {best:.3f}")
    return np.array(accs)


def save_figure(model, Xte, yte, device, path="figure.png"):
    """t-SNE of the CLS embeddings on the test subjects, coloured by class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(Xte).to(device), return_embed=True)
        emb = out[1].cpu().numpy() if isinstance(out, tuple) else None
    if emb is None:
        print("  (your model did not return an embedding; skipping figure)")
        return
    n = len(emb)
    emb2 = TSNE(n_components=2, perplexity=min(30, max(5, n // 4)),
                init="pca", random_state=0).fit_transform(emb)
    plt.figure(figsize=(5, 4))
    for c, name in [(0, "fists"), (1, "feet")]:
        m = yte == c
        plt.scatter(emb2[m, 0], emb2[m, 1], s=18, alpha=0.7, label=name)
    plt.legend(); plt.title("Test-subject embeddings (your model)")
    plt.tight_layout(); plt.savefig(path, dpi=130)
    print(f"  saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print("\nloading data (first run downloads ~50 MB)...")
    Xtr, ytr, Xte, yte = load_data()
    C, T = Xtr.shape[1], Xtr.shape[2]
    print(f"train {Xtr.shape}  test {Xte.shape}  "
          f"(class balance train {ytr.mean():.2f} / test {yte.mean():.2f})")

    # Load electrode positions once for YourModel's AP positional encoding.
    print("loading channel positions for AP positional encoding...")
    chan_pos = get_channel_positions()   # (C, 3)

    print("\n=== Baseline 1: EEGNet (conv priors) ===")
    set_seed(0)
    net = EEGNet(C, T)
    print(f"params: {count_params(net):,}")
    best, _ = train_model(net, Xtr, ytr, Xte, yte, epochs=args.epochs,
                          device=device, verbose=True)
    print(f"EEGNet best test acc: {best:.3f}")

    print("\n=== Baseline 2: vanilla transformer (no EEG prior) ===")
    set_seed(0)
    vit = VanillaEEGTransformer(C, T)
    print(f"params: {count_params(vit):,}")
    accs_v = run_across_seeds(lambda: VanillaEEGTransformer(C, T),
                              Xtr, ytr, Xte, yte, args.seeds, device,
                              epochs=args.epochs)
    print(f"vanilla transformer: best {accs_v.max():.3f}  "
          f"worst {accs_v.min():.3f}  mean {accs_v.mean():.3f}")

    print("\n=== YOUR MODEL ===")
    set_seed(0)
    ym = YourModel(C, T, chan_positions=chan_pos)
    print(f"params: {count_params(ym):,}")
    accs = run_across_seeds(lambda: YourModel(C, T, chan_positions=chan_pos),
                            Xtr, ytr, Xte, yte, args.seeds, device,
                            epochs=args.epochs)
    print(f"your model: best {accs.max():.3f}  worst {accs.min():.3f}  "
          f"mean {accs.mean():.3f}  std {accs.std():.3f}")

    print("\nsaving figure...")
    set_seed(0)
    ym = YourModel(C, T, chan_positions=chan_pos)
    train_model(ym, Xtr, ytr, Xte, yte, epochs=args.epochs, device=device)
    save_figure(ym, Xte, yte, device)

    print("\ndone. Report best AND worst in the README — no cherry-picking.")


if __name__ == "__main__":
    main()
