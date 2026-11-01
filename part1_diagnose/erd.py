"""Event-related desynchronisation (ERD) for hands-vs-feet motor imagery.

Formula:
    ERD% = (P(t) - P_baseline) / P_baseline * 100

where P(t) is the band-limited instantaneous power (squared Hilbert envelope)
averaged across trials.  Negative values = desynchronisation (ERD).
Positive values = synchronisation (ERS).

BUG 6 FIX: original BASELINE = (0.0, 0.5) was INSIDE the motor imagery window
(t=0 is cue onset; imagery begins at t>=0).  Dividing by active ERD power
produces compressed curves near zero and can invert polarity.
CORRECT BASELINE: (-1.5, -0.5) — 1.0 s of pre-cue rest, ending 500 ms before
the cue to avoid any anticipatory Bereitschaftspotential (readiness potential).

Usage:
    python erd.py --band mu              # saves erd_mu.png
    python erd.py --band beta            # saves erd_beta.png
    python erd.py --subjects 1 2 3 --band mu
"""
import argparse
import numpy as np
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_raw, get_events, EVENT_ID, TMIN, TMAX, SUBJECTS

# Frequency bands for ERD analysis.
BANDS = {
    "mu":   (8.0,  13.0),   # mu rhythm: contralateral sensorimotor cortex
    "beta": (13.0, 30.0),   # beta rhythm: foot imagery dominant (Pfurtscheller effect)
}

# BUG 6 FIX: pre-cue rest baseline, NOT post-cue imagery onset.
BASELINE = (-1.5, -0.5)

# Central electrodes: lateral (C3/C4) vs. medial vertex (Cz).
PICKS = ["C3", "Cz", "C4"]


def band_power_epochs(raw, events, band: tuple):
    """Band-pass the continuous signal then epoch the squared Hilbert envelope.

    Filtering the raw (continuous) signal first avoids Gibbs-ringing edge
    effects that arise when band-passing already-epoched segments.
    """
    r = raw.copy().filter(*band, fir_design="firwin", verbose="error")
    r.apply_hilbert(envelope=True, verbose="error")
    epochs = mne.Epochs(
        r, events, EVENT_ID, TMIN, TMAX,
        baseline=None, preload=True, verbose="error")
    return epochs.apply_function(np.square)  # envelope → instantaneous power


def erd_percent(epochs, cond: str):
    """Compute ERD% time course for condition `cond` relative to pre-cue baseline."""
    p = epochs[cond].get_data(copy=True).mean(axis=0)   # (n_ch, n_times)
    t = epochs.times
    b = (t >= BASELINE[0]) & (t < BASELINE[1])
    ref = p[:, b].mean(axis=1, keepdims=True)            # (n_ch, 1)
    erd = (p - ref) / (ref + 1e-12) * 100.0             # avoid division by zero
    return erd, t


def main():
    ap = argparse.ArgumentParser(
        description="Compute and plot ERD/ERS for hands-vs-feet motor imagery.")
    ap.add_argument("--subjects", type=int, nargs="+", default=SUBJECTS,
                    help="Subject IDs to average over (default: all 10).")
    ap.add_argument("--band", choices=list(BANDS), default="mu",
                    help="Frequency band: 'mu' (8-13 Hz) or 'beta' (13-30 Hz).")
    args = ap.parse_args()

    band   = BANDS[args.band]
    curves = {c: [] for c in EVENT_ID}

    for s in args.subjects:
        raw    = load_raw(s)
        events = get_events(raw)
        ep     = band_power_epochs(raw, events, band)
        ch_idx = [ep.ch_names.index(ch) for ch in PICKS]
        for c in EVENT_ID:
            erd, t = erd_percent(ep, c)
            curves[c].append(erd[ch_idx])   # (n_picks, n_times)

    # ── Plotting ──────────────────────────────────────────────────────────── #
    fig, axes = plt.subplots(1, len(PICKS), figsize=(13, 3.8), sharey=True)
    colors = {"hands": "#e05c5c", "feet": "#4c7be0"}

    for i, ch in enumerate(PICKS):
        ax = axes[i]
        for c in EVENT_ID:
            mean_curve = np.mean(curves[c], axis=0)[i]   # avg across subjects
            ax.plot(t, mean_curve, label=c, color=colors[c], lw=1.8)
        ax.axvline(0, color="k", lw=1.0, ls="--", label="cue onset")
        ax.axhline(0, color="k", lw=0.5)
        ax.axvspan(*BASELINE, alpha=0.1, color="grey", label="baseline")
        ax.set_title(f"{ch}  [{args.band}: {band[0]:.0f}–{band[1]:.0f} Hz]",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Time from cue onset (s)", fontsize=9)
        ax.set_xlim(TMIN, TMAX)

    axes[0].set_ylabel("ERD / ERS (%)", fontsize=9)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(
        f"Event-Related (De)Synchronisation — {args.band.upper()} band"
        f"  (n={len(args.subjects)} subjects, pre-cue baseline [{BASELINE[0]}, {BASELINE[1]}] s)",
        fontsize=10)
    fig.tight_layout()
    out = f"erd_{args.band}.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
