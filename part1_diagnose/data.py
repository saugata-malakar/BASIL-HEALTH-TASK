"""Data loading for the hands-vs-feet motor imagery task.

Dataset : PhysioNet EEG Motor Movement/Imagery (EEGBCI), 64 channels, 160 Hz.
Task    : Imagined opening/closing of BOTH FISTS (T1) vs BOTH FEET (T2).
Reference: Goldberger et al. (2000), Schalk et al. (2004). PhysioNet EEGBCI.

Protocol notes (critical for Bug 1):
  Runs 5, 9, 13  → REAL (overt) bilateral movement  — WRONG for this task.
  Runs 6, 10, 14 → IMAGINED bilateral movement       — CORRECT (motor imagery).
  Loading the wrong runs inflates accuracy by classifying peripheral EMG
  artefacts (~10-100x larger than cortical EEG) rather than true imagery.
"""
import numpy as np
import mne
from mne.datasets import eegbci

# Built-in template montage (renamed in MNE ≥ 1.2; fall back to standard_1005).
MONTAGE = (
    "colin27_1005"
    if "colin27_1005" in mne.channels.get_builtin_montages()
    else "standard_1005"
)

# Run numbers: IMAGINED bilateral fists (T1) vs feet (T2).
# BUG 1 FIX: original starter had [5, 9, 13] (overt movement). Correct: [6, 10, 14].
RUNS = [6, 10, 14]

SUBJECTS = list(range(1, 11))  # subjects 1–10

# Broadband filter applied to the continuous recording before epoching.
# train.py may temporarily override these for frequency-band ablation studies.
L_FREQ, H_FREQ = 1.0, 40.0

# Epoch window relative to cue onset (s).
# NOTE: TMAX=5.0 extends 0.9 s into the run boundary seam.  MNE inserts a
# BAD boundary annotation at concatenation points, so the 15th trial of each
# run (whose [TMIN, TMAX] window crosses that boundary) is silently dropped.
# With 3 runs × 10 subjects × 15 trials each = 450 expected; 420 are returned.
TMIN, TMAX = -2.0, 5.0

# Temporal crop passed to the network (avoids pre-cue and post-imagery noise).
CROP = (0.0, 4.0)

EVENT_ID = dict(hands=1, feet=2)


def load_raw(subject: int, runs: list = RUNS):
    """Download (first call only) and concatenate the requested runs as one Raw.

    eegbci.standardize() normalises inconsistent channel naming across subjects
    (e.g., 'Fcz' → 'FCz', 'C3.' → 'C3') so ch_names are stable across calls.
    """
    files = eegbci.load_data(subject, runs)
    raws  = [mne.io.read_raw_edf(f, preload=True, verbose="error") for f in files]
    raw   = mne.concatenate_raws(raws)
    eegbci.standardize(raw)
    raw.set_montage(MONTAGE)
    if raw.info["sfreq"] != 160:
        raise ValueError(f"S{subject:03d}: unexpected sfreq {raw.info['sfreq']} Hz")
    return raw


def get_events(raw):
    """Extract T1/T2 events from annotations (T0 rest events are ignored)."""
    events, _ = mne.events_from_annotations(
        raw, event_id=dict(T1=1, T2=2), verbose="error")
    return events


def epoch_subject(subject: int):
    """Return X (n_trials, 64, n_times) in microvolts, y (0=hands, 1=feet), ch_names.

    Filtering uses the module-level L_FREQ / H_FREQ, which train.py may override
    temporarily for the frequency-band ablation experiment (Task 3).
    """
    raw = load_raw(subject)
    raw.filter(L_FREQ, H_FREQ, fir_design="firwin", verbose="error")
    events = get_events(raw)
    epochs = mne.Epochs(
        raw, events, EVENT_ID, TMIN, TMAX,
        baseline=None, preload=True, verbose="error")
    X = epochs.copy().crop(*CROP).get_data(copy=True) * 1e6  # V → µV
    y = epochs.events[:, 2] - 1                               # {1,2} → {0,1}
    return X.astype(np.float32), y.astype(np.int64), epochs.ch_names


def build_dataset(subjects: list = SUBJECTS):
    """Stack epochs across all subjects. `groups` holds each trial's subject id."""
    Xs, ys, gs = [], [], []
    for s in subjects:
        X, y, ch_names = epoch_subject(s)
        Xs.append(X)
        ys.append(y)
        gs.append(np.full(len(y), s, dtype=np.int64))
    return (
        np.concatenate(Xs),
        np.concatenate(ys),
        np.concatenate(gs),
        ch_names,
    )
