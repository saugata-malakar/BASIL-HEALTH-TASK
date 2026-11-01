"""Data loading for the hands-vs-feet motor imagery task.

Dataset : PhysioNet EEG Motor Movement/Imagery (EEGBCI), 64 channels, 160 Hz.
Task    : Imagined opening/closing of BOTH FISTS (T1) vs BOTH FEET (T2).
Reference: Goldberger et al. (2000), Schalk et al. (2004). PhysioNet EEGBCI.

Protocol notes (critical for Bug 1):
  Runs 5, 9, 13  -> REAL (overt) bilateral movement  -- WRONG for this task.
  Runs 6, 10, 14 -> IMAGINED bilateral movement       -- CORRECT (motor imagery).
  Loading the wrong runs inflates accuracy by classifying peripheral EMG
  artefacts (~10-100x larger than cortical EEG) rather than true imagery.
"""
import numpy as np
import mne
from mne.datasets import eegbci

# Built-in template montage (renamed in MNE >= 1.2; fall back to standard_1005).
MONTAGE = (
    "colin27_1005"
    if "colin27_1005" in mne.channels.get_builtin_montages()
    else "standard_1005"
)

# Run numbers: IMAGINED bilateral fists (T1) vs feet (T2).
# BUG 1 FIX: original starter had [5, 9, 13] (overt movement). Correct: [6, 10, 14].
RUNS = [6, 10, 14]

SUBJECTS = list(range(1, 11))  # subjects 1-10

# Default broadband filter applied to the continuous recording before epoching.
# Callers may pass l_freq/h_freq overrides to epoch_subject() and build_dataset()
# for frequency-band ablation experiments (see Task 3 in train.py).
L_FREQ, H_FREQ = 1.0, 40.0

# Epoch window relative to cue onset (s).
# NOTE on MNE boundary trial dropping:
#   mne.concatenate_raws() inserts a BAD boundary annotation at each run seam.
#   With TMAX = 5.0 the epoch window extends 0.9 s past the run boundary for
#   the 15th (last) trial of each run, causing MNE to silently reject it.
#   Result: 3 runs x 10 subjects x 14 retained trials = 420 trials (not 450).
TMIN, TMAX = -2.0, 5.0

# Temporal crop fed to the network (discards pre-cue baseline and post-imagery).
CROP = (0.0, 4.0)

EVENT_ID = dict(hands=1, feet=2)


def load_raw(subject: int, runs: list = RUNS):
    """Download (first call only) and concatenate the requested runs as one Raw.

    eegbci.standardize() normalises inconsistent channel naming across subjects
    ('Fcz' -> 'FCz', 'C3.' -> 'C3', etc.) so ch_names are stable across calls.
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


def epoch_subject(subject: int,
                  l_freq: float = L_FREQ,
                  h_freq: float = H_FREQ):
    """Return (X, y, ch_names) for one subject.

    Parameters
    ----------
    subject : int
        Subject number (1-10).
    l_freq  : float
        Band-pass lower edge in Hz (default L_FREQ = 1.0 Hz).
        Pass a higher value (e.g. 8.0) for the EOG-ablation experiment
        to strip out sub-8 Hz ocular step potentials.
    h_freq  : float
        Band-pass upper edge in Hz (default H_FREQ = 40.0 Hz).

    Returns
    -------
    X        : float32 array (n_trials, n_channels, n_times) in microvolts.
    y        : int64 array (n_trials,)  0 = hands, 1 = feet.
    ch_names : list[str]  channel names in the order they appear in X.
    """
    raw = load_raw(subject)
    raw.filter(l_freq, h_freq, fir_design="firwin", verbose="error")
    events = get_events(raw)
    epochs = mne.Epochs(
        raw, events, EVENT_ID, TMIN, TMAX,
        baseline=None, preload=True, verbose="error")
    X = epochs.copy().crop(*CROP).get_data(copy=True) * 1e6   # V -> uV
    y = epochs.events[:, 2] - 1                                # {1,2} -> {0,1}
    return X.astype(np.float32), y.astype(np.int64), epochs.ch_names


def build_dataset(subjects: list = SUBJECTS,
                  l_freq: float = L_FREQ,
                  h_freq: float = H_FREQ):
    """Stack epochs across all subjects.

    Parameters
    ----------
    subjects : list[int]
        Subject IDs to load (default: all 10).
    l_freq, h_freq : float
        Band-pass edges forwarded to epoch_subject().  Override for ablation
        studies without monkey-patching module-level constants.

    Returns
    -------
    X        : float32 (N, C, T) -- concatenated epochs across subjects.
    y        : int64  (N,)       -- class labels.
    groups   : int64  (N,)       -- subject ID for every trial.
    ch_names : list[str]         -- channel names.
    """
    Xs, ys, gs = [], [], []
    for s in subjects:
        X, y, ch_names = epoch_subject(s, l_freq=l_freq, h_freq=h_freq)
        Xs.append(X)
        ys.append(y)
        gs.append(np.full(len(y), s, dtype=np.int64))
    return (
        np.concatenate(Xs),
        np.concatenate(ys),
        np.concatenate(gs),
        ch_names,
    )
