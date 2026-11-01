"""Data loading for the hands-vs-feet motor imagery task.

Dataset: PhysioNet EEG Motor Movement/Imagery (EEGBCI), 64 channels, 160 Hz.
Task: imagined opening/closing of BOTH FISTS (T1) vs BOTH FEET (T2).
"""
import numpy as np
import mne
from mne.datasets import eegbci

# Built-in template montage (renamed in recent MNE versions).
MONTAGE = ("colin27_1005" if "colin27_1005" in mne.channels.get_builtin_montages()
           else "standard_1005")

# Runs containing IMAGINED both-fists vs both-feet trials.
# PhysioNet EEGBCI protocol: runs 5,9,13 = REAL movement; 6,10,14 = IMAGINED movement.
# The task is imagined motor imagery, so the correct runs are [6, 10, 14].
RUNS = [6, 10, 14]
SUBJECTS = list(range(1, 11))

L_FREQ, H_FREQ = 1.0, 40.0   # broadband filter applied to the continuous recording
TMIN, TMAX = -2.0, 5.0       # epoch window around the cue (s)
CROP = (0.0, 4.0)            # part of each trial the network sees (s)
EVENT_ID = dict(hands=1, feet=2)


def load_raw(subject, runs=RUNS):
    """Download (first call only) and load the requested runs as one Raw object."""
    files = eegbci.load_data(subject, runs)
    raws = [mne.io.read_raw_edf(f, preload=True, verbose="error") for f in files]
    raw = mne.concatenate_raws(raws)
    eegbci.standardize(raw)              # "C3.." -> "C3", "Fcz" -> "FCz", ...
    raw.set_montage(MONTAGE)
    if raw.info["sfreq"] != 160:
        raise ValueError(f"S{subject:03d}: unexpected sfreq {raw.info['sfreq']}")
    return raw


def get_events(raw):
    events, _ = mne.events_from_annotations(
        raw, event_id=dict(T1=1, T2=2), verbose="error")
    return events


def epoch_subject(subject):
    """Return X (n_trials, 64, n_times) in microvolts, y (0=hands, 1=feet), ch_names."""
    raw = load_raw(subject)
    raw.filter(L_FREQ, H_FREQ, fir_design="firwin", verbose="error")
    events = get_events(raw)
    epochs = mne.Epochs(raw, events, EVENT_ID, TMIN, TMAX, baseline=None,
                        preload=True, verbose="error")
    X = epochs.copy().crop(*CROP).get_data(copy=True) * 1e6
    y = epochs.events[:, 2] - 1
    return X.astype(np.float32), y.astype(np.int64), epochs.ch_names


def build_dataset(subjects=SUBJECTS):
    """Stack all subjects. `groups` holds the subject id of every trial."""
    Xs, ys, gs = [], [], []
    for s in subjects:
        X, y, ch_names = epoch_subject(s)
        Xs.append(X); ys.append(y); gs.append(np.full(len(y), s))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs), ch_names
