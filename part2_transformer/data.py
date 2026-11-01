"""
EEG Transformer Sprint — data loading.

Downloads the PhysioNet EEG Motor Movement/Imagery (EEGBCI) motor-imagery runs,
band-pass filters to mu+beta, epochs, and returns a cross-subject split.

Task: BOTH FISTS vs BOTH FEET  (imagined movement).
  - Runs 6, 10, 14 are the imagined bilateral task.
  - Annotation T1 = both fists, T2 = both feet, T0 = rest (dropped).

NOTE: this is a BILATERAL task. The discriminative signal is largely
anterior-posterior (hand/arm motor area vs. the medial foot representation
near the vertex, ~Cz), NOT the classic left/right lateralised mu/beta ERD.
A spatial prior that only encodes left vs right will not help here.

The first call downloads ~50 MB to the MNE data dir and caches it.
"""
import numpy as np
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

mne.set_log_level("ERROR")

RUNS = [6, 10, 14]                       # imagined: both fists vs both feet
TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8]
TEST_SUBJECTS = [9, 10]                   # held-out subjects -> cross-subject

TMIN, TMAX = 0.0, 3.0                     # 3 s window -> 481 samples at 160 Hz
L_FREQ, H_FREQ = 7.0, 30.0               # mu + beta


def _load_subjects(subjects):
    raws = []
    for s in subjects:
        fnames = eegbci.load_data(s, RUNS, verbose="ERROR")
        raws.append(concatenate_raws(
            [read_raw_edf(f, preload=True, verbose="ERROR") for f in fnames]))
    raw = concatenate_raws(raws)
    # Strip trailing dots some PhysioNet EDFs put on channel names, then set montage.
    raw.rename_channels(lambda c: c.strip("."))
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"),
                    match_case=False, on_missing="ignore")
    raw.filter(L_FREQ, H_FREQ, fir_design="firwin", verbose="ERROR")

    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    # In these runs T1/T2 map to annotation ids 2/3 (T0=1 is rest).
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    epochs = mne.Epochs(raw, events, event_id={"fists": 2, "feet": 3},
                        tmin=TMIN, tmax=TMAX, picks=picks,
                        baseline=None, preload=True, verbose="ERROR")
    X = epochs.get_data().astype("float32")          # (N, C, T)
    y = (epochs.events[:, -1] == 3).astype("int64")  # 0 = fists, 1 = feet
    return X, y


def get_channel_positions():
    """3D electrode coordinates (C, 3), aligned to the channel order in X.

    Handy if your idea uses electrode geometry as a structural/positional prior.
    """
    fnames = eegbci.load_data(1, RUNS[:1], verbose="ERROR")
    raw = concatenate_raws(
        [read_raw_edf(f, preload=True, verbose="ERROR") for f in fnames])
    raw.rename_channels(lambda c: c.strip("."))
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"),
                    match_case=False, on_missing="ignore")
    raw.filter(L_FREQ, H_FREQ, fir_design="firwin", verbose="ERROR")
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    pos = np.array([raw.info["chs"][p]["loc"][:3] for p in picks], dtype="float32")
    return pos


def load_data(standardize=True):
    """Return (Xtr, ytr, Xte, yte). Cross-subject split: train 1-8, test 9-10."""
    Xtr, ytr = _load_subjects(TRAIN_SUBJECTS)
    Xte, yte = _load_subjects(TEST_SUBJECTS)
    if standardize:
        # Per-channel z-score using TRAIN statistics only (no test leakage).
        mu = Xtr.mean(axis=(0, 2), keepdims=True)
        sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
    return Xtr, ytr, Xte, yte


if __name__ == "__main__":
    Xtr, ytr, Xte, yte = load_data()
    print(f"train X {Xtr.shape}  y {ytr.shape}  balance {ytr.mean():.2f}")
    print(f"test  X {Xte.shape}  y {yte.shape}  balance {yte.mean():.2f}")
    print(f"channel positions {get_channel_positions().shape}")
