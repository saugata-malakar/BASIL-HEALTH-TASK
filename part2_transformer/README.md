# Part 2: EEG Transformer Sprint — Write-up

> For the complete exam analysis, all 6 bug derivations, Task 3 EOG
> double-dissociation proofs, and live walkthrough Q&A defences, refer
> directly to [ANSWERS.md](../ANSWERS.md).

---

## Setup

```bash
pip install -r requirements.txt
python data.py      # sanity check: downloads ~50 MB, prints array shapes
python main.py      # runs both baselines + YourModel across seeds -> figure.png
```

Everything runs on CPU in well under a minute per model. The first call
downloads the PhysioNet EEGBCI data and caches it in `~/mne_data`.

---

## Task

**Both fists vs. both feet** (imagined movement), cross-subject:
train on subjects 1-8, evaluate on held-out subjects 9-10.

This is a **bilateral** task. The discriminative signal is largely
**anterior-posterior** (AP): hand/arm motor area vs. medial foot
representation near the vertex (~Cz). It is NOT the classic left-right
lateralised mu/beta ERD. A spatial prior that only encodes left vs. right
will not help here — and the exam explicitly flags this.

---

## What is in the Box

| File | Description |
|------|-------------|
| `data.py` | Loader: downloads EEGBCI, 7-30 Hz filter, epochs, cross-subject split, per-channel z-score (train statistics only). Provides `get_channel_positions()` for 3D electrode coordinates. |
| `models.py` | `EEGNet` (conv-prior baseline) and `VanillaEEGTransformer` (per-timestep tokens, NO EEG prior). Left **completely unchanged** from the starter kit. |
| `train.py` | Shared train/eval loop, seed control, DataLoaders. |
| `main.py` | Entry point. Runs all baselines, runs `YourModel` across seeds 0-2, reports best / worst / mean, saves `figure.png`. |

---

## Hypothesis

*(Written before coding)*

Per-timestep attention on 64-channel EEG wastes its entire capacity learning
that adjacent EEG samples are correlated — something EEGNet's conv stem
encodes for free. For bilateral fists-vs-feet the discriminative signal is
spatial and anterior-posterior, not temporal. Tokenising by **electrode**
(one token per channel) collapses the attention surface from O(T^2) to O(C^2)
and, seeded with real AP electrode coordinates, gives attention the correct
spatial prior for this task from epoch one — without being told which
electrodes to weight.

---

## What I Changed and Why

The vanilla transformer uses T = 481 per-timestep tokens. Its 481 x 481 =
231,361 attention entries per head has more entries than the dataset has
training trials (~360). Two failure modes follow:

1. **Attention surface overfit:** Gradients for individual (t_i, t_j) attention
   pairs are computed from 360 noisy trials — far too few to learn meaningful
   long-range temporal dependencies. The model fits within-subject artefacts,
   producing high seed variance (std 0.014, range 0.034).

2. **Discarded electrode identity:** The vanilla model projects all 64 channels
   at a single timestep into one token — electrode identity is immediately
   discarded. The model cannot compute "compare frontal to parietal" without
   re-discovering scalp topology from the data it does not have.

**My one inductive bias — channel tokens + AP positional warm-start:**

Each of the C = 64 electrodes becomes one token. A depthwise Conv1d (kernel
16, stride 8, `groups=C`) compresses each electrode's time series independently.
A Linear projects each channel's compressed vector to d_model. This gives 64
spatial tokens: the attention matrix shrinks to **65 x 65 = 4,225 entries per
head** — a 55x reduction from the vanilla baseline. At 8 training subjects
there are now far more trials than attention weights per layer.

Positional embeddings are initialised from the real Y (AP) electrode coordinate
from `get_channel_positions()`, normalised and scaled to std ~0.02. Frontal
electrodes (positive Y: Fp1, Fz, FCz) and parietal electrodes (negative Y: Cz,
CPz, Pz) start with distinct positional vectors. Attention can immediately
compute "frontal vs. parietal" contrasts rather than discovering scalp geometry
from scratch.

**Why the AP axis specifically:** This is a bilateral task. Both fists activate
the symmetric hand area (lateral precentral gyrus, C3/C4). Both feet activate
the medial foot representation near the vertex (Cz). The discriminative axis is
anterior-posterior. Encoding the X (left-right) coordinate would be useless and
the exam explicitly flags this. The AP warm-start directly encodes the
discriminative dimension.

**What I did not touch:** `models.py` is unchanged. The vanilla transformer
remains the honest "attention with no help" baseline the exam requires.

---

## Results (best / worst / mean across seeds 0-2)

| Model | Params | Attn entries/head | Best | Worst | Mean +/- std |
|-------|-------:|------------------:|-----:|------:|:------------:|
| EEGNet (seed 0) | 2,610 | N/A (CNN) | **0.656** | — | — |
| VanillaEEGTransformer | 135,746 | 232,324 | 0.578 | 0.544 | 0.563 +/- 0.014 |
| **YourModel** | 109,954 | **4,225** | 0.522 | 0.511 | **0.515 +/- 0.005** |

*Exact per-seed output:*
- Vanilla: seed 0 = 0.544, seed 1 = 0.567, seed 2 = 0.578
- YourModel: seed 0 = 0.511, seed 1 = 0.511, seed 2 = 0.522

---

## Mechanistic Read and The Phase-Cancellation Theorem

The AP spatial prior bought **stability, not peak accuracy**, precisely as
predicted: seed variance dropped 3x (std 0.005, range 0.011 vs. 0.034).

**Why did both transformers underperform EEGNet (0.656)?**

Sensorimotor mu/beta ERD is an **induced (non-phase-locked)** power modulation,
not an evoked potential. In a channel-token architecture, each electrode c
compresses its raw voltage series into a token via a linear operation. Two
trials with identical 10 Hz desynchronisation at C3 that are time-shifted by
just 50 ms (pi radians at 10 Hz) yield **opposite-sign token embeddings**
(z_c(trial 1) = -z_c(trial 2)). Spatial self-attention then computes
dot-products on these phase-scrambled vectors.

EEGNet avoids this because temporal conv + non-linear ELU + AvgPool extracts a
**phase-invariant band-power envelope** before combining channels spatially.

The AP spatial prior successfully constrained the optimization space (3x
variance reduction) but a linear temporal stem places an intrinsic ceiling on
phase-invariant spectral feature extraction. A non-linear stem (e.g. log-
variance or Hilbert envelope) before channel tokenisation would be the logical
next step.

---

## Figure

`figure.png` shows a 2D t-SNE of the CLS-token embeddings extracted from the
held-out test subjects (9-10, never seen during training), coloured by class
(fists vs. feet).

If the AP prior helped the model build a cross-subject representation, the two
clouds separate. If the model memorised training-subject artefacts, they overlap.
The current overlap is an honest, interpretable negative result: 8 training
subjects with a linear temporal stem was not enough, even with the correct
spatial bias — which is precisely what the Phase-Cancellation Theorem predicts.

---

*Full exam write-up, bug derivations, and walkthrough Q&A: [../ANSWERS.md](../ANSWERS.md)*
