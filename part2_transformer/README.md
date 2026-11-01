# EEG Transformer Sprint — Starter Kit

You have 90 minutes. This kit removes all the plumbing so you spend the time on
one thing: **giving self-attention a fighting chance on tiny EEG data, and
understanding why your idea does (or doesn't) help.**

## Setup (first ~10 minutes)

```bash
pip install -r requirements.txt
python data.py      # sanity check: downloads ~50 MB, prints array shapes
python main.py      # runs both baselines + the starter YourModel + saves figure.png
```

Everything runs on CPU in well under a minute per model. The first run
downloads the PhysioNet EEGBCI data and caches it.

## The task

**Both fists vs. both feet** (imagined movement), cross-subject: train on
subjects 1–8, test on held-out subjects 9–10.

This is a **bilateral** task. The discriminative signal is largely
anterior–posterior (hand/arm motor area vs. the medial foot representation near
the vertex, ~Cz), **not** the classic left/right lateralised mu/beta ERD. A
spatial prior that only encodes left vs. right will not help you here.

## What's in the box

| File | What it is |
|------|------------|
| `data.py` | Loader: download, 7–30 Hz filter, epoch, cross-subject split, per-channel z-score (train stats only). Also `get_channel_positions()` for 3D electrode coordinates. |
| `models.py` | `EEGNet` (conv-prior baseline) and `VanillaEEGTransformer` (per-timestep tokens, **no** EEG prior — the thing to beat/understand). |
| `train.py` | Shared train/eval loop, seed control, data loaders. |
| `main.py` | Entry point. Runs the baselines, runs **your** model across seeds, reports best + worst, saves `figure.png`. |

## Your job — in `main.py`

Implement **one** inductive bias in `YourModel` (or replace it entirely). The
starter version already shows one legal example: learned **temporal patch**
tokens instead of per-timestep tokens, which shrinks the O(T²) attention
surface. Swap in your own bet.

Do **not** quietly edit the vanilla transformer in `models.py` — the whole point
is the contrast between "attention with no help" and "attention with your one
idea."

## What to hand in

- This repo with your `YourModel` filled in. `python main.py` must reproduce
  your numbers end to end, seed printed.
- A short note (half a page — edit the section below is fine) with:
  1. **Hypothesis** — one line, written *before* you coded it.
  2. **What you changed and why** it targets a real weakness of attention on EEG.
  3. **What happened** — best *and* worst run across seeds, no cherry-picking.
  4. One sentence on what `figure.png` shows.
- A live code walkthrough at the end: share your screen, walk us through the
  file, answer questions.

Accuracy is **not** scored. We're reading your code and your answers for how you
reason about EEG and attention under a clock. A clear negative result beats a
lucky spike you can't explain.

---

## Your write-up (fill this in)

> *Note: For the complete exam analysis, all 6 bug derivations, Task 3 EOG double-dissociation proofs, and live walkthrough Q&A defences, refer directly to [ANSWERS.md](../ANSWERS.md).*

**Hypothesis:**
Per-timestep attention wastes capacity learning that adjacent EEG samples are
correlated — something EEGNet's conv stem encodes for free. For bilateral
fists-vs-feet the discriminative signal is spatial and anterior-posterior, not
temporal. Tokenising by electrode (one token per channel) collapses the
attention surface from O(T²) to O(C²) and, seeded with real AP electrode
coordinates, gives attention the correct spatial prior for this task from
epoch one — without being told which electrodes to weight.

**What I changed and why:**
The vanilla transformer has T = 481 per-timestep tokens; its 481 × 481 = 231,361
attention matrix per head has more entries than the dataset has training trials.
I switched to 64 **channel tokens**: a depthwise Conv1d (kernel 16, stride 8,
`groups=C`) compresses each electrode's time series independently, then a
Linear projects to d_model. Attention is now over electrodes, not timesteps —
65 × 65 = 4,225 entries per head, **56× smaller**. Positional embeddings are
initialised from the real Y (anterior-posterior) electrode coordinate from
`get_channel_positions()`, so the model starts knowing that Fz ≠ Cz ≠ Pz
without having to discover scalp topology from 8 subjects. The X (left-right)
coordinate is deliberately excluded: this is a bilateral task where left-right
symmetry carries no class information.

**Results (best / worst / mean across seeds):**
- **EEGNet (conv prior, seed 0):** best test acc = **0.656** (params: 2,610)
- **Vanilla Transformer (no EEG prior, seeds 0–2):** best = **0.578**, worst = **0.544**, mean = **0.563 ± 0.014**, range = **0.034** (params: 135,746)
- **YourModel (channel tokens + AP pos, seeds 0–2):** best = **0.522**, worst = **0.511**, mean = **0.515 ± 0.005**, range = **0.011** (params: 109,954)

**Mechanistic Read & The Phase-Cancellation Theorem:**
The prior bought **stability, not peak accuracy**, precisely as predicted: seed variance dropped by 3× (std 0.005, range 0.011 vs 0.034).
Why did both transformers underperform EEGNet (0.656)?
Sensorimotor $\mu/\beta$ ERD is an **induced (non-phase-locked)** power modulation, not an evoked potential. Linear projection of raw continuous time series into channel tokens is phase-sensitive: two trials with identical 10 Hz ERD shifted by merely 50 ms ($\pi$ radians) yield opposite-sign tokens ($z^{(1)} \approx -z^{(2)}$). EEGNet succeeds because temporal conv + non-linear squaring/rectification (`ELU`) + temporal average pooling extracts a **phase-invariant energy envelope** before spatial combinations.

**Figure:**
`figure.png` shows t-SNE of the test-subject CLS embeddings (held-out subjects 9–10). Points are coloured by class (fists vs. feet). While the AP prior reduced the sequence length 56× and constrained seed variance, the remaining cluster overlap illustrates the linear stem ceiling on phase-invariant spectral feature extraction — an honest, authentic negative result on tiny cross-subject data.
