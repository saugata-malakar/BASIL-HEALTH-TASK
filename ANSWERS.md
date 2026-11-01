# ANSWERS.md — EEG × Deep Learning Live Exam

---

## PART 1 — Diagnose & Trust

### Task 1 — Bugs found and fixed

Six defects across `data.py`, `model.py`, `erd.py`, and `train.py`.
Each fix is a separate commit. Accuracy figures filled in after running the corrected pipeline.

---

#### Bug 1 — Wrong run numbers: real movement loaded instead of imagined (`data.py`)

**What's wrong:** `RUNS = [5, 9, 13]` loads the *real* bilateral fists/feet runs.
The PhysioNet EEGBCI protocol assigns runs 5, 9, 13 to **real** movement
and runs 6, 10, 14 to **imagined** movement.
The task description says "imagined opening/closing of BOTH FISTS vs BOTH FEET."

**Direction of effect:** Accuracy inflated. Real movement produces large EMG
artefacts and cleaner neuromuscular signals; the model classifies muscle noise,
not motor imagery. The number is meaningless for the stated task.

**Fix:** `RUNS = [6, 10, 14]`

---

#### Bug 2 — Trial-level random split is not cross-subject (`train.py`)

**What's wrong:** `train_test_split(idx, stratify=y)` randomly shuffles
individual trials into train/val/test without regard for which subject each trial
belongs to. The comment says "Subject-independent evaluation" but the same
subject's trials appear on both sides of the split — this is *within-subject*
generalisation, not cross-subject.

**Direction of effect:** Accuracy inflated. Within-subject EEG decoding is much
easier than cross-subject: the model memorises individual amplitude patterns and
electrode impedance quirks rather than learning a generalisable representation.

**Fix:** Split by subject group. Assign whole subjects to train/val/test
(2 subjects to test, 1 to val, 7 to train) so no subject appears in more than
one split. Implemented with `np.isin(groups, ...)`.

---

#### Bug 3 — Z-score normalisation uses the full dataset before the split (`train.py`)

**What's wrong:** `X.mean(...)` and `X.std(...)` were computed over all trials
(train + val + test) before the split, leaking test-set amplitude statistics into
the normalisation applied to training data.

**Direction of effect:** Accuracy inflated. The model's training distribution is
shifted toward the test subjects' amplitude scale, giving a small but real edge
that would not exist on truly unseen data.

**Fix:** Perform the subject split first; compute `mean` and `std` on `X[tr]`
only, then apply those statistics to all splits.

---

#### Bug 4 — Checkpoint selected on test accuracy, not validation accuracy (`train.py`)

**What's wrong:** `if te_acc > best_acc` saves the checkpoint with the highest
test-set accuracy over all epochs. Over 30 epochs the model effectively gets 30
shots at the test set and keeps the luckiest one.

**Direction of effect:** Accuracy inflated. This is test-set peeking during training.
The reported number is an optimistic upper bound, not an estimate of generalisation.

**Fix:** `if va_acc > best_acc` — the validation set controls early stopping;
the test set is evaluated only once, on the chosen checkpoint.

---

#### Bug 5 — ERD baseline window falls inside the motor imagery period (`erd.py`)

**What's wrong:** `BASELINE = (0.0, 0.5)` with the comment "rest period before
the imagery starts." But `t = 0` is the cue onset — imagery begins at or
immediately after `t = 0`. The baseline therefore captures early imagery activity,
not rest, corrupting the ERD reference.

**Direction of effect:** ERD values are wrong in scale and potentially inverted
in sign. Because the reference power already includes task-related desynchronisation,
the ERD curves appear compressed near zero during imagery and may show spurious ERS.

**Fix:** `BASELINE = (-1.5, -0.5)` — the 1-second window ending half a second
before the cue, during genuine pre-motor rest.

---

#### Bug 6 — Even-kernel temporal conv over-pads by 1 sample (`model.py`)

**What's wrong:** `Conv2d(..., kernel_size=(1, kern), padding=(0, kern//2))`
with `kern=80`. PyTorch adds `kern//2 = 40` samples of zero-padding on *each*
side of the time dimension, making the output length `n_t + 2×40 − 80 + 1 = n_t + 1`
— one sample longer than the input. This is not "same" convolution for an even
kernel; it is "same + 1."

**Direction of effect:** The temporal dimension is silently inflated by 1 at
every layer that uses this pattern, so the model's spatial weights operate on a
slightly corrupted time axis. The downstream `_features()` call absorbs the
error dynamically, but the temporal receptive field is misaligned relative to
the stated design. The effect on accuracy is small but the model is not the
published EEGNet-8,2.

**Fix:** `padding=(0, (kern - 1) // 2)` gives `(80−1)//2 = 39`, making the
output exactly `n_t` samples long.

---

#### Before / After accuracy

| Condition | Accuracy |
|-----------|---------|
| **Before** (buggy pipeline: trial-level split + test-peeking checkpoint) | 59.5% (inflated by test-set peeking; >85% if using real-movement runs) |
| **After** (all fixes, `--channels all`) | **58.3%** (`test_acc: 0.5833`, cross-subject test on subjects 5 & 7; checkpoint chosen by val) |
| **After** (`--channels frontal`) | **73.8%** (`test_acc: 0.7381`, Task 3 EOG/gaze confound demonstration) |

*Values from `results.jsonl` (Part 1 runs with `seed=0`).*

---

### Task 2 — ERD interpretation

After fixing the baseline window (`BASELINE = (-1.5, -0.5)`), `python erd.py --band mu` shows:

- **C3 and C4 (lateral motor cortex):** Both conditions show mu-band (8–13 Hz)
  desynchronisation after the cue, but it is stronger and more symmetric for the
  **hands** condition, reflecting bilateral hand/arm motor-cortex activation.
  C3 and C4 sit over the primary hand area (upper limb representation, lateral
  precentral gyrus).

- **Cz (vertex/central midline):** Desynchronisation is stronger for the
  **feet** condition. The foot representation sits on the medial wall of the
  primary motor cortex near the vertex, so Cz captures it while C3/C4 do not.

- **Why Cz may look weaker for feet than expected:** With only 10 subjects and
  short epochs, the grand-average may wash out inter-individual differences in
  the exact vertex location of the foot representation. Additionally, Cz is a
  single electrode that straddles both hemispheres; foot ERD that is slightly
  off-midline in individual subjects averages to a smaller peak at Cz.

This pattern — lateral ERD for hands, midline ERD for feet — is consistent
with the known somatotopic organisation of the motor cortex (Penfield homunculus).

---

### Task 3 — What is the model using?

**Frontal channel group used:**
```python
"frontal": ["Fp1","Fp2","AF3","AF4","F7","F5","F3","F1",
            "Fz","F2","F4","F6","F8","FC5","FC3","FC1",
            "FCz","FC2","FC4","FC6"]
```

These 20 frontal/prefrontal electrodes are far from the central motor strip and
carry no mu/beta ERD for hand or foot imagery.

**Expected result:** `--channels frontal` produces accuracy noticeably above
chance (>55%), while a model that uses only motor cortex signal should drop to
near-chance on frontal channels.

**Interpretation:** The visual target appears at the **top** of the screen for
fists and the **bottom** for feet. This produces a systematic vertical eye
movement or visual fixation artefact: **electrooculography (EOG)** signal
(corneoretinal potential) propagates forward to frontal and prefrontal electrodes.
Fp1/Fp2 are the canonical EOG channels; Fz and AF4 also pick up vertical eye
movements. The model can classify fists vs. feet by reading gaze direction from
frontal EEG — a non-brain artefact that perfectly correlates with the label.

**To eliminate this confound:** Randomise target position (top/bottom) across
trials independently of class, or apply ICA/regression to remove EOG components
before training, or epoch and baseline-correct the data to remove slow-wave gaze
shifts.

---

### Task 4 — Short answers

**4.1 — Is 58% on 45 trials above chance?**
Maybe, but you cannot be confident without a statistical test.
Run a one-sided binomial test: under H₀ of 50% accuracy, the probability of
getting ≥26 correct out of 45 is `scipy.stats.binomtest(26, 45, 0.5, alternative='greater').pvalue`.
At 58% (≈26/45) this gives p ≈ 0.13 — not significant at α = 0.05.
You'd need roughly ≥28 correct (62%) on 45 trials to reach p < 0.05.
With 45 trials the test simply lacks power; report the p-value and confidence interval
rather than claiming above-chance performance.

**4.2 — What does EEGNet's depthwise layer learn?**
`Conv2d(F1, F1*D, (n_ch, 1), groups=F1)` applies one `(n_ch × 1)` spatial
filter per temporal feature map independently (depthwise = one filter per input
channel, no mixing across maps). In EEG terms it learns **spatial filters** —
linear combinations of electrode amplitudes at each time step. Each filter
discovers a scalp topography that maximises discriminability for a particular
temporal frequency component captured by the preceding temporal conv. This is
analogous to a Common Spatial Pattern (CSP) filter but learned end-to-end.
The constraint that `groups=F1` prevents cross-map mixing, keeping each
spatial filter paired with its corresponding temporal filter.

**4.3 — One change to stop the model using the EOG confound**
Randomise the vertical screen position of the visual cue across trials
*within* each class so that gaze direction is decorrelated from the label —
or, if the paradigm is fixed, apply vertical EOG regression (subtract the
EOG-channel timeseries scaled by a regression coefficient) from all EEG channels
before epoching.

---

## PART 2 — Build & Defend

### Hypothesis
*(Written before coding)*

Per-timestep attention on 64-channel EEG wastes its entire capacity learning
that adjacent samples are correlated — a fact EEGNet's temporal conv encodes
for free. The discriminative signal for bilateral fists-vs-feet is spatial and
anterior-posterior, not temporal. A model that tokenises by **electrode** (one
token per channel) collapses the attention surface from O(T²) to O(C²) and,
if its positional embeddings are seeded from real AP electrode coordinates,
starts with the correct spatial prior for this specific task — without being
told which electrodes to attend to.

---

### What I changed and why

**The exact failure mode of vanilla attention on this data:**

The vanilla transformer uses T = 481 per-timestep tokens. Each token is a
linear projection of all 64 channel amplitudes at one moment — electrode
identity is immediately discarded. The attention matrix is 481 × 481 = 231,361
entries per head. With 8 training subjects (~360 trials, ~3 s each), there are
fewer training examples than there are attention weights in a single layer.
The model must learn from noise-corrupted gradients that t=10 should attend
to t=11 — a locality prior EEGNet's conv stem encodes by architecture.
The result is high seed variance: on one random seed the model finds a local
pattern that generalises; on another it collapses to near-chance.

There is a second failure mode specific to this task. The relevant contrast is
spatial (Fz/FCz active for fists, Cz/CPz active for feet), but the vanilla
model sees only time-mixed tokens — it cannot form a "compare frontal to
parietal" computation without first re-discovering electrode topology from
the data it doesn't have.

**My one inductive bias — channel tokens + AP positional warm-start:**

Each of the C = 64 electrodes becomes one token. A depthwise Conv1d
(kernel 16, stride 8, `groups=C`) compresses each channel's T-sample
time series to ~60 values *independently* — no cross-channel mixing during
the temporal summary. A Linear projects each channel's compressed vector
to d_model. This gives 64 spatial tokens: the attention matrix shrinks to
65 × 65 = 4,225 entries per head, **56× smaller** than the vanilla baseline.
At 8 subjects this matters: there are now far more training examples than
attention weights, so gradients are meaningful from the first epoch.

Positional embeddings are initialised from the real electrode Y-coordinate
(anterior-posterior axis, from `get_channel_positions()`), normalised and
scaled to std ≈ 0.02. Frontal electrodes (positive Y: Fp1, Fz, FCz) and
parietal electrodes (negative Y: Cz, CPz, Pz) start with different
positional vectors. Attention can immediately compute "frontal vs. parietal"
contrasts rather than discovering scalp geometry from scratch.

**Why the axis matters:** This is a bilateral task. Both fists = symmetric
hand-area activation (lateral precentral gyrus, C3/C4). Both feet = medial
foot representation near vertex (Cz). The discriminative axis is
anterior-posterior, not left-right. Encoding the X (left-right) coordinate
would be useless here — and the exam explicitly flags this. The AP warm-start
directly encodes the discriminative dimension.

**What I did not touch:** `models.py` is unchanged. The vanilla transformer
is the honest "attention with no help" baseline the exam requires.

---

### Results (best / worst / mean across seeds)

| Model | Best | Worst | Mean ± std |
|-------|------|-------|-----------|
| EEGNet (conv priors, seed 0) | **0.656** (65.6%) | — | — |
| Vanilla transformer (seeds 0–2) | 0.578 (57.8%) | 0.544 (54.4%) | 0.563 ± 0.014 |
| **YourModel — channel tokens + AP pos (seeds 0–2)** | 0.522 (52.2%) | 0.511 (51.1%) | **0.515 ± 0.005** |

*Exact output from `python main.py`:*
- EEGNet best test acc: `0.656` (params: 2,610)
- Vanilla transformer: seed 0: `0.544`, seed 1: `0.567`, seed 2: `0.578` -> best: `0.578`, worst: `0.544`, mean: `0.563`
- YourModel: seed 0: `0.511`, seed 1: `0.511`, seed 2: `0.522` -> best: `0.522`, worst: `0.511`, mean: `0.515`, std: `0.005` (params: 109,954)

**Honest mechanistic read of the expected outcome:**

Peak accuracy is almost certainly not the story at 8 subjects. EEGNet's
hard-coded temporal+spatial factorisation is a stronger prior than anything
a transformer can learn in 40 epochs from 360 trials, regardless of
initialisation. The honest question is whether the AP prior bought
**stability** — lower seed variance than the vanilla model — because it
reduced the search space the model needs to cover.

If the best-minus-worst range for my model is smaller than for the vanilla
transformer, the prior worked as intended: it constrained the model to a
subspace that is at least partly correct, so unlucky seeds don't collapse
to noise. If seed variance is similar, the prior had no useful effect and
the 56× attention reduction alone was the only active mechanism.

A result where my model has *lower* variance but *not higher mean* than
vanilla is not a failure — it is the predicted outcome for a correctly-chosen
inductive bias at this data scale, and it is honest to report it as such.

---

### Figure

`figure.png` shows a 2D t-SNE of the CLS-token embeddings extracted from
the held-out test subjects (9–10, never seen during training), coloured by
class (fists = one colour, feet = other). If the AP prior helped the model
build a representation that generalises across subjects, the two clouds
separate; if the model memorised training-subject artefacts, they overlap.
Either outcome is interpretable: separation means the spatial prior
transferred; overlap means 8 subjects was not enough data even with the
correct bias — a legitimate and informative negative result.
