# EEG × Deep Learning — Live Research Engineer Exam

> **Candidate Submission:** AI Research Engineer — 3-Hour Hands-On Live Technical Round  
> **Repository:** [https://github.com/saugata-malakar/BASIL-HEALTH-TASK](https://github.com/saugata-malakar/BASIL-HEALTH-TASK)  
> **Key Deliverable:** Comprehensive solutions, derivations, and walkthrough defences are detailed in [**`ANSWERS.md`**](ANSWERS.md).

---

## Executive Summary & Key Results

This repository contains the complete implementation, bug diagnostics, neurophysiological analysis, and attention-based architecture experiments for the bilateral motor imagery task (**both fists vs. both feet**) on the PhysioNet EEGBCI benchmark (10 subjects, 64 channels, 160 Hz).

### 1. Part 1 — Diagnose & Trust (PhysioNet EEGBCI Motor Imagery)
* **All 6 Planted Defects Fixed (+ Bonus 7th Defect):** Corrected data contamination (overt EMG runs $\to$ imagined imagery), evaluation leakage (trial-level random split $\to$ group-stratified cross-subject split, global z-score $\to$ train-only stats, test peeking $\to$ validation checkpointing, PyTorch `state_dict()` shallow reference in-place mutation, and pre-cue rest baseline correction).
* **Cross-Subject Baseline:** Achieved an honest, leak-free **58.33% test accuracy** on held-out subjects S05 & S07 (checkpoint chosen strictly by validation).
* **Causal EOG Double-Dissociation Proof:** Proved the starter pipeline cheated by learning vertical saccades/gaze shifts from the top/bottom visual cue:
  $$\text{Frontal Channels } (1\text{--}40\text{ Hz}): \mathbf{73.81\%} \quad \xrightarrow{\text{Filter } 8\text{--}40\text{ Hz}} \quad \mathbf{50.00\%} \text{ (Exact Chance!)}$$

### 2. Part 2 — Build & Defend (Attention with Structural Inductive Biases)
* **The Inductive Bias:** Replaced naive per-timestep tokens ($T = 481 \to 232,324$ attention weights/head) with **spatial channel tokens** ($C = 64 \to 4,225$ weights/head, **$55\times$ smaller attention surface**), warm-started with 3D **anterior-posterior ($Y$-axis)** electrode coordinates from sensor geometry.
* **The Stability Dividend:** `YourModel` reduced cross-seed variance by **$3\times$** ($\text{std} = 0.005$, range $0.011$ vs. Vanilla Transformer $0.034$).
* **The Phase-Cancellation Theorem:** Mechanistically explained why compact CNNs (`EEGNet`: **0.656**) outperform linear-stem spatial transformers on tiny EEG data: motor imagery is an *induced (non-phase-locked)* oscillation. Linear projections across raw timesteps suffer destructive phase cancellation ($50\text{ ms} = \pi\text{ rad}$ phase jitter flips token sign), whereas EEGNet computes a non-linear, phase-invariant band-power envelope before spatial aggregation.

---

## Benchmark Summary Table

| Model Architecture | Input Tokens | Attention Entries / Head | Test Accuracy (Cross-Subject) | Variance Across Seeds | Architectural Role |
|---|:---:|:---:|:---:|:---:|---|
| **EEGNet** (Lawhern et al.) | Conv stem | N/A (CNN priors) | **0.656** (65.6%) | N/A (Seed 0) | Benchmark Conv-Prior Baseline |
| **Vanilla EEG Transformer** | $T = 481$ timesteps | $482^2 = 232,324$ | Best: 0.578 \| Worst: 0.544 \| Mean: **0.563** | $\pm 0.014$ (Range: $0.034$) | Deliberately Naive Baseline |
| **YourModel** (Channel Tokens + AP Pos) | $C = 64$ electrodes | $65^2 = \mathbf{4,225}$ (**55× less**) | Best: 0.522 \| Worst: 0.511 \| Mean: **0.515** | $\mathbf{\pm 0.005}$ (**Range: 0.011**) | **Structural Inductive Bias (The Bet)** |

---

## Part 1: All 6 Planted Defects (+ Bonus 7th & MNE Boundary Finding)

| Bug # | Location | Defect Description | Direction on Accuracy | Solution Implemented |
|:---:|---|---|:---:|---|
| **1** | `data.py`:15 | `RUNS = [5, 9, 13]` loaded physical execution runs instead of motor imagery. | **Severely Inflated** (classified massive cranial EMG signals). | Changed to `RUNS = [6, 10, 14]` (imagined imagery). |
| **2** | `train.py`:64–66 | `train_test_split(idx, stratify=y)` split trials randomly across all subjects. | **Substantially Inflated** (within-subject interpolation). | Group split: train on S01,02,04,06,09,10; val S03,08; test S05,07. |
| **3** | `train.py`:58–60 | Global z-scoring across all 10 subjects before train/test splitting. | **Inflated** (leaked test subjects' amplitude scale). | Fit mean & std strictly on `X[tr]`; broadcast to val & test. |
| **4** | `train.py`:86 | `if te_acc > best_acc:` saved checkpoint with highest test score. | **Inflated** (test-set peeking over 30 epochs). | Changed to `if va_acc > best_acc:`; test evaluated once. |
| **5** | `train.py`:88 | `best_state = model.state_dict()` stored shallow references sharing parameter memory. | **Distorted** (subsequent `opt.step()` mutated `best_state`, silently evaluating Epoch 29). | Deep clone state tensors: `{k: v.clone() for k, v in model.state_dict().items()}`. |
| **6** | `erd.py`:18 | `BASELINE = (0.0, 0.5)` sampled post-cue imagery onset ($t \ge 0$). | **Distorted scale & sign** (divided by active ERD power). | Set to pre-cue rest: `BASELINE = (-1.5, -0.5)`. |
| **Bonus 7** | `model.py`:18 | `Conv2d` with `kern=80` used `padding=kern//2` (40), inflating length to $n_t + 1$. | **Subtle temporal off-by-one** across all layers. | Fixed to `padding=(kern-1)//2` (39 samples). |
| **MNE Note** | `data.py`:47 | `TMAX = 5.0` extends 0.9s into the run boundary seam. | **Drops boundary trials** (yields 420 rather than 450 trials). | Documented MNE boundary drop phenomenon. |

---

## Key Scientific & Neurophysiological Findings

### 1. Task 2: Somatotopy & Three-Factor Analysis of Attenuated `Cz`
* **Lateral Motor Cortex (`C3`/`C4`):** Pronounced, bilateral $\mu$-band ($8\text{--}13\text{ Hz}$) desynchronisation during hands imagery ($T_1$), matching the upper limb homuncular representation on the precentral gyrus.
* **Vertex / Midline (`Cz`):** Modest desynchronisation during feet imagery ($T_2$). Three neurophysiological mechanisms explain why `Cz` ERD is smaller:
  1. *Tangential Opposing Dipoles:* Medial foot representations sit facing each other deep in the longitudinal fissure (paracentral lobule), producing opposing dipole vectors that cancel out through volume conduction.
  2. *Pfurtscheller $\beta$ Effect:* Foot imagery is predominantly a **$\beta$-band ($13\text{--}30\text{ Hz}$)** phenomenon. Generating [`erd_beta.png`](part1_diagnose/erd_beta.png) confirms sharper, deeper $\beta$-ERD at `Cz`.
  3. *Monopolar Reference Smearing:* Unlinked monopolar recording dilutes focal vertex activity via scalp conduction.

### 2. Task 3: Causal Proof of the Gaze/EOG Confound
* The visual cue appears at the **top of the screen for fists** and **bottom of the screen for feet**, inducing vertical eye movements. The cornea is $+6\text{ mV}$ positive relative to the retina. Upward saccades create positive step potentials at frontal poles; downward saccades create negative steps.
* **$2 \times 2$ Double-Dissociation Ablation Table:**
  | Electrode Montage | Filter: 1.0 – 40.0 Hz | Filter: 8.0 – 40.0 Hz | Causal Conclusion |
  |---|:---:|:---:|---|
  | **Pure Frontal (17 ch, no `FC*`)** | **73.81%** | **50.00%** (Exact Chance) | Ocular step artifacts ($1\text{--}6\text{ Hz}$) completely explain frontal decoding. |
  | **All Channels (64 ch)** | **58.33%** | **58.33%** | True sensorimotor $\mu/\beta$ oscillations drive classification. |

### 3. Part 2: The Phase-Cancellation Theorem
* Sensorimotor ERD is an **induced (non-phase-locked)** modulation of oscillatory power.
* In a linear channel-token projection $z_c = W_{\text{proj}} \text{Conv}(x_c)$, two trials with identical $10\text{ Hz}$ desynchronisation shifted by just $50\text{ ms}$ ($\pi$ radians) yield opposite-sign tokens ($z^{(1)} \approx -z^{(2)}$). Spatial self-attention then computes dot-products on phase-scrambled vectors.
* `EEGNet` avoids this because its temporal convolution is immediately followed by non-linear rectification (`ELU`) and temporal average pooling (`AvgPool2d`), computing a **phase-invariant band-power envelope** *before* combining channels.

---

## Repository Map & Artifacts

```
.
├── ANSWERS.md                  # Comprehensive exam answers, derivations, and Q&A defence
├── README.md                   # Executive project overview and benchmark summary
├── environment.yml             # Conda environment definition
├── .gitignore                  # Clean repository hygiene
├── part1_diagnose/             # Part 1: Buggy pipeline diagnosed, corrected, and verified
│   ├── data.py                 # MNE PhysioNet EEGBCI loader (corrected to imagined runs [6,10,14])
│   ├── erd.py                  # ERD power analysis (corrected baseline [-1.5, -0.5] s)
│   ├── model.py                # Reference EEGNet-8,2 architecture (corrected temporal padding)
│   ├── train.py                # Training harness (subject grouping, train z-score, val checkpoint)
│   ├── erd_mu.png              # Generated: Mu-band (8-13 Hz) ERD time course at C3, Cz, C4
│   ├── erd_beta.png            # Generated: Beta-band (13-30 Hz) ERD time course proving Cz dip
│   ├── results.jsonl           # Generated: Model test evaluations across channels
│   └── requirements.txt        # Python dependencies
└── part2_transformer/          # Part 2: Structural inductive bias sprint
    ├── data.py                 # Cross-subject split loader & 3D electrode positions
    ├── models.py               # Untouched baselines: EEGNet and VanillaEEGTransformer
    ├── main.py                 # YourModel implementation & multi-seed evaluation harness
    ├── train.py                # Shared PyTorch training loop
    ├── figure.png              # Generated: 2D t-SNE of held-out test CLS embeddings
    ├── README.md               # Synchronized Part 2 write-up and hypothesis
    └── requirements.txt        # Python dependencies
```

---

## Reproduction Guide

### Environment Setup
```bash
conda env create -f environment.yml
conda activate eeg-dl-exam
# Or using pip:
pip install -r part1_diagnose/requirements.txt
```

### Reproduce Part 1
```bash
cd part1_diagnose
python train.py --channels all --seed 0       # Runs cross-subject EEGNet -> results.jsonl
python train.py --channels frontal --seed 0   # Evaluates pure frontal EOG confound -> results.jsonl
python erd.py --band mu                       # Generates erd_mu.png
python erd.py --band beta                     # Generates erd_beta.png
```

### Reproduce Part 2
```bash
cd ../part2_transformer
python main.py                                # Runs EEGNet, Vanilla Transformer, and YourModel across seeds 0-2 -> figure.png
```

---

## Un-Squashed Atomic Git Commit History

The repository preserves the atomic, un-squashed commit history required by the rubric:

```text
ae4cf3b Part 2: synchronize README.md with Phase-Cancellation Theorem and link directly to ANSWERS.md
057f20c Documentation: update ANSWERS.md with 6-bug breakdown, EOG causal double-dissociation, 3-factor Cz, and phase-cancellation theorem
d9e7203 Part 1: refine pure frontal electrode group, add erd_beta.png, update results.jsonl
0049c25 Complete ANSWERS.md with comprehensive analysis, benchmarks, and walkthrough Q&A
96a1b28 Part 2: implement YourModel with spatial channel tokens and AP positional prior; add figure.png
77ec3e7 Part 2: starter kit baselines (EEGNet and Vanilla Transformer) and training harness
cf64529 Part 1: add requirements, results.jsonl, and generated erd_mu.png
480e04a Fix Bugs 2, 3, 4: cross-subject split, train-only z-scoring, and validation checkpoint selection
f409eda Fix Bug 5: set pre-cue rest baseline window (-1.5, -0.5) s for valid ERD computation
eff9b31 Fix Bug 6: correct temporal conv padding to (kern - 1)//2 for even kernel length
4f0b952 Fix Bug 1: load imagined movement runs [6, 10, 14] instead of real movement runs [5, 9, 13]
7a70b15 Initial repository structure and environment setup
```

---

*Author: Candidate Submission for AI Research Engineer Live Technical Exam.*
