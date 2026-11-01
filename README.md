# EEG × Deep Learning — Live Exam (3 hours)

Two parts, 90 minutes each, with a 5-minute break between them. Read the
problem-statement PDF for the full brief; this file is the map.

```
part1_diagnose/     Part 1 — a buggy motor-imagery pipeline you must fix and interpret
part2_transformer/  Part 2 — give self-attention a fair shot on tiny EEG data
```

## Before the clock starts

Set up the environment once (conda covers both parts):

```bash
conda env create -f environment.yml
conda activate eeg-dl-exam
```

Or use pip per part (each folder has its own `requirements.txt`, they overlap):

```bash
pip install -r part1_diagnose/requirements.txt
```

Then kick off the data download at the start of each part and read code while
it runs:

```bash
cd part1_diagnose       && python train.py
# ...later...
cd ../part2_transformer && python main.py
```

Both parts use the PhysioNet EEGBCI dataset (subjects 1–10, 64 channels,
160 Hz), downloaded and cached automatically by MNE (~80 MB total). CPU is
enough; each run is 1–2 minutes.

## The arc

- **Part 1 — Diagnose & Trust.** The pipeline runs cleanly but the accuracy it
  prints is wrong. Find the planted defects, fix them, interpret the ERD, and
  work out what signal the model is really using. You end with an EEGNet
  baseline whose number you can *trust*.
- **Part 2 — Build & Defend.** Carry that trusted baseline forward. Inject one
  inductive bias into an attention-based decoder and show, honestly, whether it
  helps — and whether you understand why.

Part 1 tests rigor; Part 2 tests ideas. They're weighted equally. Accuracy is
not scored — your reasoning is.

## What you hand in

One GitHub repo (commit as you work, one commit per bug fix, no squashing) with
both parts' code, an `ANSWERS.md`, and the generated artifacts
(`results.jsonl`, `erd_mu.png`, the Part 2 figure). Then a live code
walkthrough with Q&A. Details in the PDF.
