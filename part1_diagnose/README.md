# Hands vs Feet: starter code

| File | What it does |
| --- | --- |
| `data.py` | Downloads runs via MNE, filters, epochs, returns trials in µV, labels, subject ids |
| `erd.py` | Mu/beta ERD time courses at C3, Cz, C4 → `erd_<band>.png` |
| `model.py` | EEGNet |
| `train.py` | Trains EEGNet, reports held-out accuracy, appends to `results.jsonl` |

```bash
pip install -r requirements.txt
python train.py --channels all
python erd.py --band mu
```

First run downloads about 80 MB to `~/mne_data`. CPU is enough.

The code runs without errors. That does not mean it is correct.
