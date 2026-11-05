# Ciphertext-Only Classification — CBC Disjoint-Key Experiment

Code and results for the matched ECB/CBC shared-key vs. disjoint-key (unseen-key)
evaluation reported in Section VII-F and Table 12 of:

> E. Omara, N. E. Hammami, A. A. Hassanain, and M. M. Aboghazalah,
> "Ciphertext-Only Classification of File Encryption Algorithms Using Multi-Order
> Hex-Character N-Grams and Mode-Aware Learning," *IEEE Access* (under review).

## Contents

| File | Purpose |
|---|---|
| `protocol_exp.py` | Runs the 900-file protocol experiment (ECB and CBC; 1, 3, 6 keys; shared and disjoint keys; six classifiers) |
| `analyze.py` | Aggregates results over seeds and computes the values in Table 12 (mean ± SD, drop, B-Cipher, N/E) |
| `fig2.py` | Regenerates Figure 2 (hex-character N-gram sliding window) |
| `results/raw_results_5seeds.json` | Per-seed, per-model test predictions and confusion matrices (seeds 42, 1, 2, 3, 4) |
| `results/summary.json` | Aggregated values reported in Table 12 |
| `figures/Fig02_hex_char_ngram_window.png` | Figure 2 as published |

## Data

The experiment uses the public dataset of J. Kowalewski, 1,000 files per format
(TXT, CSV, HTML, PY, BMP, WAV):

```bash
git clone https://github.com/jkowalewski00/detection_encryption_files_1000_per_format data1000
```

## Setup

Tested with Python 3.13, numpy 2.5, scikit-learn 1.9, cryptography 50.

```bash
pip install -r requirements.txt
```

## Reproduce Table 12

```bash
# 1) run all configurations for the five seeds (about 9-10 min per seed on 2 CPU cores)
python protocol_exp.py data1000 results/raw_results_5seeds.json 42,1,2,3,4

# 2) aggregate -> prints the Table 12 values and writes summary.json
python analyze.py results/raw_results_5seeds.json
```

## Protocol summary

* 150 files per format (900 source files), 180 per class (AES-128, DES, three-key 3DES, RC4-128, None).
* Source-level 70/20/10 split before encryption (630/180/90), stratified by class and format.
* PKCS#7 padding; CBC uses a fresh random IV per file, prepended to the ciphertext.
  DES is implemented as three-key EDE with K1 = K2 = K3, which is equivalent to single DES.
* The first 256 KiB of each file are kept and encoded as lowercase hex; TF-IDF over
  hex-character 1-3-grams, with at most 1,000 features fitted on the training partition only.
* Classifiers: Random Forest (200 trees), Bagging (50 estimators), Decision Tree,
  Logistic Regression, k-NN (k = 5), MLP (one hidden layer, 256 units).
  The best model is selected on the validation partition.
* Keys: 1, 3, or 6, assigned cyclically. Under the disjoint protocol, the validation and
  test keys come from a separate pool never used in training.
* Each seed re-draws the files, partitions, keys, and IVs.

## License

MIT (see `LICENSE`).
