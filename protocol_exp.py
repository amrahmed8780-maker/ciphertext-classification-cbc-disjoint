"""Reimplementation of the 900-file protocol-focused experiment
(shared-key vs disjoint-key) under ECB and CBC.

Follows the manuscript description:
  * 150 files per format x 6 formats = 900 source files, 180 per class
  * source-level 70/20/10 split before encryption (630/180/90)
  * AES-128, DES, 3-key EDE3 3DES (PKCS#7), RC4-128, None
  * CBC: fresh random IV per file, prepended to ciphertext
  * first 256 KiB retained, lossless lowercase hex encoding
  * TF-IDF over hex-character 1-3 grams, max 1,000 features, fitted on train only
  * models: RF, Bagging, DT, LR, KNN, MLP; best model chosen on validation
  * key settings 1/3/6, cyclic assignment;
    disjoint: train keys and val/test keys are completely separate pools
"""
import os, sys, json, glob, time, random
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, modes, algorithms
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.decrepit.ciphers.algorithms import TripleDES, ARC4
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

DATA = sys.argv[1]
OUT = sys.argv[2]
SEEDS = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [42]
FORMATS = ["bmp", "csv", "html", "pyth", "txt", "wav"]
CLASSES = ["AES", "DES", "3DES", "RC4", "None"]
PER_FORMAT = 150
MAXB = 262144


def load_sources(seed):
    rng = random.Random(seed)
    src = []
    for fmt in FORMATS:
        files = sorted(glob.glob(os.path.join(DATA, fmt, "*")))
        pick = rng.sample(files, PER_FORMAT)
        # 30 files of each format per class -> balanced classes and formats
        labels = [CLASSES[i % 5] for i in range(PER_FORMAT)]
        rng.shuffle(labels)
        for f, c in zip(pick, labels):
            with open(f, "rb") as fh:
                src.append((fmt, c, fh.read(MAXB + 64)))
    return src


def split(src, seed):
    """Stratified (class x format) 70/20/10 split by source id."""
    rng = random.Random(seed + 1)
    tr, va, te = [], [], []
    groups = {}
    for i, (fmt, c, _) in enumerate(src):
        groups.setdefault((fmt, c), []).append(i)
    for g, idx in sorted(groups.items()):
        rng.shuffle(idx)  # 30 per group -> 21/6/3
        tr += idx[:21]; va += idx[21:27]; te += idx[27:30]
    return tr, va, te


def make_keys(rng, n):
    return [{"AES": bytes(rng.getrandbits(8) for _ in range(16)),
             "DES": bytes(rng.getrandbits(8) for _ in range(8)),
             "3DES": bytes(rng.getrandbits(8) for _ in range(24)),
             "RC4": bytes(rng.getrandbits(8) for _ in range(16))} for _ in range(n)]


def encrypt(c, data, key, mode, rng):
    if c == "None":
        return data
    if c == "RC4":
        e = Cipher(ARC4(key["RC4"]), None).encryptor()
        return e.update(data)
    if c == "AES":
        alg, bs = algorithms.AES(key["AES"]), 128
    elif c == "DES":
        alg, bs = TripleDES(key["DES"] * 3), 64  # K||K||K EDE == single DES
    else:
        alg, bs = TripleDES(key["3DES"]), 64
    p = padding.PKCS7(bs).padder()
    pt = p.update(data) + p.finalize()
    if mode == "ECB":
        e = Cipher(alg, modes.ECB()).encryptor()
        return e.update(pt) + e.finalize()
    iv = bytes(rng.getrandbits(8) for _ in range(bs // 8))
    e = Cipher(alg, modes.CBC(iv)).encryptor()
    return iv + e.update(pt) + e.finalize()


def ngram_counts(b):
    """Exact counts of hex-character 1-,2-,3-grams of lowercase hex(b).
    Index layout: [0,16) unigrams, [16,272) bigrams, [272,4368) trigrams."""
    a = np.frombuffer(b, dtype=np.uint8)
    nib = np.empty(2 * len(a), dtype=np.int64)
    nib[0::2] = a >> 4
    nib[1::2] = a & 15
    v = np.zeros(16 + 256 + 4096, dtype=np.float64)
    v[:16] = np.bincount(nib, minlength=16)
    if len(nib) >= 2:
        v[16:272] = np.bincount(nib[:-1] * 16 + nib[1:], minlength=256)
    if len(nib) >= 3:
        v[272:] = np.bincount(nib[:-2] * 256 + nib[1:-1] * 16 + nib[2:], minlength=4096)
    return v


def models(seed):
    return {
        "RF": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
        "Bagging": BaggingClassifier(n_estimators=50, random_state=seed, n_jobs=-1),
        "DT": DecisionTreeClassifier(random_state=seed),
        "LR": LogisticRegression(max_iter=2000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "MLP": MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, random_state=seed),
    }


def assign_keys(idx_list, pool):
    return {i: pool[j % len(pool)] for j, i in enumerate(idx_list)}


def run(seed):
    src = load_sources(seed)
    tr, va, te = split(src, seed)
    y = np.array([CLASSES.index(c) for _, c, _ in src])
    results = []
    for mode in ["ECB", "CBC"]:
        for nk in [1, 3, 6]:
            for proto in ["shared", "disjoint"]:
                rng = random.Random(f"{seed}-{mode}-{nk}-{proto}")
                if proto == "shared":
                    pool = make_keys(rng, nk)
                    km = {}
                    for part in (tr, va, te):
                        km.update(assign_keys(part, pool))
                else:
                    pool_tr = make_keys(rng, nk)
                    pool_ev = make_keys(rng, nk)  # never seen in training
                    km = assign_keys(tr, pool_tr)
                    km.update(assign_keys(va, pool_ev))
                    km.update(assign_keys(te, pool_ev))
                X = np.vstack([ngram_counts(encrypt(c, d, km[i], mode, rng)[:MAXB])
                               for i, (_, c, d) in enumerate(src)])
                top = np.argsort(-X[tr].sum(0), kind="stable")[:1000]
                tf = TfidfTransformer().fit(X[tr][:, top])
                Xtr, Xva, Xte = (tf.transform(X[p][:, top]) for p in (tr, va, te))
                rows = {}
                for name, m in models(seed).items():
                    m.fit(Xtr, y[tr])
                    pv, pt = m.predict(Xva), m.predict(Xte)
                    cm = confusion_matrix(y[te], pt, labels=range(5))
                    rows[name] = {"val": accuracy_score(y[va], pv),
                                  "test": accuracy_score(y[te], pt),
                                  "f1": f1_score(y[te], pt, average="macro"),
                                  "cm": cm.tolist(), "pred": pt.tolist()}
                best = max(rows, key=lambda k: rows[k]["val"])
                rec = {"seed": seed, "mode": mode, "keys": nk, "protocol": proto,
                       "best": best, "y_test": y[te].tolist(), "models": rows}
                results.append(rec)
                print(seed, mode, nk, proto, "best(val)=", best,
                      "test=%.3f" % rows[best]["test"],
                      {k: round(v["test"], 3) for k, v in rows.items()}, flush=True)
    return results


if __name__ == "__main__":
    allres = []
    for s in SEEDS:
        t = time.time()
        allres += run(s)
        print("seed", s, "done in %.0fs" % (time.time() - t), flush=True)
        with open(OUT, "w") as fh:
            json.dump(allres, fh)
