import json, sys, numpy as np
res = []
for f in sys.argv[1:]:
    res += json.load(open(f))
seeds = sorted({r["seed"] for r in res})
print("seeds", seeds)


def bcipher(cm):
    cm = np.array(cm)  # classes AES,DES,3DES,RC4,None
    blk = cm[:3, :3].trace() / cm[:3].sum()
    # None vs encrypted
    ne = (cm[4, 4] + cm[:4, :4].sum()) / cm.sum()
    # RC4 vs block among encrypted samples
    enc = cm[:4, :4]
    rb = (enc[3, 3] + enc[:3, :3].sum()) / cm[:4].sum()
    return blk, ne, rb


def get(mode, k, proto, model=None):
    out = []
    for s in seeds:
        r = [x for x in res if x["seed"] == s and x["mode"] == mode and x["keys"] == k and x["protocol"] == proto][0]
        m = model or r["best"]
        d = r["models"][m]
        out.append((s, m, d["test"], d["f1"], *bcipher(d["cm"])))
    return out


summary = {}
for mode in ["ECB", "CBC"]:
    for k in [1, 3, 6]:
        row = {}
        for proto in ["shared", "disjoint"]:
            g = get(mode, k, proto)
            acc = np.array([x[2] for x in g]); f1 = np.array([x[3] for x in g])
            blk = np.array([x[4] for x in g]); ne = np.array([x[5] for x in g]); rb = np.array([x[6] for x in g])
            s42 = [x for x in g if x[0] == 42][0]
            row[proto] = dict(s42_model=s42[1], s42_acc=s42[2], mean=acc.mean(), sd=acc.std(ddof=1),
                              f1=f1.mean(), blk=blk.mean(), blk_sd=blk.std(ddof=1), ne=ne.mean(), rb=rb.mean(),
                              accs=acc.tolist())
            for mdl in ["RF", "Bagging"]:
                a = np.array([x[2] for x in get(mode, k, proto, mdl)])
                row[proto][mdl] = (a.mean(), a.std(ddof=1))
        drops = np.array(row["shared"]["accs"]) - np.array(row["disjoint"]["accs"])
        row["drop_mean"] = drops.mean(); row["drop_sd"] = drops.std(ddof=1)
        row["n_drop_pos"] = int((drops > 0).sum())
        summary[f"{mode}-{k}"] = row
        sh, dj = row["shared"], row["disjoint"]
        print(f"{mode} {k}k | shared s42 {sh['s42_model']} {sh['s42_acc']*100:.1f} mean {sh['mean']*100:.1f}±{sh['sd']*100:.1f} "
              f"| disjoint s42 {dj['s42_model']} {dj['s42_acc']*100:.1f} mean {dj['mean']*100:.1f}±{dj['sd']*100:.1f} "
              f"| drop {row['drop_mean']*100:.1f}±{row['drop_sd']*100:.1f} ({row['n_drop_pos']}/{len(seeds)}) "
              f"| Bcipher sh {sh['blk']*100:.1f}±{sh['blk_sd']*100:.1f} dj {dj['blk']*100:.1f}±{dj['blk_sd']*100:.1f} "
              f"| NE {sh['ne']*100:.1f}/{dj['ne']*100:.1f} RB {sh['rb']*100:.1f}/{dj['rb']*100:.1f} "
              f"| RF {sh['RF'][0]*100:.1f}/{dj['RF'][0]*100:.1f} Bag {sh['Bagging'][0]*100:.1f}/{dj['Bagging'][0]*100:.1f}")
json.dump(summary, open("summary.json", "w"), indent=1)
