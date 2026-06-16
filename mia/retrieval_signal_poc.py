"""
Proof-of-concept: retrieval-layer membership signal for LLM-agent memory (EHRAgent).

This validates the *necessary condition* behind a behavioral MIA on agent memory:
under the agent's real retrieval mechanism (edit-distance over stored questions,
faithful to EHRAgent's default `retrieve_method=edit_distance`), is "this record
is in memory" detectable, and does difficulty-CALIBRATION beat a naive distance
threshold -- especially when the attacker only holds a *paraphrase* of the
candidate (the realistic agent-memory threat)?

We do NOT call any LLM or DB here. We reproduce only the retriever, which is the
layer that converts membership into downstream behavioral leakage.

Outputs: ROC-AUC and TPR@{1%,5%}FPR for naive vs. calibrated attacks, swept over
paraphrase/"phrasing-drift" levels.
"""
import json, random, argparse
import Levenshtein
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

ALPHA = "abcdefghijklmnopqrstuvwxyz"


def load_questions(path):
    return json.load(open(path))


def perturb(q, rate, rng):
    """Simulate phrasing drift: edit a fraction `rate` of ALPHABETIC chars
    (substitute/insert/delete). Digits (patient IDs / dates = the sensitive
    entities the attacker is assumed to know) are preserved. rate=0 -> identity."""
    if rate <= 0:
        return q
    out = []
    for ch in q:
        if ch.isalpha() and rng.random() < rate:
            r = rng.random()
            if r < 0.34:                       # substitute
                out.append(rng.choice(ALPHA))
            elif r < 0.67:                     # insert then keep
                out.append(rng.choice(ALPHA)); out.append(ch)
            # else: delete (append nothing)
        else:
            out.append(ch)
    return "".join(out)


def dist_vec(cand, mem):
    return np.array([Levenshtein.distance(cand, m) for m in mem], dtype=float)


def features(cand, mem, rng, n_pert=8, pert_rate=0.10):
    """Retrieval-layer features observable to a gray-box attacker (distances)
    or *induced* in behavior for a black-box attacker."""
    d = dist_vec(cand, mem)
    d_sorted = np.sort(d)
    L = max(len(cand), 1)
    d1 = d_sorted[0]
    d2 = d_sorted[1] if len(d_sorted) > 1 else d_sorted[0]
    # A0 naive: closeness to nearest stored item (length-normalized)
    f_naive = -d1 / L
    # A1 NN-gap: a true (even paraphrased) member has ONE uniquely-closest stored
    #            item -> gap; a non-member in a dense template region has d1~=d2.
    f_gap = (d2 - d1) / L
    # local corpus density (calibration context): how crowded is this region
    f_density = -np.mean(np.sort(d)[:8]) / L
    # A2 curvature / self-calibration: perturb the candidate; a member sits in a
    #    sharp retrieval basin (distance rises under perturbation), a non-member's
    #    neighborhood is comparatively flat.
    base = d1
    rises = []
    for _ in range(n_pert):
        dp = dist_vec(perturb(cand, pert_rate, rng), mem).min()
        rises.append(dp - base)
    f_curv = np.mean(rises) / L
    return dict(naive=f_naive, gap=f_gap, density=f_density, curv=f_curv)


def tpr_at_fpr(y, s, fpr_target):
    fpr, tpr, _ = roc_curve(y, s)
    idx = np.searchsorted(fpr, fpr_target, side="right") - 1
    idx = max(idx, 0)
    return tpr[idx]


def evaluate(name, y, s):
    auc = roc_auc_score(y, s)
    return dict(attack=name, auc=auc,
                tpr1=tpr_at_fpr(y, s, 0.01), tpr5=tpr_at_fpr(y, s, 0.05))


def run(mem_size=200, drift=0.0, seed=42,
        path="EHRAgent/running/memory_split/500.json"):
    rng = random.Random(seed)
    qs = load_questions(path)
    members = qs[:mem_size]                       # in memory
    nonmembers = qs[mem_size:]                    # held out, same distribution
    mem_questions = list(members)

    rows, y = [], []
    for q in members:
        rows.append(features(perturb(q, drift, rng), mem_questions, rng)); y.append(1)
    for q in nonmembers:
        rows.append(features(perturb(q, drift, rng), mem_questions, rng)); y.append(0)
    y = np.array(y)

    feat = {k: np.array([r[k] for r in rows]) for k in rows[0]}

    results = []
    results.append(evaluate("A0 naive distance", y, feat["naive"]))
    results.append(evaluate("A1 NN-gap", y, feat["gap"]))
    results.append(evaluate("A2 curvature(self-cal)", y, feat["curv"]))

    # CALIBRATED combo: logistic over all features, 5-fold out-of-fold scores
    X = np.column_stack([feat["naive"], feat["gap"], feat["density"], feat["curv"]])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    results.append(evaluate("CAL combo (5-fold)", y, oof))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem_size", type=int, default=200)
    ap.add_argument("--path", type=str, default="EHRAgent/running/memory_split/500.json")
    args = ap.parse_args()

    print(f"\nMemory={args.mem_size} members vs {500-args.mem_size} non-members "
          f"(in-distribution split) | retriever=edit_distance (EHRAgent default)\n")
    header = f"{'drift':>6} | {'attack':24} | {'AUC':>6} | {'TPR@1%':>7} | {'TPR@5%':>7}"
    print(header); print("-" * len(header))
    for drift in [0.0, 0.10, 0.20, 0.30]:
        for r in run(mem_size=args.mem_size, drift=drift, path=args.path):
            print(f"{drift:6.2f} | {r['attack']:24} | {r['auc']:6.3f} | "
                  f"{r['tpr1']:7.3f} | {r['tpr5']:7.3f}")
        print("-" * len(header))
