"""
Hard regime: semantic retrieval, ID-agnostic membership, strong paraphrase.

The lexical+ID experiment showed membership leaks trivially when records carry
unique identifiers and retrieval is character-based (any score gets AUC~1.0).
That is a *finding*, but a poor testbed for methods. Here we remove that crutch
to study the realistic, hard setting that motivates a behavioral/calibrated MIA:

  * digits (patient IDs/dates) are masked to <NUM>  -> no unique fingerprint;
    membership must be inferred from phrasing/intent, like real semantic memory.
  * retrieval = cosine over word TF-IDF (a semantic-ish stand-in for the
    SentenceTransformer retriever used in EHRAgent's cosine mode; HF is blocked
    in this sandbox so we approximate it).
  * candidates are STRONG paraphrases (word dropout + substitution): the attacker
    knows the gist, not the wording.

We compare a naive similarity-threshold MIA against calibrated variants.
"""
import json, random, argparse, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics.pairwise import cosine_similarity

STOP = set("the a an of to in on for and or is are was were has have had did do "
           "what when how many much who which that with within at by since".split())


def mask_ids(q):
    return re.sub(r"\d+", " NUM ", q)


def tokenize(q):
    return [t for t in re.findall(r"[a-z]+|NUM", q.lower())]


def paraphrase(q, drop=0.3, sub=0.2, rng=None):
    """Strong paraphrase: drop a fraction of content tokens, substitute some with
    a generic placeholder. Keeps NUM tokens (entity slots the attacker knows)."""
    toks = tokenize(q)
    out = []
    for t in toks:
        if t == "NUM" or t in STOP:
            out.append(t); continue
        r = rng.random()
        if r < drop:
            continue
        elif r < drop + sub:
            out.append("term")
        else:
            out.append(t)
    rng.shuffle(out) if rng.random() < 0.0 else None  # keep order
    return " ".join(out) if out else "query"


def tpr_at_fpr(y, s, t):
    fpr, tpr, _ = roc_curve(y, s)
    i = max(np.searchsorted(fpr, t, side="right") - 1, 0)
    return tpr[i]


def evaluate(name, y, s):
    return dict(attack=name, auc=roc_auc_score(y, s),
                tpr1=tpr_at_fpr(y, s, 0.01), tpr5=tpr_at_fpr(y, s, 0.05))


def run(mem_size=200, drop=0.3, sub=0.2, seed=42,
        path="EHRAgent/running/memory_split/500.json"):
    rng = random.Random(seed)
    qs = [mask_ids(x) for x in json.load(open(path))]
    members, nonmembers = qs[:mem_size], qs[mem_size:]

    # Public char/word vectorizer (vocabulary fit on a generic background = the
    # union; this leaks vocabulary only, not membership labels). A real attack
    # would use the agent's public embedder.
    vec = TfidfVectorizer(tokenizer=tokenize, preprocessor=lambda x: x,
                          token_pattern=None, ngram_range=(1, 2), min_df=1)
    vec.fit(qs)
    M = vec.transform(members)                      # memory matrix

    def sims(text):                                 # cosine to every member
        return cosine_similarity(vec.transform([text]), M).ravel()

    def feats(cand):
        s = np.sort(sims(cand))[::-1]
        s1, s2 = s[0], s[1] if len(s) > 1 else s[0]
        naive = s1                                  # A0: max similarity
        gap = s1 - s2                               # A1: uniqueness of best match
        density = -np.mean(s[:8])                   # local crowding (calibration)
        # A2 curvature/self-cal: re-paraphrase the candidate; a true member stays
        # anchored to ONE stored item (max-sim stable & high), a non-member's best
        # match wanders (max-sim drops more under re-paraphrase).
        drops = []
        for _ in range(6):
            drops.append(s1 - sims(paraphrase(cand, drop, sub, rng)).max())
        curv = np.mean(drops)
        return naive, gap, density, curv

    X, y = [], []
    for q in members:
        X.append(feats(paraphrase(q, drop, sub, rng))); y.append(1)
    for q in nonmembers:
        X.append(feats(paraphrase(q, drop, sub, rng))); y.append(0)
    X, y = np.array(X), np.array(y)

    res = [evaluate("A0 naive max-sim", y, X[:, 0]),
           evaluate("A1 sim-gap", y, X[:, 1]),
           evaluate("A2 curvature(self-cal)", y, -X[:, 3])]

    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xs, y):
        clf = LogisticRegression(max_iter=1000).fit(Xs[tr], y[tr])
        oof[te] = clf.predict_proba(Xs[te])[:, 1]
    res.append(evaluate("CAL combo (5-fold)", y, oof))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem_size", type=int, default=200)
    args = ap.parse_args()
    print(f"\nHARD regime: ID-masked, word-TFIDF cosine retrieval, strong paraphrase")
    print(f"Memory={args.mem_size} vs {500-args.mem_size} non-members (in-distribution)\n")
    h = f"{'drop/sub':>9} | {'attack':24} | {'AUC':>6} | {'TPR@1%':>7} | {'TPR@5%':>7}"
    print(h); print("-" * len(h))
    for drop, sub in [(0.0, 0.0), (0.3, 0.2), (0.5, 0.2)]:
        for r in run(mem_size=args.mem_size, drop=drop, sub=sub):
            print(f"{drop:.1f}/{sub:.1f}".rjust(9) +
                  f" | {r['attack']:24} | {r['auc']:6.3f} | {r['tpr1']:7.3f} | {r['tpr5']:7.3f}")
        print("-" * len(h))
