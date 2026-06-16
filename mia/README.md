# Membership Inference Attacks (MIA) against LLM-Agent Memory

This directory proposes and prototypes a **membership inference attack** for LLM
agent memory, as a complement to the **extraction** attack (MEXTRA) in
*"Unveiling Privacy Risks in LLM Agent Memory"* (this repo).

- **MEXTRA (existing):** *dump the memory* — recover the stored records.
- **MIA (this work):** *targeted confirmation* — given one candidate record,
  decide whether it is stored in the agent's memory. Weaker assumption, harder
  to defend, and still a real privacy violation (e.g. "is *this patient's* query
  in the hospital agent's memory?"). MIA works **even when extraction is blocked**
  (output filtering, refusal to repeat examples), which is its main selling point.

## Why agent memory needs its own MIA (not just RAG-doc MIA)

Agent memory is **non-parametric** (a retrieval bank, not model weights), so the
classic loss/perplexity MIA does not apply directly. Two properties make it
different from plain RAG-text MIA and create new signal:

1. Records are **(question, knowledge, code/trajectory)** triples — *experiences*,
   not passages.
2. The agent **executes** the retrieved plan and exposes rich behavior:
   first-try success, #debug retries, plan/code stability, self-consistency.

## Proposed method (flagship): behavioral, self-calibrated MIA

**Idea.** If a record is a member, a probe near its stored phrasing retrieves the
record itself, so the agent *copies a gold trajectory*: it succeeds on the first
try, with stable code and high self-consistency. If it is a non-member, the agent
solves from scratch: more retries, more variance. Membership ⇒ a **competence
cliff** between the *exact* probe and *paraphrase* probes; non-members are flat.

**Signals (unique to agents, output/gray-box):**
- first-try execution success / #debug iterations,
- code & plan similarity across stochastic re-runs (self-consistency entropy),
- agreement of the final answer across paraphrase probes.

**Self-calibration (no shadow models):** score = behavior(exact probe) −
behavior(paraphrase probes). Using the agent as its own reference cancels the #1
MIA confounder — intrinsic task difficulty. Complemented by the retrieval
**uniqueness/gap** signal validated below.

## What is prototyped here (retrieval layer)

The full behavioral attack needs the live agent (OpenAI API + MIMIC-III DB),
unavailable in this sandbox. We therefore validate the **necessary condition** —
the retrieval layer that turns membership into behavioral leakage — faithfully
reproducing EHRAgent's retriever on the real memory questions.

- `retrieval_signal_poc.py` — **edit-distance** retriever (EHRAgent default).
- `semantic_signal_poc.py` — **cosine/TF-IDF** retriever, ID-masked, strong
  paraphrase (the hard, realistic regime; HF blocked here so TF-IDF stands in
  for the SentenceTransformer embedder).

Run: `python3 mia/retrieval_signal_poc.py` / `python3 mia/semantic_signal_poc.py`
(in-distribution member/non-member split from `memory_split/500.json`).

## Key empirical findings

**1. Lexical + unique IDs ⇒ membership leaks trivially.** Under edit-distance
retrieval, even with 30% character noise on the probe, AUC ≈ **1.000**,
TPR@1%FPR ≈ **1.0**. Patient IDs/dates act as near-perfect fingerprints. Strong
motivation: agent memory *is* vulnerable.

**2. Semantic + strong paraphrase + no ID anchor ⇒ the real research problem.**
Naive max-similarity MIA degrades as phrasing drifts:

| paraphrase (drop/sub) | naive AUC | naive TPR@1% |
|---|---|---|
| 0.0 / 0.0 (exact) | 1.000 | 1.000 |
| 0.3 / 0.2 | 0.872 | 0.455 |
| 0.5 / 0.2 | 0.795 | 0.310 |

**3. Calibration helps.** The retrieval **uniqueness/gap** (best vs. 2nd-best
match) and a small calibrated combo beat the naive threshold, e.g. at drop=0.3:

| attack | AUC | TPR@1% |
|---|---|---|
| A0 naive max-sim | 0.872 | 0.455 |
| A1 sim-gap (calibrated) | 0.888 | 0.510 |
| CAL combo (5-fold) | **0.906** | **0.535** |

**4. Larger memory ⇒ harder MIA** (more semantic neighbors); calibration's edge
grows with size (m=200: 0.906 vs 0.872; m=400: 0.871 vs 0.863).

These confirm the mechanism and the value of calibration at the retrieval layer.
The behavioral channel (success/retries/self-consistency) should add
substantially more signal and is the next experiment (needs API + DB).

## Roadmap to the full attack
1. Wire `MedAgent` with API + MIMIC-III; log per-probe `successful_flag`,
   retry count, `code`, `knowledge`, final answer.
2. For each candidate: 1 exact probe + N paraphrase probes (LLM paraphrases),
   T stochastic re-runs each.
3. Features = behavioral cliff + self-consistency + retrieval gap → AUC, TPR@low-FPR.
4. Baselines: naive similarity, RAG-MIA (S²MIA/MBA-style) adapted; ablate signals.
5. Defenses: output filtering, paraphrase-normalized memory keys, dummy records,
   k-anonymized memory — show MIA survives where extraction fails.
