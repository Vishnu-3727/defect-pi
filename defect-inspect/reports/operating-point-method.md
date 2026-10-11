# Choosing the threshold: a method, and why F1 cannot do it

`ksdd2-prevalence.md` showed prevalence decides the operating point. This turns
that into a procedure, and states precisely why the conventional recipe fails.

Tool: `scripts/operating_point.py`.

---

## The formal problem with "take the peak of the F1 curve"

That recipe appears in this project's own docs and across the defect-inspection
literature. For a PASS/FAIL ejector it is degenerate.

The decision is image-level: **is this part defective?** On a defect-only
benchmark every image *is* defective, so any image the model flags is a true
positive and **precision = 1 at every threshold**. Then

```
F1 = 2PR/(P+R) = 2R/(1+R)
```

which is **strictly increasing in recall**. Maximising F1 is therefore identical
to maximising recall, and recall is maximised by the lowest threshold available.

**On a defect-only benchmark, "tune for F1" and "use the lowest threshold" are
the same instruction.** The recipe is not merely imprecise; it carries no
information about the operating point at all.

This applies to the image-level decision. Box-level F1 is not degenerate, since
localisation error keeps precision below 1 — but box-level F1 is not what an
ejector optimises, and using it for the eject decision is the conflation
recorded in `threshold-decoupling.md`.

## The procedure

Expected cost per inspected part, in units of one false reject:

```
E[cost](t) = p · escape(t) · R  +  (1 − p) · overkill(t)
```

with `p` the defect prevalence and `R = C_escape / C_overkill`. Pick the `t`
that minimises it. Two inputs are required that **no dataset contains**:

- `p` — the fraction of parts actually defective on the line;
- `R` — how many good parts you would scrap to avoid one escape.

That is the substantive claim: the operating point is not a property of the
model, and no benchmark can supply it. A paper reporting "we selected conf=0.25
from the F1 curve" has reported its dataset's prevalence, not a design decision.

## What it recommends, on the measured KSDD2 curve

| prevalence | R | prevalence-aware | F1 recipe |
|---|--:|--:|--:|
| 0.01 | 100 | 0.25 | 0.01 |
| **0.05** | **30** | **0.25** | 0.01 |
| 0.11 | 10 | 0.25 | 0.01 |
| 0.50 | 10 | 0.05 | 0.01 |
| 1.00 | 10 | 0.01 | 0.01 |

The F1 column is constant by construction — it is always the lowest threshold
offered. The two agree **only at prevalence 1.00**, the defect-only regime,
which is precisely where the degeneracy makes F1 vacuous rather than correct.

At a realistic 5% prevalence with R = 30, the F1 recipe costs **3.1× more per
part** than the cost-optimal choice.

## Honest limits, enforced in the tool

- It **will not invent an overkill rate.** The NEU-DET curve is flagged in the
  output as an upper bound, because its overkill comes from the defect-free
  region proxy; the KSDD2 curve is flagged as measured.
- It interpolates nothing: it scores only thresholds that were actually
  measured. A finer grid needs more evaluation, not more arithmetic.
- `p` and `R` are the line owner's numbers. The tool refuses to default them,
  since a default would reintroduce exactly the silent assumption it exists to
  remove.
- The cost model is linear and ignores rework, batching, and the possibility
  that an escape's cost depends on which defect escaped. Those are real and not
  modelled.
