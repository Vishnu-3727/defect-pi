"""Choose the eject threshold from prevalence and cost, not from F1.

    python scripts/operating_point.py --prevalence 0.05 --cost-ratio 30

WHY THIS EXISTS
The usual recipe is "take the peak of the F1-vs-confidence curve". F1 is the
harmonic mean of precision and recall computed **on the labelled set**, so on a
defect-only benchmark it is conditioned on every part being defective. It
therefore cannot express the cost of rejecting a good one, and on NEU-DET
-- where prevalence is 1.0 -- no threshold is ever penalised for over-rejecting.

Measured on KolektorSDD2, which has real defect-free images, the cost-optimal
threshold at prevalence 1.0 is 0.01 for *every* cost ratio including 1:1, while
at the real 11% prevalence the same model wants 0.35. A factor of 35, decided by
a quantity F1 never sees. See reports/ksdd2-prevalence.md.

WHAT THIS NEEDS FROM YOU
Two numbers nobody but the line owner can supply:

  --prevalence   fraction of parts that are actually defective
  --cost-ratio   how many false rejects one escape is worth

Neither is in any dataset. That is the point: the operating point is not a
property of the model, and a benchmark cannot hand it to you.

WHAT IT CANNOT DO
Invent an overkill rate. Without defect-free images the overkill column is
unmeasurable, and this script will say so rather than guess. The built-in
NEU-DET curve uses the region proxy from reports/overkill-and-prevalence.md and
is an UPPER bound; the KSDD2 curve is measured on real negatives.
"""
from __future__ import annotations

import argparse

# escape / overkill measured per dataset. thr -> (escape, overkill)
CURVES = {
    # reports/overkill-and-prevalence.md -- overkill from defect-free REGIONS,
    # so an upper bound rather than a measurement.
    "neu-det": {
        0.35: (0.088, 0.234), 0.25: (0.044, 0.319), 0.15: (0.017, 0.468),
        0.10: (0.006, 0.596), 0.05: (0.000, 0.702),
    },
    # reports/ksdd2-prevalence.md -- both rates measured on real images.
    "ksdd2": {
        0.50: (0.282, 0.006), 0.35: (0.164, 0.017), 0.25: (0.118, 0.043),
        0.15: (0.082, 0.111), 0.10: (0.064, 0.195), 0.05: (0.036, 0.322),
        0.01: (0.009, 0.706),
    },
}
UPPER_BOUND = {"neu-det"}


def expected_cost(escape: float, overkill: float, p: float, r: float) -> float:
    """Cost per inspected part, in units of one false reject."""
    return p * escape * r + (1 - p) * overkill


def recommend(curve: dict, p: float, r: float):
    scored = {t: expected_cost(e, o, p, r) for t, (e, o) in curve.items()}
    best = min(scored, key=scored.get)
    return best, scored


def f1_optimal(curve: dict):
    """What the conventional recipe picks: best F1 *on the defective set*.

    Recall = 1 - escape. Precision is approximated from the overkill rate at the
    benchmark's own prevalence, which for a defect-only set is 1.0 -- so
    precision is 1.0 everywhere and F1 reduces to recall. That is exactly the
    degeneracy this script exists to expose: on a defect-only benchmark, "tune
    for F1" and "tune for recall" are the same instruction.
    """
    best, best_f1 = None, -1.0
    for t, (e, _) in curve.items():
        recall = 1 - e
        f1 = recall          # precision == 1 when prevalence == 1
        if f1 > best_f1:
            best, best_f1 = t, f1
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="ksdd2", choices=sorted(CURVES))
    ap.add_argument("--prevalence", type=float, required=True,
                    help="fraction of parts actually defective, e.g. 0.05")
    ap.add_argument("--cost-ratio", type=float, required=True,
                    help="how many false rejects one escape is worth")
    args = ap.parse_args()

    if not 0 < args.prevalence <= 1:
        raise SystemExit("--prevalence must be in (0, 1]")
    if args.cost_ratio <= 0:
        raise SystemExit("--cost-ratio must be positive")

    curve = CURVES[args.dataset]
    best, scored = recommend(curve, args.prevalence, args.cost_ratio)
    conventional = f1_optimal(curve)

    print(f"\ncurve: {args.dataset}"
          + ("   [overkill is an UPPER BOUND -- region proxy, no real negatives]"
             if args.dataset in UPPER_BOUND else "   [both rates measured]"))
    print(f"prevalence {args.prevalence}   cost ratio {args.cost_ratio}\n")
    print(f"{'thr':>6}{'escape':>9}{'overkill':>10}{'E[cost]':>11}")
    print("-" * 36)
    for t in sorted(curve, reverse=True):
        e, o = curve[t]
        mark = "  <-- recommended" if t == best else ""
        print(f"{t:>6.2f}{e:>9.3f}{o:>10.3f}{scored[t]:>11.4f}{mark}")

    print(f"\nprevalence-aware recommendation : {best}")
    print(f"conventional F1 recipe          : {conventional}")
    if best != conventional:
        ce = expected_cost(*curve[conventional], args.prevalence, args.cost_ratio)
        cb = scored[best]
        print(f"\nthe F1 recipe costs {ce / cb:.1f}x more per part at this prevalence.")
        print("On a defect-only benchmark precision is 1 by construction, so F1")
        print("reduces to recall and always prefers the lowest threshold offered.")
    else:
        print("\nboth agree here.")


if __name__ == "__main__":
    main()
