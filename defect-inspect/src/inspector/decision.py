"""Turn detections and an anomaly score into one bit: eject, or do not.

This is the smallest file in the project and the one a factory would argue
about longest, because it encodes a business decision, not a technical one:
what counts as bad enough to throw away.

Keeping it separate from the detector matters. The detector answers "what do I
see". This answers "what do we do about it". Those change for different reasons
and at different times -- a new customer specification changes this file and
nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

from .detector import Detection

PASS = "PASS"
FAIL = "FAIL"


@dataclass(frozen=True)
class Verdict:
    verdict: str
    reason: str
    defect_class: str | None
    confidence: float | None
    n_defects: int
    anomaly_score: float | None

    @property
    def is_fail(self) -> bool:
        return self.verdict == FAIL


def decide(detections: list[Detection], anomaly_score: float | None,
           min_area_px: int, anomaly_threshold: float) -> Verdict:
    """Fuse the two models. Either one alone can reject a part.

    The fusion is OR, not AND, and that is the safety-critical choice in this
    project. The two models fail in different ways: YOLOv8 misses defect types
    it was never taught, PatchCore cannot name what it finds. Requiring both to
    agree would mean the system only catches defects both can see, which is
    strictly worse than either alone. Requiring either means a missed defect
    needs both to fail at once.

    The cost of OR is a higher false-reject rate, and for surface inspection
    that is the cheaper error: re-inspecting a good part costs seconds, shipping
    a defective one costs a customer.
    """
    # Tiny boxes are usually the detector reacting to sensor noise or a speck of
    # dust, not a defect. Filtering by area is cruder than raising the
    # confidence threshold but it removes a failure mode confidence does not:
    # a small, high-confidence false positive.
    real = [d for d in detections if d.area >= min_area_px]

    if real:
        worst = max(real, key=lambda d: d.confidence)
        reason = f"{worst.class_name} {worst.confidence:.2f}"
        if len(real) > 1:
            reason += f" (+{len(real) - 1} more)"
        return Verdict(FAIL, reason, worst.class_name, worst.confidence,
                       len(real), anomaly_score)

    if anomaly_score is not None and anomaly_score >= anomaly_threshold:
        # PatchCore knows the surface is not normal but cannot say why, so the
        # class stays NULL. The operator sees "unclassified anomaly" and looks.
        return Verdict(FAIL, f"unclassified anomaly {anomaly_score:.2f}",
                       None, None, 0, anomaly_score)

    suppressed = len(detections) - len(real)
    reason = "no defects"
    if suppressed:
        reason = f"no defects ({suppressed} below {min_area_px} px suppressed)"
    return Verdict(PASS, reason, None, None, 0, anomaly_score)


def demo() -> None:
    """Self-check every branch, including the ones that are easy to invert."""
    big = Detection(5, "scratches", 0.91, (10, 10, 60, 60))       # 2500 px
    bigger = Detection(2, "patches", 0.95, (0, 0, 100, 100))      # 10000 px
    tiny = Detection(0, "crazing", 0.99, (10, 10, 13, 13))        # 9 px

    v = decide([], None, 40, 2.5)
    assert v.verdict == PASS and v.n_defects == 0, v

    v = decide([big], None, 40, 2.5)
    assert v.verdict == FAIL and v.defect_class == "scratches", v

    # A tiny box is suppressed even at 0.99 confidence, and the reason says so.
    v = decide([tiny], None, 40, 2.5)
    assert v.verdict == PASS, v
    assert "suppressed" in v.reason, v.reason

    # The reported class is the highest-confidence surviving box, not the first.
    v = decide([big, bigger], None, 40, 2.5)
    assert v.defect_class == "patches" and v.n_defects == 2, v
    assert "+1 more" in v.reason, v.reason

    # Anomaly alone can fail a part, with no class attached.
    v = decide([], 3.2, 40, 2.5)
    assert v.verdict == FAIL and v.defect_class is None, v
    assert v.anomaly_score == 3.2

    # Below threshold it does not.
    v = decide([], 1.0, 40, 2.5)
    assert v.verdict == PASS, v

    # A detection wins over a quiet anomaly score: the named reason is better
    # for the operator than "something is odd".
    v = decide([big], 0.1, 40, 2.5)
    assert v.verdict == FAIL and v.defect_class == "scratches", v
    assert v.anomaly_score == 0.1, "the score is still recorded even when unused"

    print("decision self-check passed: OR fusion, area suppression, "
          "highest-confidence class, anomaly-only failure")


if __name__ == "__main__":
    demo()
