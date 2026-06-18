# confidence.py

import math
from collections import Counter

def column_entropy(column):
    ent = 0.0
    total = len(column)
    counts = Counter(column)
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


def column_confidence(aln, seed, threshold=0.5):
    confident = []
    ambiguous = []

    for idx, (c1, c2) in enumerate(zip(zip(*aln), zip(*seed))):
        ent = column_entropy(c1)
        agreement = sum(a == b for a, b in zip(c1, c2)) / len(c1)

        confidence = agreement * (1 / (1 + ent))

        if confidence >= threshold:
            confident.append(idx)
        else:
            ambiguous.append(idx)

    return confident, ambiguous
