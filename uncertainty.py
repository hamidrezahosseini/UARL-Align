# uncertainty.py

import math
from collections import Counter

def alignment_uncertainty(aln):
    ent = 0.0
    for col in zip(*aln):
        counts = Counter(col)
        total = len(col)
        for c in counts.values():
            p = c / total
            ent -= p * math.log2(p)
    return ent / len(aln[0])
