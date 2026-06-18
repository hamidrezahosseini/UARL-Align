# alignment_ops.py

import random

def shift_gap(seq, max_shift):
    seq = list(seq)
    gaps = [i for i, c in enumerate(seq) if c == "-"]
    if not gaps:
        return "".join(seq)

    pos = random.choice(gaps)
    shift = random.randint(-max_shift, max_shift)
    new_pos = max(0, min(len(seq)-1, pos + shift))

    seq.pop(pos)
    seq.insert(new_pos, "-")
    return "".join(seq)


def mutate_alignment(alignment, rate, max_shift):
    new_aln = []
    for s in alignment:
        if random.random() < rate:
            s = shift_gap(s, max_shift)
        new_aln.append(s)
    return new_aln
