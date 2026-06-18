# scoring.py

def sum_of_pairs(aln):
    score = 0
    cols = zip(*aln)
    for col in cols:
        for i in range(len(col)):
            for j in range(i+1, len(col)):
                if col[i] == col[j] and col[i] != "-":
                    score += 1
    return score


def column_score(aln, ref):
    correct = 0
    total = len(ref[0])

    for c1, c2 in zip(zip(*aln), zip(*ref)):
        if c1 == c2:
            correct += 1
    return correct / total
