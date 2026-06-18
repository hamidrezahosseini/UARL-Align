# fasta_io.py

def read_fasta(path):
    seqs = []
    with open(path) as f:
        seq = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq:
                    seqs.append(seq)
                    seq = ""
            else:
                seq += line
        if seq:
            seqs.append(seq)
    return seqs


def write_fasta(seqs, path):
    with open(path, "w") as f:
        for i, s in enumerate(seqs):
            f.write(f">seq{i}\n{s}\n")
