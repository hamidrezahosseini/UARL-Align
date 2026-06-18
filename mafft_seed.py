# mafft_seed.py

import subprocess
from fasta_io import write_fasta

def run_mafft(seqs, mafft_cmd):
    write_fasta(seqs, "_tmp_input.fasta")

    command = f'"{mafft_cmd}" --auto _tmp_input.fasta'

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError("MAFFT execution failed:\n" + result.stderr)

    aligned = []
    seq = ""

    for line in result.stdout.splitlines():
        if line.startswith(">"):
            if seq:
                aligned.append(seq)
                seq = ""
        else:
            seq += line.strip()

    if seq:
        aligned.append(seq)

    return aligned
