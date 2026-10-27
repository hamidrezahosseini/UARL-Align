
import math
from collections import Counter


DNA_ALPHABET = set("ACGT")
RNA_ALPHABET = set("ACGU")
PROTEIN_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")

# Ambiguous nucleotide symbols are recognized for type detection but
# are excluded from entropy calculations.
NUCLEOTIDE_SYMBOLS = set("ACGTUNRYKMSWBDHV")


def _validate_alignment(aln):
    if not aln:
        raise ValueError("Alignment must contain at least one sequence.")

    lengths = {len(seq) for seq in aln}
    if len(lengths) != 1:
        raise ValueError("All aligned sequences must have the same length.")

    return len(aln[0])


def infer_sequence_type(aln):
    """
    Infer DNA, RNA, or protein from observed symbols.

    For reliable classification, set SEQUENCE_TYPE explicitly in config.py
    when the biological sequence type is known.
    """
    observed = {
        char.upper()
        for seq in aln
        for char in seq
        if char not in "-."
    }

    if not observed:
        raise ValueError("No residues were found in the alignment.")

    if observed.issubset(NUCLEOTIDE_SYMBOLS):
        if "U" in observed and "T" not in observed:
            return "rna"
        return "dna"

    if observed.issubset(PROTEIN_ALPHABET | {"X", "B", "Z", "J"}):
        return "protein"

    raise ValueError(
        "Could not infer sequence type from the alignment. "
        "Set SEQUENCE_TYPE explicitly and check for unsupported symbols."
    )


def get_alphabet(sequence_type):
    """Return the canonical alphabet used for entropy normalization."""
    sequence_type = sequence_type.lower()

    if sequence_type in {"dna", "rna"}:
        return DNA_ALPHABET

    if sequence_type == "protein":
        return PROTEIN_ALPHABET

    raise ValueError(
        "sequence_type must be 'dna', 'rna', or 'protein'."
    )


def _canonical_residue(char, sequence_type):
    """Return a canonical residue, or None for gaps/ambiguous symbols."""
    char = char.upper()

    if char in {"-", "."}:
        return None

    if sequence_type == "rna" and char == "U":
        return "T"

    alphabet = get_alphabet(sequence_type)
    return char if char in alphabet else None


def column_uncertainty(col, sequence_type="auto"):
    """
    Calculate normalized Shannon entropy for one alignment column.

    Gaps and ambiguous/non-canonical symbols are excluded.
    The result is in [0, 1] for columns with informative residues.
    """
    if sequence_type == "auto":
        sequence_type = infer_sequence_type(["".join(col)])

    residues = [
        residue
        for char in col
        if (residue := _canonical_residue(char, sequence_type)) is not None
    ]

    if not residues:
        return 0.0

    counts = Counter(residues)
    total = len(residues)

    entropy = -sum(
        (count / total) * math.log(count / total)
        for count in counts.values()
    )

    alphabet_size = len(get_alphabet(sequence_type))
    max_entropy = math.log(alphabet_size)

    return entropy / max_entropy if max_entropy > 0 else 0.0


def column_uncertainties(aln, sequence_type="auto"):
    """Return normalized entropy for every alignment column."""
    n_cols = _validate_alignment(aln)

    if sequence_type == "auto":
        sequence_type = infer_sequence_type(aln)

    return [
        column_uncertainty(
            [seq[j] for seq in aln],
            sequence_type=sequence_type,
        )
        for j in range(n_cols)
    ]


def alignment_uncertainty(aln, sequence_type="auto"):
    """
    Mean normalized entropy over columns containing at least two
    canonical, informative residues.
    """
    n_cols = _validate_alignment(aln)

    if sequence_type == "auto":
        sequence_type = infer_sequence_type(aln)

    eligible_uncertainties = []

    for j in range(n_cols):
        col = [seq[j] for seq in aln]
        informative_count = sum(
            _canonical_residue(char, sequence_type) is not None
            for char in col
        )

        if informative_count >= 2:
            eligible_uncertainties.append(
                column_uncertainty(col, sequence_type)
            )

    if not eligible_uncertainties:
        return 0.0

    return sum(eligible_uncertainties) / len(eligible_uncertainties)
