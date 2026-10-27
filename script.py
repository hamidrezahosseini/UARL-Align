
import config as cfg

from uncertainty import column_uncertainties
from mutations import select_target_columns, adaptive_mutation
from eo_optimizer import _validate_candidate


alignment = [
    "AC-GT",
    "A-TGT",
    "ACCGT",
    "AT-GT",
]

uncertainty = column_uncertainties(
    alignment,
    sequence_type="dna",
)

targets = select_target_columns(
    uncertainty,
    cfg.UNCERTAINTY_HIGH,
)

print("Column uncertainty:", uncertainty)
print("Selected high-uncertainty columns:", targets)

candidate = adaptive_mutation(
    alignment,
    uncertainty,
    cfg,
    phase="explore",
)

_validate_candidate(alignment, candidate)

print("Original alignment:")
for seq in alignment:
    print(seq)

print("Candidate alignment:")
for seq in candidate:
    print(seq)

assert len(candidate) == len(alignment)
assert len({len(seq) for seq in candidate}) == 1

for old_seq, new_seq in zip(alignment, candidate):
    assert old_seq.replace("-", "") == new_seq.replace("-", "")

print("PASS: sequence order and alignment dimensions are preserved.")
