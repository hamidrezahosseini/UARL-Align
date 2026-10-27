# config.py

MAFFT_CMD = r"C:\Program Files (x86)\ClustalW2\mafft-win\mafft.bat"

# Set explicitly to "dna", "rna", or "protein" when known.
# "auto" infers the type from the observed sequence symbols.
SEQUENCE_TYPE = "auto"

# ---------- RL ----------
RL_LR = 0.005
N_EPISODES = 60
PHASE_SPLIT = 0.4   # 40% explore, 60% exploit

# ---------- Reward weights (initial) ----------
ALPHA = 1.0   # SP gain
BETA  = 0.6   # uncertainty reduction
GAMMA = 0.4   # gap penalty (exploit phase only)

# ---------- EO / Mutation ----------
POP_SIZE = 150
MAX_ITERS = 500
RANDOM_SEED = None

# ---------- Explore phase ----------
EXPLORE_MAX_SHIFT = 14
EXPLORE_BLOCK_RANGE = (5, 10)
EXPLORE_SWAP_WINDOW = 10

# ---------- Exploit phase ----------
EXPLOIT_MAX_SHIFT = 6
EXPLOIT_BLOCK_RANGE = (3, 6)
EXPLOIT_SWAP_WINDOW = 4

# ---------- Column uncertainty ----------
UNCERTAINTY_HIGH = 0.55
UNCERTAINTY_LOW  = 0.35
