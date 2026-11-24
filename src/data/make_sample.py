# src/data/make_sample.py
# Run from anywhere:  python src/data/make_sample.py
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Build paths from this file's location, not from the current folder.
# parents[2] = project root (Movie-Rec), since this file is src/data/make_sample.py
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "raw" / "ratings.parquet"
OUT = ROOT / "data" / "processed" / "ratings_sample.parquet"

N_USERS = 3000   # how many users to keep
SEED = 42        # fixed seed -> same sample every run

pf = pq.ParquetFile(SRC)
n_groups = pf.metadata.num_row_groups
print("rows:", pf.metadata.num_rows, "| row groups:", n_groups)

# Pass 1: read ONLY the user_id column, one row group at a time,
# and collect the distinct user IDs found in each group.
chunks = []
for i in range(n_groups):
    col = pf.read_row_group(i, columns=["user_id"]).column("user_id")
    chunks.append(pc.unique(col).to_numpy())
users = np.unique(np.concatenate(chunks))      # sorted, no duplicates
print("unique users:", len(users))

# Choose N_USERS at random. Same seed -> same users every time.
rng = np.random.default_rng(SEED)
chosen = rng.choice(users, size=min(N_USERS, len(users)), replace=False)
keep = pa.array(chosen, type=pa.uint32())      # match the column's type

# Pass 2: keep every rating belonging to a chosen user.
parts = []
for i in range(n_groups):
    t = pf.read_row_group(i)                   # all 3 columns this time
    parts.append(t.filter(pc.is_in(t["user_id"], value_set=keep)))
sample = pa.concat_tables(parts)

OUT.parent.mkdir(parents=True, exist_ok=True)
pq.write_table(sample, OUT)
print("sample rows:", sample.num_rows, "->", OUT)
print("distinct users in sample:", len(pc.unique(sample["user_id"])))
