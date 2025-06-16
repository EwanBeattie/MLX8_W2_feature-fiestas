import pandas as pd
import random
import json
from tokeniser import tokenise  # Import the tokeniser

# Replace with your actual file path
parquet_file = "data/test-00000-of-00001.parquet"

# Read the Parquet file
df = pd.read_parquet(parquet_file)

# Replace with your actual column names if different
query_column = "query"
passage_column = "passages"

# Read the Parquet file
df = pd.read_parquet(parquet_file)

# Select the first 1000 rows
df_subset = df.head(100)

# Extract queries and passages
queries = df_subset[query_column].tolist()
passage_info = df_subset[passage_column].tolist()

all_passages = []
for passages in passage_info:
    all_passages.extend(passages['passage_text'])
all_passages = list(set(all_passages))  # Remove duplicates

triples = []

for idx, row in df_subset.iterrows():
    query = row[query_column]
    passages_list = row[passage_column]
    positives = passages_list['passage_text']

    # Select 10 random negatives that are not in positives
    negatives_pool = list(set(all_passages) - set(positives))
    negatives = random.sample(negatives_pool, min(10, len(negatives_pool)))
    triples.append({
        "query": tokenise(query),
        "positives": [tokenise(p) for p in positives],
        "negatives": [tokenise(n) for n in negatives],
    })

# print(triples[0])
