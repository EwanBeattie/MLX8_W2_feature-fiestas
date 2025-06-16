import pandas as pd
import random
import json

# Replace with your actual file path
parquet_file = "data/test-00000-of-00001.parquet"

# Replace with your actual column names if different
query_column = "query"
passage_column = "passages"

# Read the Parquet file
df = pd.read_parquet(parquet_file)

# Select the first 1000 rows
df_subset = df.head(100)

# Extract queries and passages
queries = df_subset[query_column].tolist()
passages = df_subset[passage_column].tolist()


def get_triples(df_subset):
    triples = []
    random_rows = df_subset.sample(n=min(10, len(df_subset)))

    negatives = []
    for _, rand_row in random_rows.iterrows():
        passages = rand_row[passage_column]
        # If passages is a dict, wrap in list for consistency
        if isinstance(passages, dict):
            passages = [passages]
        if passages:
            chosen = random.choice(passages)
            negatives.append(chosen['passage_text'])

    for idx, row in df_subset.iterrows():
        query = row[query_column]
        passages_list = row[passage_column]
        positives = passages_list['passage_text'] if isinstance(passages_list, dict) else [p['passage_text'] for p in passages_list]
        triples.append({
            "query": query,
            "positives": positives,
            "negatives": negatives
        })
    return triples

triples = get_triples(df_subset)

print(triples)
