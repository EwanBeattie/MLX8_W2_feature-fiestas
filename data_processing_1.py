import pandas as pd
import random
import json

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
df_subset = df.head(1)

# Extract queries and passages
queries = df_subset[query_column].tolist()
passage_info = df_subset[passage_column].tolist()

all_passages = list(passage_info[0]['passage_text'])

triples = []

for idx, row in df_subset.iterrows():
    query = row[query_column]
    passages_list = row[passage_column]
    positives = passages_list['passage_text']
    triples.append({
        "query": query,
        "positives": positives,
        "negatives": 
    })

