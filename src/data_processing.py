import pandas as pd
import random
import json
import logging
import pickle
from tokensier import tokenise
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)

# Replace with your actual file path
parquet_file = "data/raw/train-00000-of-00001.parquet"

logging.info("Reading Parquet file...")
# Read the Parquet file efficiently
parquet_file = pq.ParquetFile(parquet_file)
df = parquet_file.read_row_group(0).to_pandas().head(1000)
# df = parquet_file.read().to_pandas()

# Replace with your actual column names if different
query_column = "query"
passage_column = "passages"

# Extract queries and passages
queries = df[query_column].tolist()
passage_info = df[passage_column].tolist()

logging.info("Extracting all passages and removing duplicates...")
all_passages = []
for passages in tqdm(passage_info, desc="Extracting passages"):
    all_passages.extend(passages['passage_text'])
all_passages = list(set(all_passages))  # Remove duplicates

# Pre-compute all unique passages and their tokenized versions
logging.info("Pre-computing tokenized passages...")
all_passages_set = set()
tokenized_passages = {}  # Cache for tokenized passages
for passages in tqdm(passage_info, desc="Pre-tokenizing passages"):
    for passage in passages['passage_text']:
        if passage not in tokenized_passages:
            tokenized_passages[passage] = tokenise(passage)
        all_passages_set.add(passage)

# Pre-tokenize all queries
logging.info("Pre-tokenizing queries...")
tokenized_queries = {query: tokenise(query) for query in tqdm(queries, desc="Tokenizing queries")}

# Generate triples more efficiently
triples = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating triples"):
    query = row[query_column]
    passages_list = row[passage_column]
    positives = passages_list['passage_text']
    
    # Get tokenized query from cache
    tokenized_query = tokenized_queries[query]
    
    # Get tokenized positives from cache
    tokenized_positives = [tokenized_passages[p] for p in positives]
    
    # Create a set of positive passages for this query
    positive_set = set(positives)
    
    # Get all possible negatives in one operation
    possible_negatives = list(all_passages_set - positive_set)
    
    # Sample negatives
    num_negatives = min(10, len(possible_negatives))
    if num_negatives > 0:
        negatives = random.sample(possible_negatives, num_negatives)
        tokenized_negatives = [tokenized_passages[n] for n in negatives]
    else:
        tokenized_negatives = []
    
    triples.append({
        "query": tokenized_query,
        "positives": tokenized_positives,
        "negatives": tokenized_negatives
    })

logging.info("Saving triples to JSON file...")
with open('./data/processed/triples.json', 'w', encoding='utf-8') as f:
    json.dump(triples, f, indent=2, ensure_ascii=False)

logging.info("Opening token lookup table...")
# Load token to index mapping
with open('./data/tokenised/tkn_words_to_ids.pkl', 'rb') as f:
    token_to_idx = pickle.load(f)

logging.info("Creating indexed triples...")
# Convert tokens to indices in triples
indexed_triples = []
for triple in tqdm(triples, desc="Indexing triples"):
    indexed_triple = {
        "query": [token_to_idx.get(token, 0) for token in triple["query"]],
        "positives": [[token_to_idx.get(token, 0) for token in pos] for pos in triple["positives"]],
        "negatives": [[token_to_idx.get(token, 0) for token in neg] for neg in triple["negatives"]]
    }
    indexed_triples.append(indexed_triple)

logging.info("Saving indexed triples to JSON file...")
with open('./data/tokenised/indexed_triples.json', 'w', encoding='utf-8') as f:
    json.dump(indexed_triples, f, indent=2, ensure_ascii=False)