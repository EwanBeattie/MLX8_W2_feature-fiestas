import pandas as pd
import random
import json
import logging
import pickle
from tokensier import tokenise
import pyarrow.parquet as pq
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)

# Replace with your actual file path
parquet_file = "data/raw/train-00000-of-00001.parquet"

logging.info("Reading first 100 rows from Parquet file...")
# Read the Parquet file efficiently
parquet_file = pq.ParquetFile(parquet_file)
df = parquet_file.read_row_group(0).to_pandas().head(100)

# Replace with your actual column names if different
query_column = "query"
passage_column = "passages"

# Extract queries and passages
queries = df[query_column].tolist()
passage_info = df[passage_column].tolist()

logging.info("Extracting all passages and removing duplicates...")
all_passages = []
for passages in passage_info:
    all_passages.extend(passages['passage_text'])
all_passages = list(set(all_passages))  # Remove duplicates

#TODO: Why is positives not 10 long?
logging.info("Generating triples...")
triples = []
for idx, row in df.iterrows():
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
for triple in triples:
    indexed_triple = {
        "query": [token_to_idx.get(token, 0) for token in triple["query"]],
        "positives": [[token_to_idx.get(token, 0) for token in pos] for pos in triple["positives"]],
        "negatives": [[token_to_idx.get(token, 0) for token in neg] for neg in triple["negatives"]]
    }
    indexed_triples.append(indexed_triple)

logging.info("Saving indexed triples to JSON file...")
with open('./data/tokenised/indexed_triples.json', 'w', encoding='utf-8') as f:
    json.dump(indexed_triples, f, indent=2, ensure_ascii=False)

## Assuming embedding_matrix is an OrderedDict with token keys and tensor values
logging.info("Loading embedding matrix from .pth file...")
embedding_matrix = torch.load('./data/embeddings/2025_06_14__11_23_27.3.cbow.pth')

embedding_matrix = embedding_matrix['emb.weight']

logging.info("Averaging query embeddings for each triple...")
averaged_triples = []
for triple in indexed_triples:
    query_indices = triple["query"]
    valid_indices = [idx for idx in query_indices if 0 <= idx < embedding_matrix.size(0)]
    embeddings = embedding_matrix[valid_indices]
    avg_embedding = embeddings.mean(dim=0).tolist() if embeddings.size(0) > 0 else [0] * embedding_matrix.size(1)

    # Average each of the positives
    averaged_positives = []
    for pos_indices in triple["positives"]:
        valid_pos_indices = [idx for idx in pos_indices if 0 <= idx < embedding_matrix.size(0)]
        if valid_pos_indices:
            pos_embs = embedding_matrix[valid_pos_indices]
            avg_pos_emb = pos_embs.mean(dim=0).tolist()
        else:
            avg_pos_emb = [0] * embedding_matrix.size(1)
        averaged_positives.append(avg_pos_emb)

    # Average each of the negatives
    averaged_negatives = []
    for neg_indices in triple["negatives"]:
        valid_neg_indices = [idx for idx in neg_indices if 0 <= idx < embedding_matrix.size(0)]
        if valid_neg_indices:
            neg_embs = embedding_matrix[valid_neg_indices]
            avg_neg_emb = neg_embs.mean(dim=0).tolist()
        else:
            avg_neg_emb = [0] * embedding_matrix.size(1)
        averaged_negatives.append(avg_neg_emb)

    averaged_triples.append({
        "query": avg_embedding,
        "positives": averaged_positives,
        "negatives": averaged_negatives
    })

logging.info("Saving averaged query embeddings to JSON file...")
with open('./data/tokenised/averaged_triple_embeddings.json', 'w', encoding='utf-8') as f:
    json.dump(averaged_triples, f, indent=2, ensure_ascii=False)



