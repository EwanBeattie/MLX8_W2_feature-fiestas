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

logging.info("Reading first 100 rows from Parquet file...")
# Read the Parquet file efficiently
parquet_file = pq.ParquetFile(parquet_file)
df = parquet_file.read().to_pandas()

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

logging.info("Generating triples...")
# Convert to tensors for GPU processing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# Pre-tokenize all passages and queries in parallel
logging.info("Pre-tokenizing all passages and queries...")
all_passages = []
all_queries = []
for passages in tqdm(passage_info, desc="Collecting passages"):
    all_passages.extend(passages['passage_text'])
all_passages = list(set(all_passages))  # Remove duplicates

# Create a mapping of passage to index for faster lookup
passage_to_idx = {p: i for i, p in enumerate(all_passages)}

# Tokenize all passages in parallel
logging.info("Tokenizing passages in parallel...")
tokenized_passages = torch.tensor([
    tokenise(p) for p in tqdm(all_passages, desc="Tokenizing passages")
], device=device)

# Tokenize all queries in parallel
logging.info("Tokenizing queries in parallel...")
tokenized_queries = torch.tensor([
    tokenise(q) for q in tqdm(queries, desc="Tokenizing queries")
], device=device)

# Generate triples using vectorized operations
triples = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating triples"):
    query = row[query_column]
    passages_list = row[passage_column]
    positives = passages_list['passage_text']
    
    # Get indices for positives
    positive_indices = [passage_to_idx[p] for p in positives]
    
    # Create a mask for negatives
    negative_mask = torch.ones(len(all_passages), dtype=torch.bool, device=device)
    negative_mask[positive_indices] = False
    
    # Get negative indices
    negative_indices = torch.where(negative_mask)[0]
    
    # Sample negatives
    num_negatives = min(10, len(negative_indices))
    if num_negatives > 0:
        selected_negatives = negative_indices[torch.randperm(len(negative_indices))[:num_negatives]]
        negative_passages = tokenized_passages[selected_negatives]
    else:
        negative_passages = torch.tensor([], device=device)
    
    # Get positive passages
    positive_passages = tokenized_passages[positive_indices]
    
    # Get query
    query_idx = queries.index(query)
    query_tokens = tokenized_queries[query_idx]
    
    # Convert to lists for JSON serialization
    triples.append({
        "query": query_tokens.cpu().tolist(),
        "positives": positive_passages.cpu().tolist(),
        "negatives": negative_passages.cpu().tolist()
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

## Assuming embedding_matrix is an OrderedDict with token keys and tensor values
logging.info("Loading embedding matrix from .pth file...")
embedding_matrix = torch.load('./data/embeddings/2025_06_14__11_23_27.3.cbow.pth')

embedding_matrix = embedding_matrix['emb.weight']

logging.info("Averaging query embeddings for each triple...")
averaged_triples = []
for triple in tqdm(indexed_triples, desc="Averaging embeddings"):
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



