import torch
import torch.nn.functional as F
import torch.optim as optim
import json
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from model import TwoTowerModel
import wandb

# --- Custom Dataset ---
class TripletDataset(Dataset):
    def __init__(self, triples):
        self.samples = []
        for triple in triples:
            q = torch.tensor(triple['query'], dtype=torch.long)
            pos_lists = triple['positives']
            neg_lists = triple['negatives']
            num_pairs = min(len(pos_lists), len(neg_lists))
            for i in range(num_pairs):
                pos = torch.tensor(pos_lists[i], dtype=torch.long)
                neg = torch.tensor(neg_lists[i], dtype=torch.long)
                self.samples.append((q, pos, neg))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# --- Collate function for batching ---
def collate_fn(batch):
    queries, positives, negatives = zip(*batch)
    qry_lens = torch.tensor([len(q) for q in queries])
    pos_lens = torch.tensor([len(p) for p in positives])
    neg_lens = torch.tensor([len(n) for n in negatives])
    queries_padded = pad_sequence(queries, batch_first=True, padding_value=0)
    positives_padded = pad_sequence(positives, batch_first=True, padding_value=0)
    negatives_padded = pad_sequence(negatives, batch_first=True, padding_value=0)
    return queries_padded, qry_lens, positives_padded, pos_lens, negatives_padded, neg_lens

# --- Initialize wandb ---
wandb.init(
    project="two-tower-training",
    config={
        "learning_rate": 1e-3,  # default value
        "batch_size": 32,       # default value
        "margin": 0.2,         # default value
        "epochs": 10,          # default value
        "embedding_dim": None  # will set after loading weights
    }
)

# --- Load data ---
with open('./data/tokenised/indexed_triples.json', 'r') as f:
    triples = json.load(f)

# --- Load pretrained embedding weights ---
embedding_weights = torch.load('./data/embeddings/2025_06_14__11_23_27.3.cbow.pth')
embedding_weights = embedding_weights['emb.weight']
embedding_dim = embedding_weights.shape[1]
wandb.config.embedding_dim = embedding_dim

dataset = TripletDataset(triples)
batch_size = wandb.config.batch_size
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# --- Model, optimizer ---
model = TwoTowerModel(embedding_weights, hidden_dim=embedding_dim, freeze_emb=True)
optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

# --- Training loop ---
epochs = wandb.config.epochs
margin = wandb.config.margin

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for queries_padded, qry_lens, positives_padded, pos_lens, negatives_padded, neg_lens in loader:
        optimizer.zero_grad()
        qry_vec = model.qry_tower(queries_padded, qry_lens)
        pos_vec = model.doc_tower(positives_padded, pos_lens)
        neg_vec = model.doc_tower(negatives_padded, neg_lens)

        sim_pos = F.cosine_similarity(qry_vec, pos_vec)
        sim_neg = F.cosine_similarity(qry_vec, neg_vec)
        loss = torch.clamp(margin - (sim_pos - sim_neg), min=0).mean()

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * queries_padded.size(0)
    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    wandb.log({"epoch": epoch+1, "loss": avg_loss})

# --- Save the model ---
model_path = 'two_tower_model.pth'
torch.save(model.state_dict(), model_path)
wandb.save(model_path)
# Optionally, log as artifact:
artifact = wandb.Artifact('two_tower_model', type='model')
artifact.add_file(model_path)
wandb.log_artifact(artifact)