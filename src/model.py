import torch
import torch.nn as nn

class PretrainedEmbedding(nn.Module):
    def __init__(self, embedding_weights, freeze=True):
        super().__init__()
        num_embeddings, embedding_dim = embedding_weights.shape
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(embedding_weights, requires_grad=not freeze)

    def forward(self, x):
        return self.embedding(x)

class QryTower(nn.Module):
    def __init__(self, embedding_layer, hidden_dim):
        super().__init__()
        self.embedding = embedding_layer
        self.rnn = nn.LSTM(self.embedding.embedding.embedding_dim, hidden_dim, batch_first=True)

    def forward(self, x, lengths):
        x = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.rnn(packed)
        return h[-1]

class DocTower(nn.Module):
    def __init__(self, embedding_layer, hidden_dim):
        super().__init__()
        self.embedding = embedding_layer
        self.rnn = nn.LSTM(self.embedding.embedding.embedding_dim, hidden_dim, batch_first=True)

    def forward(self, x, lengths):
        x = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.rnn(packed)
        return h[-1]

class TwoTowerModel(nn.Module):
    def __init__(self, embedding_weights, hidden_dim=128, freeze_emb=True):
        super().__init__()
        embedding_layer = PretrainedEmbedding(embedding_weights, freeze=freeze_emb)
        self.qry_tower = QryTower(embedding_layer, hidden_dim)
        self.doc_tower = DocTower(embedding_layer, hidden_dim)

    def forward(self, qry, qry_lens, doc, doc_lens):
        qry_vec = self.qry_tower(qry, qry_lens)
        doc_vec = self.doc_tower(doc, doc_lens)
        return qry_vec, doc_vec