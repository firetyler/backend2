import math
import sys
import torch
import torch.nn as nn


import json
import re
import unicodedata
from collections import Counter
from logToken import get_logger
sys.stdout.reconfigure(encoding='utf-8')  # Forces UTF-8 output
logger = get_logger("SimpleTokenizer")
class SimpleTokenizer:
    def __init__(self, min_freq=1,vocab_file=None):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.vocab_size = 4
        self.min_freq = min_freq

    def _clean_text(self, text):
        # Normalisera unicode, ta bort punktuation och gör lowercase
        text = unicodedata.normalize("NFKD", text)
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text

    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            cleaned = self._clean_text(text)
            counter.update(cleaned.split())
        for word, freq in counter.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
        logger.debug(f"Vocab built with {self.vocab_size} words!")
        self.save_vocab("tokenizer_vocab.json")


    def update_vocab(filename, new_words):
        with open(filename, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        max_index = max(map(int, vocab["idx2word"].keys())) + 1

        for word in new_words:
            if word not in vocab["word2idx"]:
                vocab["word2idx"][word] = max_index
                vocab["idx2word"][str(max_index)] = word
                max_index += 1

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(vocab, f, indent=4)
        logger.debug(f" Vocab uppdaterad! Nya ord: {new_words}")
        
    def encode(self, text):
        cleaned = self._clean_text(text)
        tokens = []
        for w in cleaned.split():
            if w in self.word2idx:
                tokens.append(self.word2idx[w])
            else:
                logger.warning(f"Unrecognized word: '{w}', replacing with <UNK>")
                tokens.append(self.word2idx["<UNK>"])
        return [self.word2idx["<SOS>"]] + tokens + [self.word2idx["<EOS>"]]
    def decode(self, token_ids):
        # Ignorera bara <PAD> token (0)
        return " ".join([self.idx2word.get(i, "<UNK>") for i in token_ids if i != self.word2idx["<PAD>"]])

    def encode_batch(self, texts, max_len=None):
        batch_encoded = [self.encode(text) for text in texts]
        if max_len is None:
            max_len = max(len(seq) for seq in batch_encoded)
        padded_batch = []
        for seq in batch_encoded:
            if len(seq) < max_len:
                seq += [self.word2idx["<PAD>"]] * (max_len - len(seq))  # Padding med <PAD>
            else:
                seq = seq[:max_len]
            padded_batch.append(seq)
        return padded_batch
    def save_vocab(self, filename="tokenizer_vocab.json"):
        """Sparar vokabulären till en JSON-fil."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({"word2idx": self.word2idx, "idx2word": self.idx2word}, f)
        logger.info(f" Vocab saved to {filename}")
    def load_vocab(self, filename):
        """Laddar vokabulären från en JSON-fil."""
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.word2idx = data["word2idx"]
                self.idx2word = {int(k): v for k, v in data["idx2word"].items()}
                self.vocab_size = len(self.word2idx)
            logger.debug(f"Vocab loaded from {filename} ({self.vocab_size} words)")
        except FileNotFoundError:
            logger.warning(f" Vocab file {filename} not found. Using default.")


# --- Positionell kodning ---
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_mult=4, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_mult * embed_size),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * embed_size, embed_size),
        )
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        # Feedforward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class AdvancedSentenceTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size=256, num_layers=4, heads=8, max_len=512, dropout=0.1, pooling='mean'):
        super().__init__()
        self.embed_size = embed_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.position_encoding = PositionalEncoding(embed_size, max_len=max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_size, heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.pooling = pooling  # 'mean', 'max', 'cls'
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size)) if pooling == 'cls' else None
        self.dropout = nn.Dropout(dropout)

        # Optional projection head
        self.projection = nn.Linear(embed_size, embed_size)

    def forward(self, input_ids, attention_mask=None):
        B, S = input_ids.shape

        x = self.token_embedding(input_ids)
        x = self.position_encoding(x)

        if self.cls_token is not None:
            cls = self.cls_token.expand(B, -1, -1)  # shape: (B, 1, E)
            x = torch.cat([cls, x], dim=1)  # prepend cls token
            if attention_mask is not None:
                cls_mask = torch.ones(B, 1).to(attention_mask.device)
                attention_mask = torch.cat([cls_mask, attention_mask], dim=1)

        for layer in self.layers:
            x = layer(x, mask=None)  # You can add key_padding_mask for better masking

        # --- Pooling ---
        if self.pooling == 'mean':
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).expand_as(x)
                x = torch.sum(x * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            else:
                x = x.mean(dim=1)
        elif self.pooling == 'max':
            x = x.masked_fill((attention_mask == 0).unsqueeze(-1), -1e9)
            x = x.max(dim=1).values
        elif self.pooling == 'cls':
            x = x[:, 0]  # first token (cls)

        # Optional projection head
        return self.projection(self.dropout(x))


# --- Self-Attention ---
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.head_dim = embed_size // heads
        self.heads = heads
        assert self.head_dim * heads == embed_size

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.size(1), keys.size(1), query.size(1)

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == True, float("-1e20"))
        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        out = out.reshape(N, query_len, self.heads * self.head_dim)
        return self.fc_out(out)


# --- Transformerblock ---
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        return self.dropout(self.norm2(forward + x))


# --- Stackad Transformer ---
class StackedTransformer(nn.Module):
    def __init__(self, embed_size, vocab_size, num_layers, heads, forward_expansion, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.position = PositionalEncoding(embed_size)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, forward_expansion, dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.dropout(self.position(self.embedding(x)))
        for layer in self.layers:
            x = layer(x, x, x, mask)
        return self.fc_out(x)