import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=512, learned=False):
        super().__init__()
        self.learned = learned

        if learned:
            self.pe = nn.Parameter(torch.randn(1, max_len, embed_size))
        else:
            pe = torch.zeros(max_len, embed_size)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))  # shape: (1, max_len, embed_size)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)] if not self.learned else x + self.pe[:, :x.size(1), :]


class StackedTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size=256,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        max_len=512,
        forward_expansion=4,
        learned_positional_encoding=True,
        pooling="cls"  # 'cls', 'mean', 'max'
    ):
        super().__init__()

        self.pooling = pooling
        self.embed_size = embed_size

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.position_encoding = PositionalEncoding(embed_size, max_len=max_len, learned=learned_positional_encoding)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size)) if pooling == "cls" else None
        self.dropout = nn.Dropout(dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=embed_size * forward_expansion,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_size)

        # Output projection
        self.projection = nn.Linear(embed_size, embed_size)

    def forward(self, input_ids, attention_mask=None):
        B, S = input_ids.shape

        x = self.token_embedding(input_ids)
        x = self.position_encoding(x)

        # Prepend CLS token if used
        if self.cls_token is not None:
            cls = self.cls_token.expand(B, 1, -1)
            x = torch.cat([cls, x], dim=1)
            if attention_mask is not None:
                cls_mask = torch.ones(B, 1, device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([cls_mask, attention_mask], dim=1)

        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        x = self.dropout(x)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)

        # Pooling
        if self.pooling == "cls":
            x = x[:, 0]
        elif self.pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                x = x.mean(dim=1)
        elif self.pooling == "max":
            if attention_mask is not None:
                x = x.masked_fill((attention_mask == 0).unsqueeze(-1), float("-inf"))
            x = x.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling}")

        return self.projection(x)
