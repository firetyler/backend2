import torch
import torch.nn as nn
import math
import re
import numpy as np
import faiss
import json
import os

import wikipedia

from DatabaseConnector import DatabaseConnector

# 1. Positionella inbäddningar (stöder variabel längd)
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embed_size = embed_size
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        seq_length = x.shape[1]
        return x + self.pe[:, :seq_length, :]

# 2. Självuppmärksamhet
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.embed_size = embed_size

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        attention = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(self.embed_size)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-inf"))

        attention = torch.softmax(attention, dim=-1)
        out = torch.matmul(attention, values)
        return self.fc_out(out)

# 3. Transformer-block
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention = self.attention(x, x, x, mask)
        x = self.norm1(attention + x)
        x = self.dropout(x)
        forward = self.feed_forward(x)
        return self.norm2(forward + x)

# 4. Staplad transformer-modell med flexibel input
class StackedTransformer(nn.Module):
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout):
        super(StackedTransformer, self).__init__()
        self.embed_size = embed_size
        self.position_encoding = PositionalEncoding(embed_size)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        if x.dim() == 2:
            x = x.unsqueeze(-1).expand(-1, -1, self.embed_size)
        seq_length = x.shape[1]  # Dynamiskt justera efter input-längd
        max_seq_len = min(seq_length, self.position_encoding.pe.shape[1])  # Anpassa efter max positionella inbäddningar

        x = self.position_encoding(x[:, :max_seq_len])  # Anpassa positionella inbäddningar
        for layer in self.layers:
            x = layer(x, mask[:, :max_seq_len])  # Se till att masken matchar input-längd
        return x


# 5. Minneshantering med FAISS
class AetherMemory:
    def __init__(self):
        self.texts = []
        self.embeddings = []
        self.index = faiss.IndexFlatL2(384)

    def embed(self, text):
        np.random.seed(abs(hash(text)) % (10**8))
        return np.random.rand(384).astype("float32")

    def add(self, text):
        if text not in self.texts:
            emb = self.embed(text)
            self.texts.append(text)
            self.embeddings.append(emb)
            self.index.add(np.array([emb]))

    def search(self, query, k=3):
        if not self.texts:
            return []
        emb = self.embed(query)
        D, I = self.index.search(np.array([emb]), k)
        return [self.texts[i] for i in I[0] if 0 <= i < len(self.texts)]
    
    def fetch_all_memories(self):
        return self.texts
    def calculator_tool(input_str):
        # Only allow safe characters
        allowed_chars = re.compile(r"^[\d\s\.\+\-\*/\(\)]+$")  # Includes -, decimals, parentheses, etc.
        input_str = input_str.replace('–', '-')  # Replace en dash if user types it
        input_str = input_str.replace('−', '-')  # Replace minus sign symbol with real dash

        if not allowed_chars.match(input_str):
            return "Invalid characters in expression."

        try:
            result = eval(input_str)
            return f"Calculated result: {result}"
        except Exception as e:
            return f"Error evaluating expression: {e}"
        
    def wikipedia_tool(query):
        try:
            summary = wikipedia.summary(query, sentences=2)
            return f"Wikipedia result: {summary}"
        except Exception as e:
            return f"Wikipedia error: {e}"
    
# 6. AetherAgent med flexibel input
class AetherAgent:
    def __init__(self, db_connector):
        self.db_connector = db_connector
        self.memory = AetherMemory()
        self.name = self.load_name()
        self.user_preferences = self.load_preferences()
        self.model = StackedTransformer(embed_size=256, num_layers=6, heads=8, forward_expansion=4, dropout=0.1)
    
    
    def load_name(self):
        try:
            with open("aether_name.json", "r") as file:
                data = json.load(file)
                return data.get("name", "Aether")
        except FileNotFoundError:
            return "Aether"
    def load_preferences(self):
        if hasattr(self.db_connector, "fetch_user_preferences"):
             return self.db_connector.fetch_user_preferences()
        else:
            print("❌ `fetch_user_preferences()` saknas i `DatabaseConnector`")
            return {}
        
    def run(self, user_input):
        print(f"\nGOAL: {user_input}")

    # 🔧 Convert input to tensor
        input_ids = torch.tensor([ord(c) for c in user_input]).unsqueeze(0).float()
        mask = torch.ones_like(input_ids)

        response = self.model(input_ids, mask)

    # 🔧 Clamp values within Unicode range before using `chr()`
        response_numpy = response.squeeze().detach().numpy().flatten()
        clamped_response = np.clip(response_numpy, 1, 0x10FFFF).astype(int)

        generated_text = ''.join([chr(c) for c in clamped_response]).replace('\x00', '')

        print("\nGenerated Response:\n", generated_text)
        self.memory.add(user_input)
        self.db_connector.insert_conversation("User", user_input, generated_text)
        return generated_text


# 7. Starta Aether
if __name__ == "__main__":
    db_con = DatabaseConnector()
    agent = AetherAgent(db_con)
    print("\nAether is ready. Ask your questions or give it tasks.")
    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() in ["quit", "exit"]:
                print("\nShutting down Aether.")
                break
            agent.run(user_input)
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting Aether.")
            break
