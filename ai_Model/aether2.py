import os
import torch
import torch.nn as nn
import math
import re
import numpy as np
import faiss
import json
import wikipedia
from torch.utils.data import Dataset,DataLoader
# --- Enkel Tokenizer ---
class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.vocab_size = 4

    def build_vocab(self, texts):
        for text in texts:
            for word in text.lower().split():
                if word not in self.word2idx:
                    self.word2idx[word] = self.vocab_size
                    self.idx2word[self.vocab_size] = word
                    self.vocab_size += 1

    def encode(self, text):
        return [1] + [self.word2idx.get(w, 3) for w in text.lower().split()] + [2]

    def decode(self, token_ids):
        return " ".join([self.idx2word.get(i, "<UNK>") for i in token_ids if i > 2])

# --- Mask för self-attention ---
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    return mask

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

# --- Minnesmodul ---
class AetherMemory:
    def __init__(self):
        self.memories = []
        self.vector_dim = 256
        self.index = faiss.IndexFlatL2(self.vector_dim)

    def add(self, text):
        vector = self.text_to_vector(text)
        self.memories.append(text)
        self.index.add(np.array([vector]))

    def fetch_all_memories(self):
        return self.memories

    def text_to_vector(self, text):
        np.random.seed(hash(text) % (2 ** 32))
        return np.random.rand(self.vector_dim).astype('float32')

    def calculator_tool(self, expression):
        try:
            safe_expr = re.sub(r"[^0-9+\-*/(). ]", "", expression)
            return f"Result: {eval(safe_expr)}"
        except Exception as e:
            return f"Error in calculation: {e}"

    def wikipedia_tool(self, query):
        try:
            return wikipedia.summary(query, sentences=2)
        except Exception as e:
            return f"Wikipedia lookup failed: {e}"

# --- Dummy-databas ---
class DatabaseConnector:
    def insert_conversation(self, user, input_text, output_text):
        pass
    def fetch_user_preferences(self):
        return {}
class ChatDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text, target_text = self.data[idx]
        input_ids = self.tokenizer.encode(input_text)
        target_ids = self.tokenizer.encode(target_text)
        return input_ids, target_ids
# --- AetherAgent ---
class AetherAgent:
    def __init__(self, db_connector):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.db_connector = db_connector
        self.memory = AetherMemory()
        self.name = "Aether"
        self.tokenizer = SimpleTokenizer()
        self.model = None
        self.embedding_dim = 256

    def load_config(self, path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Kunde inte läsa configfilen: {e}")
            return {}

    def preprocess_data(self, training_data, config):
        all_texts = [q for q, _ in training_data] + [a for _, a in training_data]
        self.tokenizer.build_vocab(all_texts)

        num_layers = config.get("num_layers", 4)
        heads = config.get("heads", 8)
        forward_expansion = config.get("forward_expansion", 4)
        dropout = config.get("dropout", 0.1)
        embed_size = config.get("embedding_dim", 256)

        self.model = StackedTransformer(
            embed_size=embed_size,
            vocab_size=self.tokenizer.vocab_size,
            num_layers=num_layers,
            heads=heads,
            forward_expansion=forward_expansion,
            dropout=dropout
        ).to(self.device)

        inputs = []
        targets = []

        for text_input, expected_output in training_data:
            x = self.tokenizer.encode(text_input)
            y = self.tokenizer.encode(expected_output)
            inputs.append(torch.tensor(x, dtype=torch.long))
            targets.append(torch.tensor(y, dtype=torch.long))

    # Hitta maxlängd för input och output
        max_len_x = max([len(seq) for seq in inputs])
        max_len_y = max([len(seq) for seq in targets])

    # Pad sequences
        def pad_sequence(seq, max_len):
            return torch.cat([seq, torch.zeros(max_len - len(seq), dtype=torch.long)])

        inputs_padded = [pad_sequence(seq, max_len_x) for seq in inputs]
        targets_padded = [pad_sequence(seq, max_len_y) for seq in targets]

    # Flytta till device och lägg ihop i batcher (listor av tuples)
        data = [(seq_x.unsqueeze(0).to(self.device), seq_y.unsqueeze(0).to(self.device)) for seq_x, seq_y in zip(inputs_padded, targets_padded)]

        return data
    def initialize_model(self, config):
        num_layers = config.get("num_layers", 4)
        heads = config.get("heads", 8)
        forward_expansion = config.get("forward_expansion", 4)
        dropout = config.get("dropout", 0.1)
        embed_size = config.get("embedding_dim", 256)

        self.model = StackedTransformer(
            embed_size=embed_size,
            vocab_size=self.tokenizer.vocab_size,
            num_layers=num_layers,
            heads=heads,
            forward_expansion=forward_expansion,
            dropout=dropout
        ).to(self.device)

    def train_model(self, filename="ai_Model/chat_training_data.json", config_path="ai_Model/config.json"):
        config = self.load_config(config_path)
        epochs = config.get("epochs", 10)
        lr = config.get("learning_rate", 0.001)
        batch_size = config.get("batch_size", 32)

        try:
            with open(filename, 'r', encoding="utf-8") as f:
                raw_data = json.load(f)
        except Exception as e:
            print(f"Failed to load training data: {e}")
            return

         
        training_data = [(d['input'], d['output']) for d in raw_data if 'input' in d and 'output' in d]

        if not training_data:
            print("❌ Inga träningsdata hittades i filen. Kontrollera att filen innehåller korrekt träningsdata.")
            return

        token_data = self.preprocess_data(training_data, config)

        if not token_data:
            print("❌ Preprocesseringen gav inga token-data. Kontrollera preprocess_data-funktionen och träningsdataformatet.")
            return
        #dataloder
        dataset = self.ChatDataset(training_data, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=self.collate_fn)
        #loop
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss = 0.0
            for x, y in token_data:
                optimizer.zero_grad()
                mask = generate_square_subsequent_mask(x.size(1)).to(self.device)
                out = self.model(x, mask)
                loss = criterion(out.view(-1, out.size(-1)), y.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
            avg_loss = total_loss / len(token_data)
            print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")


    def save_model(self, filename="aether_model.pth"):
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename="aether_model.pth"):
        if self.model is None:
            raise RuntimeError("Model not initialized. You must train or preprocess first.")
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
        self.model.eval()

    def generate_text(self, prompt, max_length=50):
        if self.model is None:
            print("Model not initialized. Call `train_model()` or `load_model()` before generating text.")
            return ""
        self.model.eval()
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long).to(self.device)

        for _ in range(max_length):
            mask = generate_square_subsequent_mask(input_ids.size(1)).to(self.device)
            with torch.no_grad():
                out = self.model(input_ids, mask)
            next_token_logits = out[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            input_ids = torch.cat((input_ids, next_token_id), dim=1)
            if next_token_id.item() == self.tokenizer.word2idx["<EOS>"]:
                break
        return self.tokenizer.decode(input_ids[0].tolist())

    def run(self, user_input):
        print(f"GOAL: {user_input}")
        corrected_input = user_input.lower().replace("wahts", "what's")

        if "what's your name ?" in corrected_input or "what is your name" in corrected_input:
            return "My name is Aether."

        if user_input.lower().strip() == "train model":
            self.train_model()
            self.save_model()
            return "Model trained and saved."

        if "calculate" in user_input.lower():
            expression = user_input.lower().replace("calculate", "").strip()
            return self.memory.calculator_tool(expression)

        if "wikipedia" in user_input.lower():
            query = user_input.lower().replace("wikipedia", "").strip()
            return self.memory.wikipedia_tool(query)

        generated = self.generate_text(user_input)
        self.memory.add(user_input)
        self.db_connector.insert_conversation(name="User", input=user_input, output="")
        self.db_connector.insert_conversation(name="Aether", input="", output=generated)
        return generated

# --- Kör agenten ---
if __name__ == "__main__":
    db = DatabaseConnector()
    agent = AetherAgent(db)

    config = agent.load_config("ai_Model/config.json")
    
    try:
        with open(config.get("train_data_path", "ai_Model/chat_training_data.json"), "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        training_data = [(d['input'], d['output']) for d in raw_data if 'input' in d and 'output' in d]
        all_texts = [q for q, _ in training_data] + [a for _, a in training_data]
        agent.tokenizer.build_vocab(all_texts)
    except Exception as e:
        print("Failed to build vocab:", e)

    # Initiera och ladda modell
    agent.initialize_model(config)
    print("Current working directory:", os.getcwd())
    print("Files in current directory:", os.listdir())
    try:
        agent.load_model(filename=config["model_path"])
        print("✅ Modell laddad.")
    except:
        print("⚠️ Kunde inte ladda modell – tränar ny modell...")
        agent.train_model(config_path="ai_Model/config.json")
        agent.save_model(filename=config["model_path"])

    print("🤖 Aether är redo. Fråga något!")

    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() in ["exit", "quit"]:
                break
            output = agent.run(user_input)
            print("Svar:", output)
        except KeyboardInterrupt:
            print("\n👋 Avslutar...")
            break
