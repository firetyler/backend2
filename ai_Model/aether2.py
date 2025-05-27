from collections import Counter
import os
import sys
import unicodedata
import torch
import torch.nn as nn
import math
import re
import numpy as np
import faiss
import json
import wikipedia
from torch.utils.data import Dataset,DataLoader

sys.stdout.reconfigure(encoding='utf-8')  # Forces UTF-8 output
from DatabaseConnector import DatabaseConnector
from logger_setup import get_logger
from code_executor import CodeExecutor

logger = get_logger("aether2")
# --- Enkel Tokenizer ---
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
        print(f" Vocab uppdaterad! Nya ord: {new_words}")

    

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

#TODO fix if brocken
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

# --- Minnesmodul ---
class AetherMemory:
    def __init__(self, embed_model, tokenizer, device):
        self.embed_model = embed_model
        self.memories = []
        self.vector_dim = embed_model.embed_size
        
        self.index = faiss.IndexFlatL2(self.vector_dim)
        self.tokenizer = tokenizer
        self.device = device

    def add(self, text):
        vector = self.text_to_vector(text)
        self.memories.append(text)
        self.index.add(np.array([vector]))
        logger.info(f"Memory added: '{text[:50]}...'")

    def fetch_all_memories(self):
        return self.memories.copy()

    def text_to_vector(self, text):
        self.embed_model.eval()
        with torch.no_grad():
            token_ids = self.tokenizer.encode(text)
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            attention_mask = (input_ids != 0).long()
            embedding = self.embed_model(input_ids, attention_mask=attention_mask) 
            return embedding[0].detach().cpu().numpy().astype('float32')
        
    def semantic_search(self, query, top_k=3):
        query_vector = self.text_to_vector(query)
        if len(self.memories) == 0:
            return []
        distances, indices = self.index.search(np.array([query_vector]), top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.memories):
                results.append((self.memories[idx], float(distances[0][i])))
        return results
    
    def calculator_tool(self, expression):
        try:
            logger.info(f" Evaluating expression: {expression}")
            safe_expr = re.sub(r"[^0-9+\-*/(). ]", "", expression)
            result = eval(safe_expr, {"__builtins__": {}})
            logger.info(f"Calculation result: {result}")
            return f"Result: {eval(safe_expr)}"
        except Exception as e:
            logger.error(f" Error evaluating expression '{expression}': {e}")
            return f"Error in calculation: {e}"

    def wikipedia_tool(self, query):
        try:
            summary = wikipedia.summary(query, sentences=2)
            logger.info(f"Wikipedia result for '{query}': {summary[:100]}...")
            return wikipedia.summary(query, sentences=2)
        
        except wikipedia.exceptions.DisambiguationError as e:
            logger.warning(f" Multiple results for '{query}': {e.options[:3]}")
            return f"Disambiguation error: Try one of: {', '.join(e.options[:3])}"
        except wikipedia.exceptions.PageError:
            logger.error(f" No page found for '{query}'")
            return f"No Wikipedia page found for: {query}"
        except Exception as e:
            logger.error(f" Wikipedia lookup failed for '{query}': {e}")
            return f"Wikipedia lookup failed: {e}"
        

class ChatDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data  # list of (input_text, output_text) tuples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text, output_text = self.data[idx]
        
        # Tokenisera input och output och konvertera till tensorer
        input_tokens = torch.tensor(self.tokenizer.encode(input_text), dtype=torch.long)
        output_tokens = torch.tensor(self.tokenizer.encode(output_text), dtype=torch.long)
        
        return input_tokens, output_tokens





# --- AetherAgent ---
class AetherAgent:
    def __init__(self, db_connector):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.db_connector = db_connector
        self.name = "Aether"
        self.tokenizer = SimpleTokenizer()
        self.model = None
        self.code_executor = CodeExecutor()
        self.embed_model = None
        self.embed_size = 256  # Mock for embed_model.embed_size
        self.embed_model = AdvancedSentenceTransformer(vocab_size=self.tokenizer.vocab_size).to(self.device)
        self.memory = AetherMemory(self.embed_model, self.tokenizer, self.device)
        self.ChatDataset = ChatDataset
    def handle_code_question(self, language, code):
        return self.code_executor.run_code(code, language)
    def load_config(self, path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f" Could not read the config file.: {e}")
            return {}
    def save_checkpoint(self, epoch, optimizer, filename):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'tokenizer': self.tokenizer.word2idx
        }
        torch.save(checkpoint, filename)
        logger.info(f"Checkpoint saved: {filename}")

 
    def load_checkpoint(self, filename, optimizer, config):
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location=self.device)

            # Ladda tokenizern från sparad vokabulär först
            self.tokenizer.load_vocab(config.get("tokenizer_vocab_path", "tokenizer_vocab.json"))
            logger.info(f"Tokenizer vocab loaded from file with size {self.tokenizer.vocab_size}")

            # 🔄 Om checkpoint innehåller en äldre vokabulär, behåll den senaste istället
            if 'tokenizer' in checkpoint:
                if len(checkpoint['tokenizer']) > len(self.tokenizer.word2idx):
                    logger.warning(" Checkpoint tokenizer has more entries, updating...")
                    self.tokenizer.word2idx = checkpoint['tokenizer']
                    self.tokenizer.idx2word = {idx: word for word, idx in self.tokenizer.word2idx.items()}
                    self.tokenizer.vocab_size = len(self.tokenizer.word2idx)
                    logger.info(f" Tokenizer updated from checkpoint with vocab size {self.tokenizer.vocab_size}")
                else:
                    logger.info("Keeping latest tokenizer vocabulary.")
            else:
                logger.error(" Tokenizer vocab not found in checkpoint.")
                return 0

            # Initialisera modellen med värden från config-filen
            vocab_size = self.tokenizer.vocab_size
            logger.info(f"Setting vocab_size to {vocab_size} before loading checkpoint.")

            self.model = StackedTransformer(
                embed_size=config.get("embedding_dim", 256),
                vocab_size=vocab_size,  # Anpassa vocab_size dynamiskt
                num_layers=config.get("num_layers", 4),
                heads=config.get("heads", 8),
                forward_expansion=config.get("forward_expansion", 4),
                dropout=config.get("dropout", 0.1)
            ).to(self.device)

            #  Ladda modellens state_dict men filtrera inkompatibla delar
            model_dict = self.model.state_dict()
            pretrained_dict = {
                k: v for k, v in checkpoint['model_state_dict'].items()
                if k in model_dict and model_dict[k].shape == v.shape
            }
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            logger.info(f"Checkpoint loaded from {filename}, starting from epoch {start_epoch}")

            # Spara tokenizer-vokabulären så att den är tillgänglig nästa gång
            self.tokenizer.save_vocab(config.get("tokenizer_vocab_path", "tokenizer_vocab.json"))

            return start_epoch
        else:
            logger.warning(f" No checkpoint file found: {filename}")
            return 0



           
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
        """Train the AI model using data from JSON files."""
        
        #  Load configuration
        config = self.load_config(config_path)
        epochs = config.get("epochs", 10)
        lr = config.get("learning_rate", 0.001)
        batch_size = config.get("batch_size", 32)
        filenames = config.get("train_data_paths", ["ai_Model/chat_training_data.json"])

        #  Load training data properly
        training_data = []
        for filename in filenames:
            try:
                with open(filename, 'r', encoding="utf-8") as f:
                    raw_data = json.load(f)
                    
                    if not raw_data:
                        logger.warning(f" No training data found in {filename}. Skipping...")
                        continue
                    
                    valid_data = [(d['input'], d['output']) for d in raw_data if 'input' in d and 'output' in d]
                    training_data.extend(valid_data)
            except Exception as e:
                logger.error(f"Failed to load training data from {filename}: {e}")
                return
            
        if not training_data:
            logger.error("No valid training data loaded. Aborting training...")
            return

        #  Prepare dataset & DataLoader
        dataset = self.ChatDataset(training_data, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=self.collate_fn)

        #  Initialize optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        self.model.train()

        #  Handle checkpoints correctly
        checkpoint_path = "checkpoint_latest.pth"
        start_epoch = 0
        if os.path.exists(checkpoint_path):
            logger.info(f" Found checkpoint: Loading from {checkpoint_path}...")
            start_epoch = self.load_checkpoint(checkpoint_path, optimizer, config)
        else:
            logger.warning(f"No checkpoint found – starting training from scratch...")
            start_epoch = 0  # Ensure training starts at the correct epoch
        
        #  Run training loop
        logger.info(f"Starting model training for {epochs} epochs...")
        for epoch in range(start_epoch, epochs):
            total_loss = 0.0
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                mask = generate_square_subsequent_mask(x.size(1)).to(self.device)
                out = self.model(x, mask)
                loss = criterion(out.view(-1, out.size(-1)), y.view(-1))

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

            # Save checkpoint after each epoch
            self.save_checkpoint(epoch + 1, optimizer, checkpoint_path)

        logger.info("Training complete! Model saved successfully.")

    def collate_fn(self, batch):
        inputs, targets = zip(*batch)
    
    # Hitta maxlängden bland både inputs och targets
        max_len = max(
        max(len(seq) for seq in inputs),
        max(len(seq) for seq in targets)
        )
    
        def pad_sequences(sequences, max_len):
            padded_seqs = []
            for seq in sequences:
                seq_tensor = seq.detach().clone()
                pad_size = max_len - len(seq)
                if pad_size > 0:
                    padding = torch.zeros(pad_size, dtype=torch.long)
                    seq_tensor = torch.cat([seq_tensor, padding])
                padded_seqs.append(seq_tensor)
            return torch.stack(padded_seqs)
    
        inputs_padded = pad_sequences(inputs, max_len)
        targets_padded = pad_sequences(targets, max_len)
    
        return inputs_padded, targets_padded
    
    def save_model(self, filename="aether_model.pth"):
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename="aether_model.pth"):
        if self.model is None:
            raise RuntimeError("Model not initialized. You must train or preprocess first.")
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
        self.model.eval()

    import json

    def generate_text(self, prompt, max_length=50):
        if self.model is None:
            logger.warning("Model not initialized. Call `train_model()` or `load_model()` before generating text.")
            return json.dumps({"error": "Model not initialized."})  #  JSON-format

        self.model.eval()
        tokens = self.tokenizer.encode(prompt)
        print(f"Tokenized input: {tokens}")

        # ✅ Se till att tokenizern returnerar data
        if len(tokens) == 0:
            logger.error("Tokenizer returned an empty sequence!")
            return json.dumps({"error": "Invalid input, couldn't process text."})  #  JSON-format

        input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)

        for _ in range(max_length):
            mask = generate_square_subsequent_mask(input_ids.size(1)).to(self.device)
            
            with torch.no_grad():
                out = self.model(input_ids, mask)

            next_token_logits = out[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            if next_token_id.item() not in self.tokenizer.word2idx.values():
                logger.error("Next token ID is out of range!")
                return json.dumps({"error": "Generated an invalid token."})  #  JSON-format

            input_ids = torch.cat((input_ids, next_token_id), dim=1)
            if next_token_id.item() == self.tokenizer.word2idx["<EOS>"]:
                break

        generated_text = self.tokenizer.decode(input_ids[0].tolist())

        return json.dumps({"response": generated_text})  #  Returnera JSON-struktur


    def run(self, user_input):
        logger.info(f"GOAL: {user_input}")
        corrected_input = re.sub(r"\s+", " ", user_input.strip().lower())
        name_patterns = [
        r"^(what('?s| is) your name\??)$",
        r"^(who are you\??)$"
        ]
        if any(re.match(pattern, corrected_input) for pattern in name_patterns):
            return "My name is Aether."




        lowered = user_input.lower()
        if lowered.startswith("run "):
            try:
                parts = user_input.split(" ", 2)
                if len(parts) < 3:
                    logger.warning("Invalid format for the 'run' command.")
                    return "Use the format: run <language> <code>"
                logger.info(f"Running code in language: {language}")
                logger.debug(f"Code being executed:\n{code}")
                language = parts[1]
                code = parts[2]
                result = self.handle_code_question(language, code)
                logger.info(f"Code execution succeeded for language: {language}")
                return f"Result for {language}:\n{result}"
            except Exception as e:
                logger.exception(f"Error during code execution in language: {language}")
                return f"Something went wrong during code execution: {e}"
        

        if user_input.lower().strip() == "train model":
            self.train_model()
            self.save_model()
            print("✅ Training finished")
            return "Model trained and saved."

        if "calculate" in user_input.lower():
            expression = user_input.lower().replace("calculate", "").strip()
            return self.memory.calculator_tool(expression)

        if "wikipedia" in user_input.lower():
            query = user_input.lower().replace("wikipedia", "").strip()
            return self.memory.wikipedia_tool(query)
        
        if lowered in ["try again", "that's wrong", "incorrect", "answer again", "that doesn't sound right"]:
            if self.memory.memories:
                last_prompt = self.memory.memories[-1]
                logger.info(f" User requested retry for: {last_prompt}")
                regenerated = self.generate_text(last_prompt)
                self.db_connector.insert_conversation(name="Aether", input="", output=regenerated)
                return regenerated
            else:
                return "I don't have a previous message to retry."

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
        logger.error(f"Failed to build vocab: {e}")

    # Initiera och ladda modell
    agent.initialize_model(config)
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Files in current directory: {os.listdir()}")
    try:
        agent.load_model(filename=config["model_path"])
        logger.info("Model loaded.")
    except:
        logger.warning(" Kunde inte ladda modell – tränar ny modell...")
        agent.train_model(config_path="ai_Model/config.json")
        agent.save_model(filename=config["model_path"])

    logger.info("Aether is ready. Ask me anything!")

    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() in ["exit", "quit"]:
                break
            output = agent.run(user_input)
            logger.info(f"Response: {output}")
        except KeyboardInterrupt:
            logger.info(" Exiting...")
            break
