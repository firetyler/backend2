import torch
import torch.nn as nn
import math
import re
import numpy as np
import faiss
import json
import wikipedia
from transformers import AutoTokenizer
from DatabaseConnector import DatabaseConnector

# --- Hjälpfunktion för causal mask ---
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    return mask

# --- Positionell kodning ---
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
        return x + self.pe[:, :seq_length, :].to(x.device)

# --- Self-Attention ---
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (
            self.head_dim * heads == embed_size
        ), "Embed size needs to be divisible by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)        # (N, key_len, heads, head_dim)
        queries = self.queries(queries)  # (N, query_len, heads, head_dim)
        
        # Einsum to get scores: (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == True, float("-1e20"))
        
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
        out = self.fc_out(out)
        return out

# --- Transformer Block ---
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

# --- Stacked Transformer ---
class StackedTransformer(nn.Module):
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout):
        super(StackedTransformer, self).__init__()
        self.embed_size = embed_size
        self.num_layers = num_layers
        
        self.positional_encoding = PositionalEncoding(embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, forward_expansion, dropout)
                for _ in range(num_layers)
            ]
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_size, 30522)  # Antal tokens i BERT-base uncased

    def forward(self, x, mask=None):
        out = self.dropout(self.positional_encoding(x))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        out = self.fc_out(out)
        return out

# --- Minneshantering ---
class AetherMemory:
    def __init__(self):
        self.memories = []
        self.vector_dim = 256  # Samma som embed size i modellen
        self.index = faiss.IndexFlatL2(self.vector_dim)
        self.id_map = []

    def add(self, text):
        vector = self.text_to_vector(text)
        self.memories.append(text)
        self.index.add(np.array([vector]))
        self.id_map.append(len(self.memories) - 1)

    def fetch_all_memories(self):
        return self.memories

    def text_to_vector(self, text):
        # En enkel text till vektor - kan förbättras med ex BERT embeddings
        np.random.seed(hash(text) % (2 ** 32))
        return np.random.rand(self.vector_dim).astype('float32')

    def calculator_tool(self, expression):
        try:
            # Rensa expression från farliga tecken
            safe_expression = re.sub(r"[^0-9+\-*/(). ]", "", expression)
            result = eval(safe_expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error in calculation: {e}"

    def wikipedia_tool(self, query):
        try:
            summary = wikipedia.summary(query, sentences=2)
            return summary
        except Exception as e:
            return f"Wikipedia lookup failed: {e}"

# --- AetherAgent ---
class AetherAgent:
    def __init__(self, db_connector):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.db_connector = db_connector
        self.memory = AetherMemory()
        self.name = self.load_name()
        self.user_preferences = self.load_preferences()
       
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.embedding = nn.Embedding(self.tokenizer.vocab_size, 256).to(self.device)
        self.model = StackedTransformer(embed_size=256, num_layers=4, heads=8, forward_expansion=4, dropout=0.1)
        self.model.to(self.device)

        

    def load_name(self):
        try:
            with open("aether_name.json", "r") as file:
                data = json.load(file)
                return data.get("name", "Aether")
        except FileNotFoundError:
            return "Aether"

    def save_name(self, new_name):
        with open("aether_name.json", "w") as file:
            json.dump({"name": new_name}, file)
        self.name = new_name

    def preprocess_data(self, training_data):
        processed_data = []
        for text_input, expected_output in training_data:
            input_tokens = self.tokenizer(text_input, return_tensors="pt")["input_ids"].to(self.device)
            output_tokens = self.tokenizer(expected_output, return_tensors="pt")["input_ids"].to(self.device)
            processed_data.append((input_tokens, output_tokens))
        return processed_data

    def train_model(self, filename="None", epochs=15):
        if not filename or filename == "None":
            filename = "ai_Model/chat_training_data.json"
        print(f"📂 Laddar träningsdata från: {filename}")
        try:
            with open(filename, 'r', encoding="utf-8") as f:
                raw_data = json.load(f)
                print(f"✅ Träningsdata laddad, antal poster: {len(raw_data)}")
        except Exception as e:
            print(f"❌ Kunde inte läsa träningsdata! {e}")
            return
        
        training_data = []
        for item in raw_data:
            chat = item.get("chat_data", None)
            if chat:
                question = chat.get("question", "")
                answer = chat.get("answer", "")
                if question and answer:
                    training_data.append((question, answer))

        training_tokens = self.preprocess_data(training_data)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_function = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for input_tokens, target_tokens in training_tokens:
                optimizer.zero_grad()
                seq_len = input_tokens.size(1)
                mask = generate_square_subsequent_mask(seq_len).to(self.device)
                predictions = self.model(input_tokens, mask=mask)
                
                loss = loss_function(predictions.view(-1, predictions.size(-1)), target_tokens.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"✅ Epoch {epoch + 1}/{epochs} — Loss: {total_loss:.4f}")

    def load_preferences(self):
        if hasattr(self.db_connector, "fetch_user_preferences"):
            return self.db_connector.fetch_user_preferences()
        else:
            print("❌ `fetch_user_preferences()` saknas i `DatabaseConnector`")
            return {}

    def save_model(self, filename="aether_trained_model.pth"):
        torch.save(self.model.state_dict(), filename)
        print(f"✅ Modellens viktningar har sparats i {filename}")

    def load_model(self, filename="aether_trained_model.pth"):
        try:
            self.model.load_state_dict(torch.load(filename, map_location=self.device))
            self.model.eval()
            print(f"✅ Tränade viktningar har laddats från {filename}")
        except FileNotFoundError:
            print(f"❌ Filen {filename} hittades inte!")

    def generate_text(self, prompt, max_length=50): 
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        for _ in range(max_length):
            embedded = self.embedding(input_ids)  # Gör detta varje gång input_ids uppdateras
            seq_len = input_ids.size(1)
            mask = generate_square_subsequent_mask(seq_len).to(self.device)
            with torch.no_grad():
                outputs = self.model(embedded, mask=mask)
            next_token_logits = outputs[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            input_ids = torch.cat((input_ids, next_token_id), dim=1)

            if next_token_id.item() in [self.tokenizer.sep_token_id, self.tokenizer.eos_token_id]:
                break

        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return generated_text

    def run(self, user_input):
        print(f"\nGOAL: {user_input}")

        if user_input.lower().strip() == "train model":
            print("🧠 Startar träning av modellen...")
            self.train_model(filename="ai_Model/chat_training_data.json", epochs=15)
            self.save_model("ai_Model/aether_chat_trained.pth")
            print("✅ Träning klar och modellen sparad!")
            return

        if "your name" in user_input.lower() or "what's your name" in user_input.lower():
            reply = f"My name is {self.name}."
            self.memory.add(f"User asked my name. I replied: {reply}")
            print(f"\nAETHER: {reply}")
            self.db_connector.insert_conversation(self.name or "User", user_input, reply)
            return reply

        if "show memories" in user_input.lower():
            return "\n".join(self.memory.fetch_all_memories()) or "No memories."

        if "calculate" in user_input.lower() or "solve" in user_input.lower():
            expression = user_input.lower().replace("calculate", "").replace("solve", "").strip()
            result = self.memory.calculator_tool(expression)
            print(f"\nAETHER: {result}")
            self.db_connector.insert_conversation("User", user_input, result)
            return result

        if "wikipedia" in user_input.lower() or "search wiki" in user_input.lower():
            query = user_input.lower().replace("wikipedia", "").replace("search wiki", "").strip()
            result = self.memory.wikipedia_tool(query)
            print(f"\nAETHER: {result}")
            self.db_connector.insert_conversation("User", user_input, result)
            return result

        generated_text = self.generate_text(user_input)
        print("\nGenerated Response:\n", generated_text)
        self.memory.add(user_input)
        self.db_connector.insert_conversation("User", user_input, generated_text)
        return generated_text


# --- Kör agenten ---
if __name__ == "__main__":
    db_con = DatabaseConnector()
    agent = AetherAgent(db_con)
    
    # Om du vill träna först:
    # agent.train_model(filename="chat_training_data.json", epochs=10)
    # agent.save_model("aether_chat_trained.pth")

    # Annars ladda modellen om den finns:
    agent.load_model("aether_chat_trained.pth")

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
