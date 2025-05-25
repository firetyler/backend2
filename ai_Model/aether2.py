import torch
import torch.nn as nn
import math
import re
import numpy as np
import faiss
import json
import os
import wikipedia
from transformers import AutoTokenizer
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
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout, vocab_size=30522):
        super(StackedTransformer, self).__init__()
        self.embed_size = embed_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_encoding = PositionalEncoding(embed_size)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        # Viktigt output-lager som mappar embed_size till vocab_size
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x, mask):
        x = self.token_embedding(x)
        seq_length = x.shape[1]
        x = self.position_encoding(x[:, :seq_length])
        for layer in self.layers:
            x = layer(x, mask[:, :seq_length])
        x = self.dropout(x)
        logits = self.fc_out(x)  # (batch_size, seq_len, vocab_size)
        return logits
        
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

    def calculator_tool(self, input_str):
        allowed_chars = re.compile(r"^[\d\s\.\+\-\*/\(\)]+$")
        input_str = input_str.replace('–', '-').replace('−', '-')
        if not allowed_chars.match(input_str):
            return "Invalid characters in expression."
        try:
            result = eval(input_str)
            return f"Calculated result: {result}"
        except Exception as e:
            return f"Error evaluating expression: {e}"

    def wikipedia_tool(self, query):
        try:
            summary = wikipedia.summary(query, sentences=2)
            return f"Wikipedia result: {summary}"
        except Exception as e:
            return f"Wikipedia error: {e}"

# 6. AetherAgent
class AetherAgent:
    def __init__(self, db_connector):
       # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.db_connector = db_connector
        self.memory = AetherMemory()
        self.name = self.load_name()
        self.user_preferences = self.load_preferences()
        self.model = StackedTransformer(embed_size=256, num_layers=400, heads=8, forward_expansion=4, dropout=0.1)
       # self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

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
            input_tokens = self.tokenizer(text_input, return_tensors="pt")["input_ids"]
            output_tokens = self.tokenizer(expected_output, return_tensors="pt")["input_ids"]
            processed_data.append((input_tokens, output_tokens))
        return processed_data

    def train_model(self, filename="None", epochs=15):
        if not filename: 
             filename = "ai_Model/chat_training_data.json"
        print(f"📂 Laddar träningsdata från: {filename}")
        try:
            with open(filename, 'r', encoding="utf-8") as f:
                training_data = json.load(f)
                print(f"✅ Träningsdata laddad, antal poster: {len(training_data)}")
        except:
            print("❌ Kunde inte läsa träningsdata!")
            return
        
        # Extrahera chat_data från varje post
        training_data = []
        for item in training_data:
            chat = item.get("chat_data", None)
            if chat:
                question = chat.get("question", "")
                answer = chat.get("answer", "")
                if question and answer:
                    training_data.append((question, answer))




        training_tokens = self.preprocess_data(training_data)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_function = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss = 0.0
            for input_tokens, target_tokens in training_tokens:
                optimizer.zero_grad()
                predictions = self.model(input_tokens, mask=torch.ones_like(input_tokens))
                loss = loss_function(predictions.squeeze(), target_tokens.squeeze())
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
            self.model.load_state_dict(torch.load(filename))
            self.model.eval()
            print(f"✅ Tränade viktningar har laddats från {filename}")
        except FileNotFoundError:
            print(f"❌ Filen {filename} hittades inte!")

    def run(self, user_input):
        print(f"\nGOAL: {user_input}")

    # Specialfall
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

        tokens = self.tokenizer(user_input, return_tensors="pt")
        input_ids = tokens["input_ids"]
        mask = torch.ones(input_ids.shape, dtype=torch.int64)  # Mask av 1:or, ingen maskering

        response = self.model(input_ids, mask)
        if response is None:
            return "Sorry, something went wrong with the model output."

        print("input_ids shape:", input_ids.shape)
        print("response shape:", response.shape)

    # response shape: (batch_size, seq_len, vocab_size)
        response_ids = response.argmax(dim=-1)  # argmax på sista dim (vocab)
        generated_text = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)

        print("\nGenerated Response:\n", generated_text)
        self.memory.add(user_input)
        self.db_connector.insert_conversation("User", user_input, generated_text)
        return generated_text


# 7. Starta Aether
if __name__ == "__main__":
    db_con = DatabaseConnector()
    agent = AetherAgent(db_con)
    agent.train_model(filename="chat_training_data.json", epochs=10)
    agent.save_model("aether_chat_trained.pth")

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