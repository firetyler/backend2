import numpy as np
import sys
import faiss
import torch
import wikipedia
import os
import re
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Import the DatabaseConnector from the same directory
from DatabaseConnector import DatabaseConnector

# Avoid duplicate OpenMP errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load tokenizer and larger GPT-2 model
print("Loading GPT-2 Medium model...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
model.eval()

# Ensure pad_token_id is set
tokenizer.pad_token = tokenizer.eos_token
MAX_INPUT_LENGTH = 1024


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
        return [self.texts[i] for i in I[0] if i < len(self.texts)]

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


def think(prompt):
    print("Thinking...")

    inputs = tokenizer.encode(prompt, return_tensors="pt")

    if inputs.shape[1] > MAX_INPUT_LENGTH:
        inputs = inputs[:, -MAX_INPUT_LENGTH:]

    attention_mask = torch.ones(inputs.shape, dtype=torch.long)

    with torch.no_grad():
        output = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


class AetherAgent:
    def __init__(self):
        self.memory = AetherMemory()
        self.name = "Aether"  # Identity

    def learn_name(self, user_input):
        if "your name is" in user_input.lower():
            parts = user_input.lower().split("your name is")
            if len(parts) > 1:
                guessed_name = parts[1].strip().split(" ")[0]
                self.name = guessed_name.capitalize()
                self.memory.add(f"My name is {self.name}.")
                return f"Thanks! I'll remember my name is {self.name}."
        return None

    def find_name_in_memory(self):
        memories = self.memory.texts
        for mem in memories:
            if "my name is" in mem.lower():
                name = mem.split("is")[-1].strip().split(".")[0]
                return name
        return None

    def run(self, user_input):
        print(f"\nGOAL: {user_input}")
        
        # 1. Learn name if user teaches
        name_learned = self.learn_name(user_input)
        if name_learned:
            print(f"\nAETHER: {name_learned}")
            return
        
        # 2. Answer 'your name' from memory immediately
        if "your name" in user_input.lower() or "what's your name" in user_input.lower():
            if not self.name:
                self.name = self.find_name_in_memory()
            if self.name:
                reply = f"My name is {self.name}."
            else:
                reply = "I don't know my name yet. You can tell me!"
            self.memory.add(f"User asked my name. I replied: {reply}")
            print(f"\nAETHER: {reply}")
            return

        # 3. Otherwise normal memory + thinking
        related = self.memory.search(user_input, k=3)

        if related:
            print("\nRecalled Memory:")
            for m in related:
                print("-", m)
        else:
            print("\nNo memory found.")

        context = "\n".join(related[:2])
        prompt = f"{context}\nThought: To achieve '{user_input}', I should"
        thought = think(prompt)
        print("\nThought:\n", thought)

        # 4. Tools if needed
        goal_lower = user_input.lower()
        if "calculate" in goal_lower:
            expression = goal_lower.replace("calculate", "").strip()
            action_result = calculator_tool(expression)
        elif any(q in goal_lower for q in ["who", "what", "where", "when", "why", "how"]):
            action_result = wikipedia_tool(user_input)
        else:
            action_result = "No tool used."

        print("\nAction Result:\n", action_result)

        reflection = f"Goal: {user_input}\nThought: {thought}\nAction: {action_result}"
        self.memory.add(reflection)
        print("\nReflection stored.")


if __name__ == "__main__":
    db_con = DatabaseConnector()
    agent = AetherAgent()
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

