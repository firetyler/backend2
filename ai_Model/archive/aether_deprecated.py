from warnings import deprecated

@deprecated("Use another function instead")
def old_function():
    pass


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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
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
        results = []
        for i in I[0]:
             if 0 <= i < len(self.texts):
                 results.append(self.texts[i])
        return results

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

    inputs = inputs.to(device)  # Skicka till GPU om tillgängligt
    attention_mask = torch.ones(inputs.shape, dtype=torch.long).to(device)

    with torch.no_grad():
            # Använd fp16 om CUDA finns
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                    output = model.generate(
                        inputs,
                        attention_mask=attention_mask,
                        max_new_tokens=80,  # 🔧 Mindre = snabbare
                        top_k=30,           # 🔧 Mindre = mindre slump
                        top_p=0.90,
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=True
                    )
        else:
                output = model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=80,
                    top_k=30,
                    top_p=0.90,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True
                )

    return tokenizer.decode(output[0], skip_special_tokens=True)

def extract_search_term(text):
        """
        Attempts to extract a clean search term from a user's question.
        Removes common question phrases and trailing punctuation.
        """
        text = text.strip().lower()
        question_starters = [
            r"what is", r"who is", r"where is", r"when is", r"why is", r"how is",
            r"what's", r"who's", r"where's", r"when's", r"how's",
            r"tell me about", r"give me information about", r"do you know",
            r"can you tell me about", r"i want to know about", r"explain", r"define"
        ]
        for starter  in question_starters:
            pattern  = rf"^{starter}\s+"
            text = re.sub(pattern, "", text)
        text = text.strip(" ?")       
        return text  # fallback


class AetherAgent:
    def __init__(self, db_connector):
        self.db_connector = db_connector
        self.name = None
        self.memory = AetherMemory()  # Anta att du har en Memory-klass

    def learn_name(self, text):
        if "my name is" in text.lower():
            name = text.split("my name is")[-1].strip().capitalize()
            self.name = name
            return f"Nice to meet you, {self.name}!"
        return None

    def find_name_in_memory(self):
        # Dummy fallback – använd egen logik om du vill
        return "Aether"

    def run(self, user_input):
        print(f"\nGOAL: {user_input}")

        # 1. Learn name
        name_learned = self.learn_name(user_input)
        if name_learned:
            print(f"\nAETHER: {name_learned}")
            self.db_connector.insert_conversation(self.name or "User", user_input, name_learned)
            return name_learned

        # 2. Respond to "your name"
        if "your name" in user_input.lower() or "what's your name" in user_input.lower():
            if not self.name:
                self.name = self.find_name_in_memory()
            if self.name:
                reply = f"My name is {self.name}."
            else:
                reply = "I don't know my name yet. You can tell me!"
            self.memory.add(f"User asked my name. I replied: {reply}")
            print(f"\nAETHER: {reply}")
            self.db_connector.insert_conversation(self.name or "User", user_input, reply)
            return reply

        # 3. Memory recall
        related = self.memory.search(user_input, k=3)
        if related:
            print("\nRecalled Memory:")
            for m in related:
                print("-", m)
        else:
            print("\nNo memory found.")

        # 4. Thought and tools
        context = "\n".join(related[:2]) if related else "No relevant memory."
        reasoning_prompt = f"""
        You are an intelligent AI assistant named Aether.
        User said: "{user_input}"
        Memory:
        {context}
        Step-by-step, think about what the user is really asking for.
        Then decide if you need to use a tool (calculator, Wikipedia), or if you can answer directly using reasoning.
        Thought:
        """



        thought = think(reasoning_prompt)
        print("\nThought:\n", thought)


        thought_lower = thought.lower()
        if "use calculator" in thought_lower or "calculate" in thought_lower:
            expression = re.findall(r"[-+*/().\d\s]+", user_input)
            expression = expression[0] if expression else user_input
            action_result = calculator_tool(expression)
        elif "search wikipedia" in thought_lower or "look up" in thought_lower or "wikipedia" in thought_lower:
            search_term = extract_search_term(user_input)
            print(f"🔍 Wikipedia search term: {search_term}")
            action_result = wikipedia_tool(search_term)
        else:
             action_result = f"Based on reasoning: {thought}"

        print("\nAction Result:\n", action_result)

        # 5. Save memory and conversation
        reflection = f"Goal: {user_input}\nThought: {thought}\nAction: {action_result}"
        self.memory.add(reflection)
        self.db_connector.insert_conversation(self.name or "User", user_input, action_result)
        print("\nReflection stored.")
        return action_result  # ✅ VIKTIGT: detta returneras


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

