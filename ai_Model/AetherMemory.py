# --- Minnesmodul ---



import faiss
import numpy as np
import torch
import wikipedia

from AetherMemoryLog import get_logger
import re

logger = get_logger("AetherMemory")
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