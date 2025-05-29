import faiss
import numpy as np
import torch
import wikipedia
import re
from AetherMemoryLog import get_logger

logger = get_logger("AetherMemory")

class AetherMemory:
    def __init__(self, embed_model, tokenizer, device):
        self.embed_model = embed_model
        self.memories = []
        self.vector_dim = embed_model.embed_size if hasattr(embed_model, "embed_size") else 256  # 🔹 Fix: Säkerställ att embed_size finns
        self.index = faiss.IndexFlatL2(self.vector_dim)  # ✅ Fix: Använd rätt dimension
        self.tokenizer = tokenizer
        self.device = device
        

        # ✅ Kontrollera att `embed_model` har `token_embedding`
        if not hasattr(embed_model, "token_embedding"):
            logger.error("embed_model saknar `token_embedding`! Kontrollera att modellen är korrekt instansierad.")
            raise ValueError("embed_model saknar `token_embedding`")

    def add(self, text):
        vector = self.text_to_vector(text)

        # ✅ Fix: Kontrollera och justera vektorens form innan FAISS används
        vector = np.array(vector, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)  # 🔹 Gör om till (1, d)-format
        if vector.shape[1] != self.vector_dim:
            logger.error(f"Vektor har fel dimension ({vector.shape}), förväntade ({self.vector_dim}).")
            return
        
        self.memories.append(text)
        self.index.add(vector)  # ✅ Fix: Nu säkerställd att vektorn har korrekt shape
        logger.info(f"Memory added: '{text[:50]}...'")

    def text_to_vector(self, text):
        self.embed_model.eval()
        with torch.no_grad():
            token_ids = self.tokenizer.encode(text)

            if not token_ids:
                logger.error("Tokenizer returned empty token list! Using fallback vector.")
                return np.zeros((1, self.vector_dim), dtype=np.float32)  # ✅ Fix: Returnera säker fallback

            if max(token_ids) >= self.embed_model.token_embedding.num_embeddings:
                logger.error(f"Token-index {max(token_ids)} utanför vokabulärens gräns ({self.embed_model.token_embedding.num_embeddings}).")
                return np.zeros((1, self.vector_dim), dtype=np.float32)

            input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            attention_mask = (input_ids != 0).long() if input_ids.numel() > 0 else torch.ones_like(input_ids)

            embedding = self.embed_model(input_ids, attention_mask=attention_mask)
            vector = embedding[0].detach().cpu().numpy().astype('float32')

            # ✅ Fix: Se till att vektorn har rätt shape innan den returneras
            if vector.ndim == 1:
                vector = vector.reshape(1, -1)  # 🔹 (1, d)-format

            return vector

    def semantic_search(self, query, top_k=3):
        query_vector = self.text_to_vector(query)
        if len(self.memories) == 0:
            return []

        distances, indices = self.index.search(query_vector, top_k)
        results = [(self.memories[idx], float(distances[0][i])) for i, idx in enumerate(indices[0]) if idx < len(self.memories)]
        return results

    def calculator_tool(self, expression):
        import ast
        try:
            logger.info(f"Evaluating expression: {expression}")
            safe_expr = re.sub(r"[^0-9+\-*/(). ]", "", expression)

            # ✅ FIX: Bytte `eval()` till `ast.literal_eval()` för bättre säkerhet
            result = ast.literal_eval(safe_expr)
            logger.info(f"Calculation result: {result}")
            return f"Result: {result}"
        except Exception as e:
            logger.error(f"Error evaluating expression '{expression}': {e}")
            return f"Error in calculation: {e}"

    def wikipedia_tool(self, query):
        try:
            summary = wikipedia.summary(query, sentences=2)
            logger.info(f"Wikipedia result for '{query}': {summary[:100]}...")
            return summary
        except wikipedia.exceptions.DisambiguationError as e:
            logger.warning(f"Multiple results for '{query}': {e.options[:3]}")
            return f"Disambiguation error: Try one of: {', '.join(e.options[:3])}"
        except wikipedia.exceptions.PageError:
            logger.error(f"No page found for '{query}'")
            return f"No Wikipedia page found for: {query}"
        except Exception as e:
            logger.error(f"Wikipedia lookup failed for '{query}': {e}")
            return f"Wikipedia lookup failed: {e}"
