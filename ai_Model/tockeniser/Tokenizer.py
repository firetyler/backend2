import math
import os
import sys
import torch
import torch.nn as nn


import json
import re
import unicodedata
from collections import Counter
from logToken import get_logger
sys.stdout.reconfigure(encoding='utf-8')  # Forces UTF-8 output
logger = get_logger("SimpleTokenizer")
class SimpleTokenizer:
    def __init__(self, min_freq=1,vocab_file=None,max_vocab_size=50000):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.vocab_size = 4
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        if self.vocab_size >= self.max_vocab_size:
            self.max_vocab_size += 5000  # Expand limit safely
            
        if vocab_file and os.path.exists(vocab_file):
            self.load_vocab(vocab_file)
            logger.info(f"Loaded vocabulary from {vocab_file} with {self.vocab_size} words.")

    def _clean_text(self, text):
        # Normalisera unicode
        text = unicodedata.normalize("NFKD", text)
        text = text.lower()
        
        # Ersätt punkter och kommatecken med " space + tecken + space "
        text = re.sub(r"([.,])", r" \1 ", text)
        
        # Ta bort andra tecken som inte är bokstäver, siffror, mellanslag eller punkter/komman
        text = re.sub(r"[^a-z0-9\s.,]", "", text)
        
        # Rensa extra mellanslag
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

    def build_vocab(self, texts):
        """
        Bygg vokabulär från lista av texter.
        Tar med alla ord som förekommer minst min_freq gånger.
        Sparar och verifierar vokabulären efter byggnad.
        """
        counter = Counter()

        # Steg 1: Räkna ord med korrekt normalisering
        for text in texts:
            cleaned = self._clean_text(text)
            counter.update(cleaned.split())

        # Steg 2: Lägg till alla ord som uppfyller min_freq och inte finns redan
        new_words_added = 0
        for word, freq in counter.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
                new_words_added += 1

        logger.info(f"Vocab built with {self.vocab_size} words! Added {new_words_added} new words.")

        # Steg 3: Spara vokabulär till disk
        self.save_vocab("tokenizer_vocab.json")

        # Steg 4: Verifiera att sparningen gick bra
        self.verify_vocab("tokenizer_vocab.json")

        def update_vocab(self, filename, new_words):
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

            logger.debug(f"Vocab uppdaterad! Nya ord: {new_words}")

    def encode(self, text):
        """Tokenize input text and dynamically expand vocabulary if needed."""
        cleaned = self._clean_text(text)
        tokens = []

        for w in cleaned.split():
            if w not in self.word2idx:
                # ✅ Ensure vocabulary does not exceed limit
                if self.vocab_size >= self.max_vocab_size:
                    logger.error(f"Cannot add '{w}'. Vocabulary limit ({self.max_vocab_size}) reached.")
                    tokens.append(self.word2idx["<UNK>"])
                    continue  # Skip adding the word

                logger.warning(f"Adding new word '{w}' to vocabulary dynamically.")
                self.word2idx[w] = self.vocab_size
                self.idx2word[self.vocab_size] = w
                self.vocab_size += 1
                self.save_vocab("tokenizer_vocab.json")  # 🔹 Update vocabulary

            tokens.append(self.word2idx[w])

        return [self.word2idx["<SOS>"]] + tokens + [self.word2idx["<EOS>"]]

    def decode(self, token_ids):
        words = []
        for i in token_ids:
            if i not in self.idx2word:
                continue
            w = self.idx2word[i]
            if w in ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]:
                # Hoppa över specialtokens
                continue
            words.append(w)
            if w == "<EOS>":  # Stoppa om slut-token hittas (om vi inkluderar <EOS> i output)
                break
        return " ".join(words)
    
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
    
    def verify_vocab(self, filename="tokenizer_vocab.json"):
        """Verifierar att vokabulären har sparats och innehåller tillräckligt många ord."""
        if not os.path.exists(filename):
            logger.warning(f"Vocab file {filename} not found!")
            return

        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Vocab file successfully saved. Contains {len(data['word2idx'])} words.")
        logger.debug(f"First 10 words in file: {list(data['word2idx'].keys())[:10]}")

    def save_vocab(self, filename="tokenizer_vocab.json"):
        """Save the updated vocabulary and verify its integrity."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({"word2idx": self.word2idx, "idx2word": self.idx2word}, f, indent=4)
        logger.info(f" Vocabulary saved successfully: {filename}")

        # ✅ Immediately verify the vocabulary to ensure consistency
        self.verify_vocab(filename)


    def load_vocab(self, filename):
        if not os.path.exists(filename):
            logger.warning(f"Vocab file {filename} not found! Rebuilding it from training data...")
            
            # 🔹 Hämta ord från träningsfiler istället för en liten hårdkodad lista
            training_data_files = ["ai_Model/chat_training_data.json", "ai_Model/chat_training_data_en.json"]
            texts = []

            for file in training_data_files:
                if os.path.exists(file):
                    with open(file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        texts.extend(data)  # Antag att träningsdatan är en lista av texter

            if not texts:
                logger.warning("No training data found! Using fallback vocabulary.")
                texts = ["hello", "world", "capital", "sweden", "question"]

            self.build_vocab(texts)  # Bygg vokabulären från hela träningsdatabasen
            self.save_vocab(filename)
            self.verify_vocab("tokenizer_vocab.json")

        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.word2idx = data["word2idx"]
        self.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        self.vocab_size = len(self.word2idx)
        logger.info(f"Vocabulary loaded from {filename} with {self.vocab_size} words.")

