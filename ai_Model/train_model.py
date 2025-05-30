# trainer.py
import sys
import os
import json
import torch
import torch.nn as nn

from AetherMemory import AetherMemory
from SimpleTokenizer import StackedTransformer, SimpleTokenizer, AdvancedSentenceTransformer
from chatDataset import ChatDataset
from logTrainer import get_logger
from mask_utils import generate_square_subsequent_mask
from torch.utils.data import DataLoader

sys.stdout.reconfigure(encoding='utf-8')
logger = get_logger("training")


class Trainer:
    def __init__(self, tokenizer=None, device=None, embed_size=256):
        self.tokenizer = SimpleTokenizer()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_size = embed_size
        self.model = None
        self.token_embedding = None

    def load_config(self, path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Could not read the config file: {e}")
            return {}

    def save_checkpoint(self, epoch, optimizer, filename):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "tokenizer": self.tokenizer.word2idx
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename, optimizer, config):
        if not os.path.isfile(filename):
            logger.warning(f"No checkpoint file found: {filename}")
            return 0

        checkpoint = torch.load(filename, map_location=self.device)

        self.tokenizer.load_vocab(config.get("tokenizer_vocab_path", "tokenizer_vocab.json"))
        logger.info(f"Tokenizer vocab loaded with size {self.tokenizer.vocab_size}")

        new_vocab_size = self.tokenizer.vocab_size
        self.token_embedding = nn.Embedding(new_vocab_size, self.embed_size, padding_idx=0)
        logger.info(f"Token embedding updated with vocab size {new_vocab_size}")

        self.model = StackedTransformer(
            embed_size=self.embed_size,
            vocab_size=new_vocab_size,
            num_layers=config.get("num_layers", 4),
            heads=config.get("heads", 8),
            forward_expansion=config.get("forward_expansion", 4),
            dropout=config.get("dropout", 0.1)
        ).to(self.device)

        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint["model_state_dict"].items()
                           if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Optimizer state loaded from checkpoint.")

        start_epoch = checkpoint.get("epoch", 0) + 1
        logger.info(f"Checkpoint loaded from {filename}, starting from epoch {start_epoch}")

        vocab_path = config.get("tokenizer_vocab_path", "tokenizer_vocab.json")
        self.tokenizer.save_vocab(vocab_path)

        if not os.path.exists(vocab_path):
            logger.warning("Vocab file not found after saving! Double-check file permissions.")
        else:
            logger.info(f"Vocabulary successfully saved to {vocab_path}.")

        return start_epoch

    def save_model(self, filename="aether_model.pth"):
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename="aether_model.pth"):
        if self.model is None:
            raise RuntimeError("Model not initialized.")
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
        self.model.eval()

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

    def train_model(self, config_path="ai_Model/config.json"):
        """Träna AI-modellen med data från flera JSON-filer."""

        # ✅ Ladda konfiguration
        config = self.load_config(config_path)
        epochs = config.get("epochs", 10)
        lr = config.get("learning_rate", 0.001)
        batch_size = config.get("batch_size", 32)
        filenames = config.get("train_data_paths", [])

        logger.info(f"Training files listed in config: {filenames}")
        logger.info(f"Current working directory: {os.getcwd()}")

        training_data = []
        for filename in filenames:
            if not isinstance(filename, str):
                logger.error(f"Invalid filename format: {filename}. Expected a string.")
                continue

            if not os.path.exists(filename):
                logger.error(f"Training data file not found: {filename}")
                continue

            try:
                with open(filename, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    if isinstance(raw_data, dict):
                        raw_data = raw_data.get("data", [])

                    valid_data = [(d["input"], d["output"]) for d in raw_data if "input" in d and "output" in d]
                    training_data.extend(valid_data)
                    logger.info(f"Loaded {len(valid_data)} examples from {filename}")

            except Exception as e:
                logger.error(f"Failed to load training data from {filename}: {e}")

        if not training_data:
            logger.error("No valid training data loaded. Aborting training...")
            return

        # ✅ Bygg och spara vokabulär från träningsdata
        all_texts = [q for q, _ in training_data] + [a for _, a in training_data]
        self.tokenizer.build_vocab(all_texts)
        self.tokenizer.save_vocab("tokenizer_vocab.json")
        logger.info(f"Tokenizer updated and saved with {len(self.tokenizer.word2idx)} tokens.")

        dataset = ChatDataset(training_data, self.tokenizer)
        dataloader = ChatDataset(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=self.collate_fn)

        vocab_size = len(self.tokenizer.word2idx)
        logger.info(f"Using vocab_size = {vocab_size} for model initialization.")

        self.model = StackedTransformer(
            embed_size=config.get("embedding_dim", 256),
            vocab_size=vocab_size,
            num_layers=config.get("num_layers", 6),
            heads=config.get("heads", 8),
            forward_expansion=config.get("forward_expansion", 4),
            dropout=config.get("dropout", 0.1)
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        self.model.train()

        checkpoint_path = "checkpoint_latest.pth"
        start_epoch = 0
        if os.path.exists(checkpoint_path):
            logger.info(f"Found checkpoint: Loading from {checkpoint_path}...")
            start_epoch = self.load_checkpoint(checkpoint_path, optimizer, config)
            if start_epoch is None or not isinstance(start_epoch, int):
                logger.warning(f"Invalid start_epoch: {start_epoch}. Setting default to 0.")
                start_epoch = 0
        else:
            logger.warning("No checkpoint found – starting training from scratch...")

        if start_epoch >= epochs:
            logger.warning(f"start_epoch ({start_epoch}) is greater than total epochs ({epochs}). Skipping training.")
            return

        if epochs <= start_epoch:
            logger.warning(f"Epochs ({epochs}) are too low. Adjusting to {start_epoch + 10}.")
            epochs = start_epoch + 10

        logger.info(f"Starting model training for {epochs} epochs...")

        for epoch in range(start_epoch, epochs):
            total_loss = 0.0
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()

                sz = x.size(1)
                if sz < 1:
                    logger.error(f"Invalid mask size: {sz}. Skipping batch.")
                    continue

                mask = generate_square_subsequent_mask(sz).to(self.device)
                out = self.model(x, mask)

                loss = criterion(out.view(-1, out.size(-1)), y.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
            self.save_checkpoint(epoch + 1, optimizer, checkpoint_path)

        logger.info("Training complete! Model saved successfully.")

        # === Ladda modellen och tokenizern efter träning ===
        logger.info("Reloading model and tokenizer after training...")

        # Ladda tokenizer vocab
        self.tokenizer.load_vocab("tokenizer_vocab.json")
        logger.info(f"Tokenizer vocab reloaded with {len(self.tokenizer.word2idx)} tokens.")

        # Ladda modell från checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()  # sätt modellen i eval-läge för inferens

        logger.info("Model reloaded and ready for inference.")
