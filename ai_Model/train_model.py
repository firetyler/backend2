import json
import os
import sys
import torch
import torch.nn as nn
import math
import re
from torch.utils.data import Dataset,DataLoader
from AetherMemory import AetherMemory
from SimpleTokenizer import AdvancedSentenceTransformer, StackedTransformer, SimpleTokenizer
from mask_utils import generate_square_subsequent_mask
sys.stdout.reconfigure(encoding='utf-8')  # Forces UTF-8 output
from DatabaseConnector import DatabaseConnector
from logger_setup import get_logger
from code_executor import CodeExecutor

logger = get_logger("aether2")
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
        try:
            torch.save(checkpoint, filename)
            logger.info(f"Checkpoint saved: {filename}")
        except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")

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
            vocab_size = len(self.tokenizer.word2idx)
            print(f"📌 Using vocab_size = {vocab_size} for model initialization.")  

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

def train_model(self, config_path="ai_Model/config.json"):
        """Träna AI-modellen med data från flera JSON-filer."""

        # ✅ Ladda konfiguration
        config = self.load_config(config_path)
        epochs = config.get("epochs", 10)
        lr = config.get("learning_rate", 0.001)
        batch_size = config.get("batch_size", 32)
        filenames = config.get("train_data_paths", [])

        # ✅ Debugging: Kontrollera filerna
        print(f"🔎 Training files listed in config: {filenames}")
        print(f"📂 Current working directory: {os.getcwd()}")

        training_data = []
        for filename in filenames:
            if not isinstance(filename, str):  
                logger.error(f"Invalid filename format: {filename}. Expected a string.")
                continue  

            # ✅ Kontrollera att träningsfilen existerar
            if not os.path.exists(filename):
                logger.error(f"Training data file not found: {filename}")
                continue  

            try:
                with open(filename, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)

                    # ✅ Se till att JSON är korrekt strukturerad
                    if isinstance(raw_data, dict):  
                        raw_data = raw_data.get("data", [])  

                    valid_data = [(d["input"], d["output"]) for d in raw_data if "input" in d and "output" in d]
                    training_data.extend(valid_data)
                    print(f"✅ Loaded {len(valid_data)} examples from {filename}")

            except Exception as e:
                logger.error(f"Failed to load training data from {filename}: {e}")

        # ✅ Kontrollera om träningsdata har laddats korrekt
        if not training_data:
            logger.error("No valid training data loaded. Aborting training...")
            return

        # ✅ Förbered dataset & DataLoader
        dataset = self.ChatDataset(training_data, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=self.collate_fn)

        # ✅ Fix: Se till att `self.model` är korrekt initialiserad innan träning
        vocab_size = len(self.tokenizer.word2idx)
        print(f"📌 Using vocab_size = {vocab_size} for model initialization.")

        self.model = StackedTransformer(
            embed_size=config.get("embedding_dim", 256),
            vocab_size=vocab_size,
            num_layers=config.get("num_layers", 6),
            heads=config.get("heads", 8),
            forward_expansion=config.get("forward_expansion", 4),
            dropout=config.get("dropout", 0.1)
        )

        # ✅ Initialisera optimizer och loss-funktion
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        self.model.train()

        # ✅ Hantera checkpoints
        checkpoint_path = "checkpoint_latest.pth"
        start_epoch = 0
        if os.path.exists(checkpoint_path):
            logger.info(f"Found checkpoint: Loading from {checkpoint_path}...")
            start_epoch = self.load_checkpoint(checkpoint_path, optimizer, config)
        else:
            logger.warning("No checkpoint found – starting training from scratch...")
            start_epoch = 0

        # ✅ Starta träningsloop
        logger.info(f" Starting model training for {epochs} epochs...")
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

            # ✅ Spara checkpoint efter varje epoch
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