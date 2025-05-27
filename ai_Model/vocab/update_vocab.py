import json

def convert_words_to_vocab(input_file, output_file):
    """
    Konverterar words_dictionary.json till tokenizer_vocab.json
    """
    try:
        # 🔹 Läs in filen
        with open(input_file, "r", encoding="utf-8") as f:
            words_dict = json.load(f)

        print(f"✅ Laddade {len(words_dict)} ord från {input_file}")

        # 🔹 Skapa rätt vokabulärformat
        word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}

        index = len(word2idx)  # Börja efter specialtokens
        for word in words_dict.keys():
            word2idx[word] = index
            idx2word[index] = word
            index += 1

        # 🔹 Spara till `tokenizer_vocab.json`
        vocab_data = {"word2idx": word2idx, "idx2word": idx2word}
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, indent=4)

        print(f"✅ Vocab sparad till {output_file} med {len(word2idx)} ord!")

    except FileNotFoundError:
        print(f"❌ Filen {input_file} hittades inte. Kontrollera sökvägen!")