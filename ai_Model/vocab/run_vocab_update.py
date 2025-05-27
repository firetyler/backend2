from ai_Model.vocab.update_vocab import convert_words_to_vocab

# 🔹 Ange filvägar
input_file = "C:\\Users\\olive\\Documents\\java\\backend\\backend\\words_dictionary.json"
output_file = "C:\\Users\\olive\\Documents\\java\\backend\\backend\\tokenizer_vocab.json"

# 🔹 Kör uppdateringen manuellt
convert_words_to_vocab(input_file, output_file)

print("✅ Tokenizer-vokabulär har uppdaterats manuellt!")