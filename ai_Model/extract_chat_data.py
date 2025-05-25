import json

# Ladda den ursprungliga JSON-filen
with open("training_data_10000.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extrahera bara chat_data (fråga och svar)
chat_training_data = []
for item in data:
    chat_item = item.get("chat_data", {})
    if "question" in chat_item and "answer" in chat_item:
        chat_training_data.append({
            "input": chat_item["question"],
            "output": chat_item["answer"]
        })

# Spara extraherad data till en ny fil
with open("chat_training_data.json", "w", encoding="utf-8") as f:
    json.dump(chat_training_data, f, ensure_ascii=False, indent=2)

print("✅ Chat-träningsdata sparad i 'chat_training_data.json'")
