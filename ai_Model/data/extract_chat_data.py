import json

# Load the original JSON file
with open("training_data_10000_en.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract only chat_data (question and answer)
chat_training_data = []
for item in data:
    chat_item = item.get("chat_data", {})
    if "question" in chat_item and "answer" in chat_item:
        chat_training_data.append({
            "input": chat_item["question"],
            "output": chat_item["answer"]
        })

# Save extracted data to a new file
with open("chat_training_data_en.json", "w", encoding="utf-8") as f:
    json.dump(chat_training_data, f, ensure_ascii=False, indent=2)

print("✅ Chat training data saved in 'chat_training_data_en.json'")

