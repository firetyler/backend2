from DatabaseConnector import DatabaseConnector

def export_to_txt(file_path="training_data.txt"):
    db = DatabaseConnector()
    conversations = db.fetch_all_conversations()

    with open(file_path, "w", encoding="utf-8") as f:
        for convo in conversations:
            user_input = convo["user_input"]
            ai_response = convo["ai_response"]
            f.write(f"User: {user_input}\nAI: {ai_response}\n\n")

    print(f"Träningsdata sparat till {file_path}")

if __name__ == "__main__":
    export_to_txt()