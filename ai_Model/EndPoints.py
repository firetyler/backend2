from logger_setup import get_logger
from flask import Flask, app, json, request, jsonify
from aether2 import AetherAgent
from DatabaseConnector import DatabaseConnector
logger = get_logger("EndPoints")
app = Flask(__name__)
# ✅ Initiera databasconnector INNAN du använder den
db_connector = DatabaseConnector()

# ✅ Skapa agent
agent = AetherAgent(db_connector)

# ✅ Läs in config
config = agent.load_config("ai_Model/config.json")

# ✅ Bygg vokabulär
try:
    with open(config.get("train_data_path", "ai_Model/chat_training_data.json"), "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    training_data = [(d['input'], d['output']) for d in raw_data if 'input' in d and 'output' in d]
    all_texts = [q for q, _ in training_data] + [a for _, a in training_data]
    agent.tokenizer.build_vocab(all_texts)
except Exception as e:
    print(f"Failed to load training data: {e}")

# ✅ Initiera och ladda modell
agent.initialize_model(config)
try:
    agent.load_model()
    print("✅ Model loaded.")
except Exception as e:
    print(f"⚠️ Model not found – training new model... ({e})")
    agent.train_model()
    agent.save_model()
# Route for Text Generation
@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        user_input = data.get("prompt", "")

        if not user_input:
            return jsonify({"error": "No prompt provided"}), 400
        
        print(f"Received input: {user_input}")
        if user_input == "train model":
            get_logger.info("🚀 Training process initialized...")
            agent.train_model()
            agent.save_model()
            return jsonify({"response": "Training initialized..."})
        agent_response = agent.run(user_input)

        if not agent_response:
            return jsonify({"error": "Agent failed to generate a response"}), 500

        db_connector.insert_conversation("User", user_input, agent_response)

        return jsonify({"response": agent_response}), 200

    except Exception as e:
        print(f" Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500  # 
        
#  Route for Asking Questions
@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        user_input = data.get("prompt", "")
        if not user_input:
            return jsonify({"error": "No prompt provided"}), 400

        agent_response = agent.run(user_input)
        if not agent_response:
            return jsonify({"error": "Agent failed to generate a response"}), 500

        db_connector.insert_conversation("User", user_input, agent_response)
        return jsonify({"input": user_input, "output": agent_response}), 200
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500  #  Always return JSON

#  Run Flask App
if __name__ == "__main__":
    app.run(port=5000)
