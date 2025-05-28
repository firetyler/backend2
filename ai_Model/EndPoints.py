import json
from EndPointLog import get_logger
from flask import Flask, request, jsonify
from aether2 import AetherAgent
from DatabaseConnector import DatabaseConnector

# ✅ Initialize logger correctly
logger = get_logger("EndPoints")

# ✅ Create Flask app
app = Flask(__name__)

# ✅ Initialize database connector BEFORE usage
db_connector = DatabaseConnector()

# ✅ Create AI agent
agent = AetherAgent(db_connector)


# ✅ Load configuration
config = agent.load_config("ai_Model/config.json")
agent.train_model(config_path="ai_Model/config.json")
agent.save_model(filename=config["model_path"])
# ✅ Build tokenizer vocabulary correctly
try:
    filenames = config.get("train_data_paths", ["ai_Model/chat_training_data.json"])  # Ensure list format
    training_data = []

    for filename in filenames:
        if not isinstance(filename, str):
            logger.error(f"invalid filename format: {filename}. Expected a string.")
            continue

        try:
            with open(filename, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                valid_data = [(d["input"], d["output"]) for d in raw_data if "input" in d and "output" in d]
                training_data.extend(valid_data)
        except Exception as e:
            logger.error(f" Failed to load training data from {filename}: {e}")

    if training_data:
        all_texts = [q for q, _ in training_data] + [a for _, a in training_data]
        agent.tokenizer.build_vocab(all_texts)
        logger.info(f"Tokenizer vocabulary updated with {len(all_texts)} examples.")
    else:
        logger.warning(" No valid training data loaded for vocabulary.")

except Exception as e:
    logger.error(f" Failed to build tokenizer vocabulary: {e}")

# ✅ Initialize and load the model correctly
agent.initialize_model(config)
try:
    agent.load_model()
    logger.info("Model loaded.")
except Exception as e:
    logger.warning(f" Model not found – training new model... ({e})")
    agent.train_model()
    agent.save_model()
    logger.info("New model trained and saved.")


def clean_response(raw_output):
    """Fix nested JSON encoding issues and extract the correct response."""
    try:
        parsed = json.loads(raw_output)  # Decode first level
        return parsed.get("response", parsed)  # Extract actual response if available
    except json.JSONDecodeError:
        return raw_output  # Return original if decoding fails






# ✅ Route for text generation
@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        user_input = data.get("prompt", "")

        if not user_input:
            return jsonify({"error": "No prompt provided"}), 400

        agent_response = agent.run(user_input)

        # 🔹 Use `clean_response()` to ensure correct JSON formatting
        cleaned_response = clean_response(agent_response) if agent_response else None

        if not cleaned_response:
            return jsonify({"error": "Agent failed to generate a response"}), 500

        db_connector.insert_conversation("User", user_input, cleaned_response)

        return jsonify({"response": cleaned_response}), 200

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        return jsonify({"error": str(e)}), 500

# ✅ Route for answering questions
@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        user_input = data.get("prompt", "")

        if not user_input:
            return jsonify({"error": "No prompt provided"}), 400

        logger.info(f"❔ Question received: {user_input}")
        agent_response = agent.run(user_input)

        if not agent_response:
            return jsonify({"error": "Agent failed to generate a response"}), 500

        db_connector.insert_conversation("User", user_input, agent_response)
        return jsonify({"input": user_input, "output": agent_response}), 200

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        return jsonify({"error": str(e)}), 500

# ✅ Run Flask app
if __name__ == "__main__":
    logger.info("Starting Flask server...")
    app.run(port=5000, debug=False)
