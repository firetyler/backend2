import json
import multiprocessing
from flask import Flask, request, jsonify
from aether2 import AetherAgent
from DatabaseConnector import DatabaseConnector
from EndPointLog import get_logger

# ✅ Initialize logger
logger = get_logger("Endpoints")

# ✅ Global database instance shared across AI clones
db_connector = DatabaseConnector()

app = Flask(__name__)

# ✅ Dictionary to store AI instances
agents = {}

def start_agent(instance_id):
    """Starts an AI instance with a shared database."""
    global db_connector
    agent = AetherAgent(db_connector)  # ✅ Shared database connection
    agents[instance_id] = agent

    # ✅ Load configuration and ensure model is trained
    config = agent.load_config("ai_Model/config.json")
    try:
        agent.load_model()
        logger.info(f"AI instance {instance_id} – Model loaded successfully.")
    except Exception as e:
        logger.warning(f"AI instance {instance_id} – Model not found, training a new model... ({e})")
        agent.train_model(config_path="ai_Model/config.json")
        agent.save_model(filename=config["model_path"])
        logger.info(f"AI instance {instance_id} – New model trained and saved.")

    print(f"AI instance {instance_id} is ready.")

@app.route("/ask/<instance_id>", methods=["POST"])
def ask_ai(instance_id):
    """Handles AI queries via Flask."""
    if instance_id not in agents:
        return jsonify({"error": f"AI instance {instance_id} does not exist."}), 404
    
    user_input = request.json.get("prompt", "").strip()
    if not user_input:
        return jsonify({"error": "No prompt provided"}), 400

    logger.info(f"AI instance {instance_id} received a question: {user_input}")
    response = agents[instance_id].run(user_input)

    if not response:
        return jsonify({"error": "AI failed to generate a response"}), 500

    db_connector.insert_conversation(instance_id, user_input, response)
    return jsonify({"instance": instance_id, "response": response}), 200

@app.route("/generate", methods=["POST"])
def generate_response():
    """Generates a response from an AI instance."""
    try:
        data = request.get_json()
        user_input = data.get("prompt", "").strip()

        if not user_input:
            return jsonify({"error": "No prompt provided"}), 400

        logger.info(f"Received request for text generation: {user_input}")

        # 🔹 Select the first available instance by default
        instance_id = list(agents.keys())[0] if agents else "0"
        response = agents[instance_id].run(user_input)

        if not response:
            return jsonify({"error": "AI failed to generate a response"}), 500

        db_connector.insert_conversation(instance_id, user_input, response)
        return jsonify({"response": response}), 200

    except Exception as e:
        logger.error(f"Error in /generate: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    num_instances = 3  # 🔹 Number of AI instances to start
    processes = []

    for i in range(num_instances):
        p = multiprocessing.Process(target=start_agent, args=(i,))
        p.start()
        processes.append(p)

    logger.info(f"Starting Flask server with {num_instances} AI instances...")
    app.run(host="0.0.0.0", port=5000, debug=False)
