from flask import Flask, request, jsonify
from aether2 import AetherAgent
from DatabaseConnector import DatabaseConnector

app = Flask(__name__)
db_connector = DatabaseConnector()
agent = AetherAgent(db_connector)
config = agent.load_config("ai_Model/config.json")  # justera sökvägen om nödvändigt
agent.tokenizer.build_vocab(["Hello world", "My name is AI", "What's your name?"])
agent.initialize_model(config)

try:
    agent.load_model()
    print("✅ Modell laddad.")
except:
    print("⚠️ Modell ej hittad – tränar ny modell...")
    agent.train_model()
    agent.save_model()

@app.route("/generate", methods=["POST"])

def generate():
    try:
        data = request.get_json()
        user_input = data.get("prompt", "")
        
        if not user_input:
            return jsonify({"error": "No prompt provided"}), 400

        print(f"Received input: {user_input}")

        # Viktigt: Får tillbaka en sträng
        agent_response = agent.run(user_input)  

        if not agent_response:
            return jsonify({"error": "Agent failed to generate a response"}), 500

        # Spara till databas
        db_connector.insert_conversation("User", user_input, agent_response)

        return jsonify({"response": agent_response}), 200

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": "An error occurred while processing the request."}), 500
    
    
@app.route('/history', methods=['GET'])
def history():
    history_data = db_connector.fetch_history()
    if not history_data:
        return jsonify({"error": "No history found"}), 404
    return jsonify(history_data), 200


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_input = data.get("prompt", "")
    if not user_input:
        return jsonify({"error": "No prompt provided"}), 400

    agent_response = agent.run(user_input)
    if not agent_response:
        return jsonify({"error": "Agent failed to generate a response"}), 500

    db_connector.insert_conversation("User", user_input, agent_response)
    return jsonify({"input": user_input, "output": agent_response}), 200




if __name__ == "__main__":
    app.run(port=5000)
