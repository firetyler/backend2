from flask import Flask, app, request, jsonify
from aether import AetherAgent
from DatabaseConnector import DatabaseConnector

EndPoints = Flask(__name__)
agent = AetherAgent(DatabaseConnector)

@app.rute("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    user_input = data.get("prompt", "")
    if not user_input:
        return jsonify({"error": "No prompt provided"}), 400
    print(f"Received input: {user_input}")
    agent.run(user_input)

    return jsonify({"response": "Processed successfully"}), 200

if __name__ == "__main__":
    EndPoints.run(port=5432)