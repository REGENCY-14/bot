from flask import Flask, request, jsonify
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_KEY")

# System prompt
system_prompt = """
You are an AI assistant that is an expert in medical health and is part of a hospital system called medicare AI.
You know about symptoms and signs of various types of illnesses.
You can provide expert advice on self-diagnosis options in the case where an illness can be treated using a home remedy.
If a query requires serious medical attention with a doctor, recommend them to book an appointment with our doctors.
If you are asked a question that is not related to medical health respond with "I'm sorry but your question is beyond my functionalities."
Do not use external URLs or blogs to refer.
Format any lists on individual lines with a dash and a space in front of each line.
"""

@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Medical Health Assistant API with GPT-3 language model"

@app.route("/message", methods=["POST"])
def message():
    try:
        # Get user input
        user_input = request.json.get("message")
        if not user_input or not isinstance(user_input, str):
            return jsonify({"error": "Invalid message format"}), 400

        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        )

        # Extract AI response
        reply = response["choices"][0]["message"]["content"].strip()
        return jsonify({"message": reply})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(port=3000, debug=True)
