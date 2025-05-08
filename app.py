from flask import Flask, request, jsonify
from convert import image_to_braille
from PIL import Image
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": "*"}})  # Enable CORS for the upload route

@app.route("/upload", methods=["POST"])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    image = Image.open(image_file.stream).convert("RGB")

    try:
        braille_text = image_to_braille(image)
        return jsonify({"text": braille_text})  # Return Braille text
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# For local development
if __name__ == "__main__":
    app.run(debug=True, port=5000)
