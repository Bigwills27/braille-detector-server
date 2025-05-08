from flask import Flask, request, jsonify
from convert import image_to_braille
from PIL import Image
from flask_cors import CORS
import io
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/upload", methods=["POST"])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    image = Image.open(image_file.stream).convert("RGB")

    try:
        braille_text = image_to_braille(image)
        return jsonify({"text": braille_text})  # Match the expected response format
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# For local development
if __name__ == "__main__":
    app.run(debug=True)

# For production deployment on Render
if __name__ != "__main__":
    # Enable proper gunicorn integration
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)