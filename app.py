from flask import Flask, request, jsonify
from convert import image_to_braille
from PIL import Image
import io

app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    image = Image.open(image_file.stream).convert("RGB")

    try:
        braille_text = image_to_braille(image)
        return jsonify({"braille": braille_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)