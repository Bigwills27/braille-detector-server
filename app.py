from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import numpy as np
import torch
import io
from PIL import Image
from ultralyticsplus import YOLO

# Import the conversion functions
def convert_to_braille_unicode(str_input: str, path: str = "./braille_map.json") -> str:
    with open(path, "r") as fl:
        data = json.load(fl)
    if str_input in data.keys():
        str_output = data[str_input]
    return str_output

def parse_xywh_and_class(boxes: torch.Tensor) -> list:
    """
    Parse detection boxes and organize them into lines of braille
    """
    # copy values from troublesome "boxes" object to numpy array
    new_boxes = np.zeros(boxes.shape)
    new_boxes[:, :4] = boxes.xywh.numpy()  # first 4 channels are xywh
    new_boxes[:, 4] = boxes.conf.numpy()  # 5th channel is confidence
    new_boxes[:, 5] = boxes.cls.numpy()  # 6th channel is class which is last channel
    
    # sort according to y coordinate
    new_boxes = new_boxes[new_boxes[:, 1].argsort()]
    
    # find threshold index to break the line
    y_threshold = np.mean(new_boxes[:, 3]) // 2
    boxes_diff = np.diff(new_boxes[:, 1])
    threshold_index = np.where(boxes_diff > y_threshold)[0]
    
    # cluster according to threshold_index
    boxes_clustered = np.split(new_boxes, threshold_index + 1)
    boxes_return = []
    
    for cluster in boxes_clustered:
        # sort according to x coordinate
        cluster = cluster[cluster[:, 0].argsort()]
        boxes_return.append(cluster)
    
    return boxes_return

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
model = None
MODEL_PATH = "snoop2head/yolov8m-braille"

def load_model():
    """Load the YOLOv8 model from Hugging Face"""
    global model
    if model is None:
        model = YOLO(MODEL_PATH)
        model.overrides["conf"] = 0.15  # NMS confidence threshold
        model.overrides["iou"] = 0.15  # NMS IoU threshold
        model.overrides["agnostic_nms"] = False  # NMS class-agnostic
        model.overrides["max_det"] = 1000  # maximum number of detections per image
    return model

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "ok"})

@app.route('/detect', methods=['POST'])
def detect_braille():
    """Endpoint to detect braille in uploaded image"""
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        # Read image from request
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Load model (if not already loaded)
        model = load_model()
        
        # Perform prediction
        with torch.no_grad():
            results = model.predict(image)
            boxes = results[0].boxes  # first image
            
            # No detections
            if len(boxes) == 0:
                return jsonify({"text": "", "message": "No braille patterns detected"}), 200
            
            # Parse boxes into lines
            list_boxes = parse_xywh_and_class(boxes)
            
            # Convert detected classes to braille text
            braille_lines = []
            for box_line in list_boxes:
                str_left_to_right = ""
                box_classes = box_line[:, -1]
                for each_class in box_classes:
                    str_left_to_right += convert_to_braille_unicode(
                        model.names[int(each_class)]
                    )
                braille_lines.append(str_left_to_right)
            
            # Join all lines with newline characters
            full_text = "\n".join(braille_lines)
            
            return jsonify({
                "text": full_text,
                "lines": braille_lines,
                "count": len(braille_lines)
            })
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # For local development
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)