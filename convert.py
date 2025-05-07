import json
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Load YOLOv5 model once
model = torch.load('best.pt', map_location='cpu')
model.eval()


def convert_to_braille_unicode(str_input: str, path: str = "./braille_map.json") -> str:
    with open(path, "r") as fl:
        data = json.load(fl)

    return data.get(str_input, "")  # Return empty string if class not found


def parse_xywh_and_class(boxes: torch.Tensor) -> list:
    """
    Parses detection boxes to group them by lines and sort within lines.

    Args:
        boxes (torch.Tensor): Tensor with shape (num_boxes, 6) [xywh, conf, cls]

    Returns:
        list: List of clusters (lines), each containing sorted detection boxes
    """

    new_boxes = np.zeros(boxes.shape)
    new_boxes[:, :4] = boxes[:, :4].cpu().numpy()      # xywh
    new_boxes[:, 4] = boxes[:, 4].cpu().numpy()        # confidence
    new_boxes[:, 5] = boxes[:, 5].cpu().numpy()        # class

    # Sort by y-coordinate (vertical)
    new_boxes = new_boxes[new_boxes[:, 1].argsort()]

    # Detect y-difference jumps to split lines
    y_threshold = np.mean(new_boxes[:, 3]) // 2  # half of mean height
    boxes_diff = np.diff(new_boxes[:, 1])
    threshold_index = np.where(boxes_diff > y_threshold)[0]
    boxes_clustered = np.split(new_boxes, threshold_index + 1)

    boxes_return = []
    for cluster in boxes_clustered:
        cluster = cluster[cluster[:, 0].argsort()]  # sort by x-coordinate
        boxes_return.append(cluster)

    return boxes_return


def image_to_braille(pil_image: Image.Image) -> str:
    """
    Converts a PIL image into Braille text using the YOLOv5 model.

    Args:
        pil_image (PIL.Image): Input image

    Returns:
        str: Braille text
    """
    img_tensor = transforms.ToTensor()(pil_image)
    results = model(img_tensor)

    boxes = results.pred[0]
    if boxes is None or boxes.shape[0] == 0:
        return "No Braille detected."

    lines = parse_xywh_and_class(boxes)

    braille_text = ""
    for line in lines:
        for box in line:
            cls_index = str(int(box[5]))
            braille_text += convert_to_braille_unicode(cls_index)
        braille_text += "\n"

    return braille_text.strip()
