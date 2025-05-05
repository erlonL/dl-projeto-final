# In[1]: install & imports
# !pip install torch torchvision tensorflow pillow numpy

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms as T
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf

# In[2]: device & detector setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load COCO-pretrained Faster R-CNN
detector = fasterrcnn_resnet50_fpn(pretrained=True)
detector.to(device).eval()

TL_CLASS_ID = 10   # COCO class index for “traffic light”

# In[3]: load your Keras color‐classifier
# assume you trained & saved a tf.keras model accepting (H,W,3) float32 in [0,1]
color_classifier = tf.keras.models.load_model("../new_dataset/model-03-05.h5")
COLOR_MAP = {0: "green", 1: "red", 2: "yellow", 3: "off"}

# In[4]: transforms
detector_tf = T.ToTensor()   # just PIL→Tensor for the detector
CLS_SIZE = (64, 64)          # whatever size you trained your CNN on


# In[5]: detection function
def detect_traffic_lights(img: Image.Image, score_thresh=0.5):
    """
    img: PIL.Image
    returns: numpy array of shape (N,4) in (x1,y1,x2,y2)
    """
    x = detector_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = detector(x)[0]
    boxes = out["boxes"].cpu().numpy()
    scores = out["scores"].cpu().numpy()
    labels = out["labels"].cpu().numpy()
    keep = (scores >= score_thresh) & (labels == TL_CLASS_ID)
    return boxes[keep]


# In[6]: classification function
def classify_color(img: Image.Image, box: np.ndarray):
    """
    img: PIL.Image
    box: array [x1,y1,x2,y2]
    returns: one of "red","yellow","green"
    """
    x1, y1, x2, y2 = box.astype(int)
    patch = img.crop((x1, y1, x2, y2))
    patch = patch.resize(CLS_SIZE)
    arr = np.asarray(patch).astype("float32") / 255.0
    arr = np.expand_dims(arr, 0)   # batch dim
    probs = color_classifier.predict(arr)
    idx = int(np.argmax(probs, axis=-1)[0])
    return COLOR_MAP[idx]


# In[7]: full pipeline
def detect_and_classify(image_path: str, score_thresh=0.6):
    img = Image.open(image_path).convert("RGB")
    boxes = detect_traffic_lights(img, score_thresh)
    draw = ImageDraw.Draw(img)
    results = []
    for box in boxes:
        color = classify_color(img, box)
        results.append({"box": box.tolist(), "color": color})
        # draw on image
        if color != "off":
            draw.rectangle(box.tolist(), outline=color if color != "green" else "lime", width=2)
        else:
            draw.rectangle(box.tolist(), outline="black", width=2)
        draw.text((box[0], box[1] - 10), color, fill="white", stroke_width=1, stroke_fill="black")
    return img, results