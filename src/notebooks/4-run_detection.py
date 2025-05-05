import os
from concurrent.futures import ProcessPoolExecutor  # Substituído
from functools import partial
from tqdm import tqdm
import pandas as pd

def _process_one(frame_path, score_thresh, output_dir):
    """
    Worker function: loads frame, runs detection+classification,
    saves annotated image, returns path.
    """
    from model_utils import detect_and_classify
    out_img, dets = detect_and_classify(frame_path, score_thresh=score_thresh)
    base = os.path.basename(frame_path)
    out_path = os.path.join(output_dir, f"annot_{base}")
    out_img.save(out_path)
    return frame_path, out_path, dets

def compute_iou(gt_box, pred_box):
    x1 = max(gt_box[0], pred_box[0])
    y1 = max(gt_box[1], pred_box[1])
    x2 = min(gt_box[2], pred_box[2])
    y2 = min(gt_box[3], pred_box[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    union = gt_area + pred_area - intersection
    return intersection / union


def match_prediction(row, all_predictions):
    img_path = row.path
    if img_path not in all_predictions:
        return 'not_detected'

    gt_box = [row.x_min, row.y_min, row.x_max, row.y_max]
    best_iou = 0
    best_color = None

    for det in all_predictions[img_path]:
        iou = compute_iou(gt_box, det['box'])
        if iou > best_iou:
            best_iou = iou
            best_color = det['color']

    return best_color if best_iou > 0.5 else 'not_detected'

import glob, os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor  # instead of ProcessPoolExecutor
from functools import partial
from IPython.display import display, clear_output
from PIL import Image
import time
from tqdm import tqdm
import yaml

with open("../new_dataset/test/test.yaml", "r") as f:
    data = yaml.safe_load(f)

new_data = []

for item in data:
    if item['boxes'] != []:
        new_data.append(item)


# 1) adjust these
frame_pattern = "../new_dataset/test/rgb/test/*.png"
output_dir    = "../new_dataset/annotated_frames"
fps           = 10
score_thresh  = 0.6

# 2) prepare
os.makedirs(output_dir, exist_ok=True)
all_frames = sorted(glob.glob(frame_pattern))
# skip frames ≤ 25000

# append ../new_dataset/test/rgb/ to every item in new_data
for item in new_data:
    item['path'] = os.path.join("../new_dataset/test/rgb/", item['path'])
new_data[:3]


# get all new data items with boxes with 30 pixels in size at least
new_data_new = []
for item in new_data:
    for box in item['boxes']:
        if box['x_max'] - box['x_min'] < 15 or box['y_max'] - box['y_min'] < 15:
            continue
        else:
            new_data_new.append(item)
            break
new_data_new[:3]


import random
# random sample 500 items from new_data_new
random.seed(42)
random.shuffle(new_data_new)
sampled_data = new_data_new[:500]
sampled_data[:3]

frame_paths = [item['path'] for item in sampled_data]

# transform to dataframe for each box in each item of sampled data
import pandas as pd

def transform_to_dataframe(data):
    rows = []
    for item in data:
        for box in item['boxes']:
            rows.append({
                'path': item['path'],
                'x_min': box['x_min'],
                'y_min': box['y_min'],
                'x_max': box['x_max'],
                'y_max': box['y_max'],
                'label': box['label']
            })
    return pd.DataFrame(rows)

df = transform_to_dataframe(sampled_data)
df['predicted'] = None


import os
if __name__ == "__main__":
    num_workers = os.cpu_count() or 4
    all_predictions = {}
    score_thresh = 0.6
    output_dir = os.path.join('..', 'new_dataset', 'annotated_frames')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    worker_fn = partial(_process_one, score_thresh=score_thresh, output_dir=output_dir)

    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        results = list(tqdm(exe.map(worker_fn, frame_paths),
                            total=len(frame_paths),
                            desc="Processing frames"))

    for frame_path, out_path, dets in results:
        all_predictions[frame_path] = dets

    df['predicted'] = [match_prediction(row, all_predictions) for row in df.itertuples()]

    df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    print("Predictions saved to predictions.csv")