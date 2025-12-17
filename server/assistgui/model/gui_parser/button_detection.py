import cv2
import numpy as np
import time
import os
import re
import glob
import json
from assistgui.model.gui_parser.utils import multivalue_image


def non_max_suppression(boxes, overlap_thresh, scores):
    boxes = np.array(boxes)

    if len(boxes) == 0:
        return [], []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    return boxes[pick].astype("int"), pick


def load_icon_templates(asset_folder, software_name="premiere", panel_name=None, icon_type="icons"):
    if panel_name:
        template_folder = f'{asset_folder}/{software_name}/{panel_name}/{icon_type}'
    else:
        template_folder = f'{asset_folder}/{software_name}'

    icon_path = glob.glob(f'{template_folder}/**/*.png', recursive=True)

    icons = []
    for template_path in icon_path:
        template = cv2.imread(template_path, cv2.IMREAD_COLOR)
        name = re.search(r'[^\\/]+(?=\.\w+$)', template_path).group(0)
        name = re.sub(r'^\d+_', '', name) + "_icon"

        icons.append({'name': name, 'template': template, 'path': template_path})
    return icons


def multi_scale_template_matching(image, template, threshold=0.9, scales=[i / 10.0 for i in range(5, 2, 21)]):
    all_matches = []
    all_score = []
    all_scale = 1
    for scale in scales:
        resized_template = cv2.resize(template, (
        int(template.shape[1] * scale * all_scale), int(template.shape[0] * scale * all_scale)))
        image = cv2.resize(image, (int(image.shape[1] * all_scale), int(image.shape[0] * all_scale)))

        if resized_template.shape[0] > image.shape[0] or resized_template.shape[1] > image.shape[1]:
            continue

        result = cv2.matchTemplate(image, resized_template, cv2.TM_CCOEFF_NORMED)

        locs = np.where(result >= threshold)
        for pt in zip(*locs[::-1]):  # Switch cols and rows
            all_matches.append((pt, scale))
            score_at_pt = result[pt[1], pt[0]]
            all_score.append(score_at_pt)

    return all_matches, all_score


def get_best_matching_scale(image, template, threshold=0.8, scales=None):
    if scales is None:
        scales = [i / 10.0 for i in range(5, 2, 21)]

    all_matches = []
    max_score = -1
    best_scale = 1
    best_location = None
    for scale in scales:
        resized_template = cv2.resize(template, (int(template.shape[1] * scale), int(template.shape[0] * scale)))

        if resized_template.shape[0] > image.shape[0] or resized_template.shape[1] > image.shape[1]:
            continue

        result = cv2.matchTemplate(image, resized_template, cv2.TM_CCOEFF_NORMED)

        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > max_score:
            max_score = max_val
            best_scale = scale
            best_location = max_loc

    return best_scale


def preprocess_image(img, software_name):
    if software_name in ["premiere", "after effect"]:
        threshold = 60
    elif software_name in ["word", "excel", "powerpoint"]:
        threshold = 190
    else:
        threshold = 130

    binary, saved_path = multivalue_image(
        img,
        mode='None',
        thresholds=[threshold],
        interval_values=[0, 255],
        save=False,
        cache_folder='your_cache_folder'
    )
    return binary


def detect_button(image, software_name="premiere", panel_name=None, asset_folder="./assistgui/asset", icon_type=None, threshold=0.78):
    binary_image = preprocess_image(image, software_name)
    templates = load_icon_templates(asset_folder, software_name, panel_name)

    all_boxes, all_scores, labels = [], [], []
    for i, template in enumerate(templates):
        icon_name = template['name']
        icon_template = template['template']
        icon_template_binary = preprocess_image(icon_template, software_name)

        # find the best scale for the template at the first iteration
        if i == 0:
            best_scale = get_best_matching_scale(binary_image, icon_template_binary)

        matches, scores = multi_scale_template_matching(binary_image, icon_template_binary, threshold=threshold,
                                                        scales=[best_scale])

        icon_width = icon_template_binary.shape[1]
        icon_height = icon_template_binary.shape[0]
        for match, score in zip(matches, scores):
            (pt_x, pt_y), scale = match

            end_x = int(pt_x + icon_width * scale)
            end_y = int(pt_y + icon_height * scale)

            all_boxes.append([pt_x, pt_y, end_x, end_y])
            all_scores.append(score)
            labels.append(icon_name)

    nms_boxes, pick = non_max_suppression(all_boxes, 0.5, all_scores)
    labels = [labels[i] for i in pick]

    button_items = []
    for ix, box in enumerate(nms_boxes):
        if 'scroll bar' in labels[ix] or 'effects submenu' in labels[ix]:
            item = {"name": labels[ix], "rectangle": list(box), 'type': ['moveTo', 'click', 'dragTo']}
        else:
            item = {"name": labels[ix], "rectangle": list(box), 'type': ['moveTo', 'click']}
        button_items.append(item)

    return button_items


def process_image_4_new(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold1 = 50
    threshold2 = 100
    threshold3 = 150

    image_trivalued = np.zeros_like(img)

    image_trivalued[img > threshold3] = 255

    image_trivalued[(img > threshold2) & (img <= threshold3)] = 0

    image_trivalued[(img > threshold1) & (img <= threshold2)] = 86

    image_trivalued[(img < threshold1)] = 172
    return image_trivalued


def process_image_3(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold1 = 40
    threshold2 = 150

    image_trivalued = np.zeros_like(img)

    image_trivalued[img > threshold2] = 255

    image_trivalued[(img > threshold1) & (img <= threshold2)] = 0

    image_trivalued[img < threshold1] = 128
    return image_trivalued


def process_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold1 = 30
    threshold2 = 50
    threshold3 = 150

    image_trivalued = np.zeros_like(img)

    image_trivalued[img > threshold3] = 255

    image_trivalued[(img > threshold2) & (img <= threshold3)] = 172

    image_trivalued[(img > threshold1) & (img <= threshold2)] = 86
    return image_trivalued

