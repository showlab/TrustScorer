import os
import re
import datetime
import numpy as np
import cv2


def multivalue_image(img, mode='None', thresholds=None, interval_values=None, save=True, cache_folder=None):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed_img = np.zeros_like(img)

    # preset the thresholds and interval values
    if mode == 'get_panel_name':
        thresholds = [5, 140]
        interval_values = [30, 0, 255]
    elif mode == 'get_button':
        thresholds = [30, 50, 150]
        interval_values = [0, 86, 172, 255]
    else:
        if not thresholds:
            # default to perform binary thresholding
            thresholds = list(range(0, 255, int(255 / 2)))
        if not interval_values:
            interval_values = list(range(0, 255, int(255 / len(thresholds))))

    num_thresholds = len(thresholds)

    for i in range(num_thresholds + 1):
        if i == 0:
            # Pixels below the first threshold
            processed_img[img <= thresholds[i]] = interval_values[i]
        elif i == num_thresholds:
            # Pixels above the last threshold
            processed_img[img > thresholds[i - 1]] = interval_values[i]
        else:
            # Pixels between current and previous thresholds
            processed_img[(img > thresholds[i - 1]) & (img <= thresholds[i])] = interval_values[i]

    if save:
        saved_path = os.path.join(cache_folder, f'multivalued-{mode}.png')
        cv2.imwrite(saved_path, processed_img)
    else:
        saved_path = None
    return processed_img, saved_path


def crop_panel(panel_bbox, screenshot_path, if_save=False, panel_name=None):
    # panel_bbox = [x1, y1, x2, y2]
    img = cv2.imread(screenshot_path)
    if if_save:
        # append panel name to the screenshot path, the file format could be png and jpg
        saved_path = ".".join(screenshot_path.split('.')[:-1]) + f'-{panel_name}.png'
        cv2.imwrite(saved_path, img[panel_bbox[1]:panel_bbox[3], panel_bbox[0]:panel_bbox[2]])
        return saved_path
    else:
        return img[panel_bbox[1]:panel_bbox[3], panel_bbox[0]:panel_bbox[2]]


def restore_coordinate(bbox, panel_bbox):
    for item in bbox:
        if 'rectangle' in item:
            item['rectangle'][0] += panel_bbox[0]
            item['rectangle'][1] += panel_bbox[1]
            item['rectangle'][2] += panel_bbox[0]
            item['rectangle'][3] += panel_bbox[1]
        elif 'bbox' in item:
            item['bbox'][0] += panel_bbox[0]
            item['bbox'][1] += panel_bbox[1]
            item['bbox'][2] += panel_bbox[0]
            item['bbox'][3] += panel_bbox[1]
    return bbox


def get_current_time():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def is_in_bbox(text_bbox, panel_bbox):
    if text_bbox[0] >= panel_bbox[0] and text_bbox[1] >= panel_bbox[1] and text_bbox[2] <= panel_bbox[2] and \
            text_bbox[3] <= panel_bbox[3]:
        return True
    else:
        return False


def find_appropriate_row(editing_control, y_center):
    """Find the appropriate row in editing_control for a given y_center."""
    for index, row in enumerate(editing_control):
        # Calculate the center of the first element in the row
        element_center = (row[0]['rectangle'][1] + row[0]['rectangle'][3]) / 2
        # Consider an acceptable range for merging. For this example, it's half of the element height.
        threshold = (row[0]['rectangle'][3] - row[0]['rectangle'][1]) / 2
        if element_center - threshold <= y_center <= element_center + threshold:
            return index
    return None


def insert_into_row(row, button):
    """Insert a button into the given row based on x coordinate."""
    x_coordinate = button['rectangle'][0]
    for index, element in enumerate(row):
        if x_coordinate < element['rectangle'][0]:
            row.insert(index, button)
            return
    row.append(button)


# Function to collect all bounding boxes and their names into a list
def collect_bounding_boxes(data, parent_name=None, collected_boxes=None):
    if collected_boxes is None:
        collected_boxes = []

    if isinstance(data, dict):
        for key, value in data.items():
            if key == 'rectangle':
                collected_boxes.append({"name": parent_name, "rectangle": value})
            elif isinstance(value, (list, dict)):
                collect_bounding_boxes(value, parent_name, collected_boxes)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, list):
                for sub_item in item:
                    collect_bounding_boxes(sub_item, sub_item.get('name', parent_name), collected_boxes)
            else:
                collect_bounding_boxes(item, item.get('name', parent_name), collected_boxes)

    return collected_boxes


def match_time_format(input_str):
    pattern = re.compile(r'\d{2}:\d{2}')
    match = pattern.search(input_str.replace(" ", ""))

    if match:
        return True
    else:
        return False


def process_image_highlight(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    threshold1 = 5
    threshold2 = 140

    image_trivalued = np.zeros_like(img)

    image_trivalued[img > threshold2] = 255

    image_trivalued[(img > threshold1) & (img <= threshold2)] = 0

    image_trivalued[img < threshold1] = 0
    return image_trivalued


def find_compact_bounding_box(data):
    # Initialize min and max coordinates with the first rectangle's coordinates
    min_x = data[0][0]['rectangle'][0]
    max_x = data[0][0]['rectangle'][2]
    min_y = data[0][0]['rectangle'][1]
    max_y = data[0][0]['rectangle'][3]

    # Iterate through all rectangles to find the extreme coordinates
    for element_group in data:
        for element in element_group:
            rectangle = element['rectangle']
            min_x = min(min_x, rectangle[0])
            max_x = max(max_x, rectangle[2])
            min_y = min(min_y, rectangle[1])
            max_y = max(max_y, rectangle[3])

    # The bounding box will be defined by the extreme coordinates
    bounding_box = [min_x, min_y, max_x, max_y]
    return bounding_box


# Flatten the meta_data
def flatten_structure(node, flattened):
    # Check if 'texts' exists and is not empty
    if 'texts' in node['properties'] and node['properties']['texts'][0]:
        # Here you process the node if it has non-empty 'texts'
        # For example, append the 'texts' list to the flattened list
        flattened.append({'properties': node['properties'], 'children': []})

    # Check for children and recurse
    if 'children' in node:
        for child in node['children']:
            flatten_structure(child, flattened)


def process_image_highlight_gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(max(img.flatten()))
    threshold1 = 5
    threshold2 = 190

    image_trivalued = np.zeros_like(img)

    image_trivalued[img > threshold2] = 0

    image_trivalued[(img > threshold1) & (img <= threshold2)] = 0

    image_trivalued[img < threshold1] = 255
    return image_trivalued


def sort_elements(elements, threshold=10):
    # flatten elements
    elements = collect_bounding_boxes(elements)

    # Sort the elements first by the top-left y-coordinate, then by the x-coordinate
    elements_sorted = sorted(elements, key=lambda x: (x['rectangle'][1] + x['rectangle'][3], x['rectangle'][0]))

    # Group elements into rows based on their y-coordinate with a threshold
    rows_with_threshold = []
    current_row = []
    current_y = elements_sorted[0]['rectangle'][1]

    for element in elements_sorted:
        # If the element is in the same row considering the threshold
        if abs(element['rectangle'][1] - current_y) <= threshold:
            current_row.append(element)
        else:
            # Add the current row to rows and start a new row
            rows_with_threshold.append(current_row)
            current_row = [element]
            current_y = element['rectangle'][1]

    # Add the last row
    rows_with_threshold.append(current_row)

    # Sort each row by the x-coordinate
    for i, row in enumerate(rows_with_threshold):
        rows_with_threshold[i] = sorted(row, key=lambda x: x['rectangle'][0])

    return rows_with_threshold

