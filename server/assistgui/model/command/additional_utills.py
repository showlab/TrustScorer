import base64
import cv2
import numpy as np
from PIL import Image



def highlight_img_case12(image1, image2):
    threshold = 90
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)
    _, binary_diff = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    result = binary_diff
    image = Image.fromarray(result)
    width, height = image.size
    x, y, w, h = 0, 0, width, height - 40 

    cropped_image = image.crop((x, y, x + w, y + h))
    image = np.array(cropped_image)
    height, width = image.shape
    top, bottom, left, right = find_diff_pixel(height, width, image)

    if left - threshold > 0:
        x1 = left - threshold
    else:
        x1 = 0

    if top - threshold > 0:
        y1 = top - threshold
    else:
        y1 = 0

    if right + threshold < width:
        x2 = right + threshold
    else:
        x2 = width

    if bottom + threshold < height - 40:
        y2 = bottom + threshold
    else:
        y2 = height - 40

    cropped_pre, cropped_lat = image1[y1:y2, x1:x2], image2[y1:y2, x1:x2]

    return img2base64(cropped_pre, cropped_lat)



def highlight_img_case3(image1, image2, json1, json2, position=False):
    if (position):
        action_position = position
    else:
        action_position = picture_focus(image1, image2)
    flag1, rec1 = get_action_panel_img(action_position,json1)
    flag2, rec2 = get_action_panel_img(action_position,json2)

    if flag1:
        x1, y1, x2, y2 = rec1
        cropped_pre = image1[y1:y2, x1:x2]
    else:
        cropped_pre = image1

    if flag2:
        x1, y1, x2, y2 = rec2    
        cropped_lat = image2[y1:y2, x1:x2]
    else:
        cropped_lat = image2

    return img2base64(cropped_pre,cropped_lat)


def picture_focus(image1, image2):
    threshold = 90
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)

    _, binary_diff = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    result = binary_diff
    image = Image.fromarray(result)
    width, height = image.size
    x, y, w, h = 0, 0, width, height - 40
    cropped_image = image.crop((x, y, x + w, y + h))
    image = np.array(cropped_image)
    height, width = image.shape
    top, bottom, left, right = find_diff_pixel(height, width, image)
    
    return (left + right)//2,(bottom + top)//2


def get_action_panel_img(action_position, metadata):
    flag = False
    if action_position:
        x, y = action_position

        for item in metadata["panel"]:
            panel_name = item["name"]
            x1, y1, x2, y2 = item["rectangle"]
            if x1 <= int(x) <= x2 and y1 <= int(y) <= y2:
                flag = True
                return flag, item["rectangle"]

        return flag, []
    
    else:
        return flag, []


def img2base64(cropped_pre,cropped_lat):
    
    cv2.imwrite('GPT4V/highlight_cache/cropped_pre.png', cropped_pre)
    cv2.imwrite('GPT4V/highlight_cache/cropped_lat.png', cropped_lat)
    _, input_pre_img = cv2.imencode('.png', cropped_pre)
    _, input_lat_img = cv2.imencode('.png', cropped_lat)
    base64_image_pre = base64.b64encode(input_pre_img.tobytes()).decode('utf-8')
    base64_image_lat = base64.b64encode(input_lat_img.tobytes()).decode('utf-8')
    return base64_image_pre, base64_image_lat


def find_diff_pixel(height, width, image):
    top, bottom, left, right = height, 0, width, 0

    for y in range(height):
        for x in range(width):
            if image[y, x] == 255:
                if y < top:
                    top = y
                break

    for y in range(height - 1, -1, -1):
        for x in range(width):
            if image[y, x] == 255:
                if y > bottom:
                    bottom = y
                break

    for x in range(width):
        for y in range(height):
            if image[y, x] == 255:
                if x < left:
                    left = x
                break

    for x in range(width - 1, -1, -1):
        for y in range(height):
            if image[y, x] == 255:
                if x > right:
                    right = x
                break

    return top, bottom, left, right


