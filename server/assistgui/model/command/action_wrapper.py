import cv2
import re
import numpy as np
import torch
import json
from PIL import Image
from assistgui.model.utils import run_llm
from assistgui.model.command.utils import Time, format_gui
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util import box_ops
from groundingdino.util.inference import predict
from assistgui.model.base_module import BaseModule
from assistgui.model_zoo.shared_model import SharedModel
from assistgui.model.gui_parser.utils import crop_panel


class BaseAction():
    name = "base_action"
    description = ""

    def __init__(self):
        super(BaseAction, self).__init__()
        self.screenshot_path = None
        self.gui = None
        self.llm = "gpt-4-0314"

    def update_screenshot(self, **kwargs):
        screenshot = kwargs.get('screenshot', None)
        if screenshot:
            self.screenshot_path = screenshot if type(screenshot) is str else screenshot.image_path

        self.gui = kwargs.get('gui', None)

    def build_model(self, model_name, **kwargs):
        """Load the dl model.
         Check SharedModel for available models and loaded elements."""
        if type(model_name) is list:
            models = {}
            for name in model_name:
                model = SharedModel(name)
                for key, value in model.model_dict.items():
                    setattr(self, f'{name}_{key}', value)
                models[name] = model
            return models
        elif type(model_name) is str:
            model = SharedModel(model_name)
            for key, value in model.model_dict.items():
                setattr(self, key, value)
            return model

    def get_gui_info(self, query, gui):
        # given a query, and gui metadata, return the answer
        prompt = "Please retrieve the information based on the provided Query and the GUI data I provide.\n\n" + \
                 f"GUI: {gui} \n\n" +  \
                 f"Query: {query} \n\n" + \
                 "Notes:\nReturn the desired ORIGINAL (Must Original) elements in the above GUI data. \n" + \
                 "Output in the following format:  must be one list -> [{'name': str, 'rectangle': list}, ...], " \
                 "where each element is a dictionary which is the retrieval result for one query. \n\n" \
                 f"Output:"
        response = run_llm(prompt, llm=self.llm, max_tokens=200, temperature=0)
        retrieved_info = eval(response)
        return retrieved_info

    def predict_dino(self, image_path, text_prompt, box_threshold=0.3, text_threshold=0.25):
        image_source, image = load_image(image_path)

        boxes, logits, phrases = predict(
            model=self.groundingdino_model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        W, H = Image.open(image_path).size
        boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        return boxes, logits, phrases

    @staticmethod
    def set_parameters(parameter_box, new_value):
        if 'rectangle' in parameter_box:
            x = (parameter_box['rectangle'][0] + parameter_box['rectangle'][2]) / 2
            y = (parameter_box['rectangle'][1] + parameter_box['rectangle'][3]) / 2
        elif 'position' in parameter_box:
            x, y = parameter_box['position']
        else:
            raise ValueError("parameter_box must have rectangle or position key.")

        code = f"click({x}, {y})\n" + \
               f"write('{new_value}')\n" + \
               f"press('enter')\n"
        return code


class UpdateGUI(BaseAction):
    name = "update_gui"
    description = (
        '''
API: update_gui()
You must use this API to update the UI if a previous action alters it. 
For example, if you plan to select an option after interacting with a dropdown menu or button, invoke this API to refresh the interface.
''')

    def __init__(self):
        super(UpdateGUI, self).__init__()

    def __call__(self, *args, **kwargs):
        return "update_gui()"


class Finish(BaseAction):
    name = "finish"
    description = (
        '''
API: finish()
You must call this api when you finish the task.
''')

    def __init__(self):
        super(Finish, self).__init__()

    def __call__(self):
        return "# finish"


class CutVideo(BaseAction):
    name = "razor_tool"
    description = (
        '''
API: razor_tool(timestamp)
:param timestamp -> int, the timestamp in second.
This tool can cut video given specific time stamps.
''')

    def __init__(self):
        super(CutVideo, self).__init__()
        self.llm = None

    def _run(self):
        playhead = find_elements(gui, "Timeline", ["playhead"])
        layer = find_elements(gui, "Timeline", ["|layer"])
        x = playhead[0]["rectangle"][0]
        y = layer[0]["rectangle"][1] + 10
        code = f"click({x}, {y})\n"
        return code

    def find_elements(data, panel_name, element_names):
        elements_found = []
        for window_name, window_data in data.items():
            for panel in window_data:
                if panel["name"] == panel_name:
                    for row in panel["elements"]:
                        for element in row:
                            if element["name"] in element_names or any(
                                    name_part in element["name"] for name_part in element_names):
                                elements_found.append(element)
        return elements_found

    def __call__(self, *args, **kwargs):
        self._run()


class Drag(BaseAction):
    name = "drag"
    description = (
        '''
API: drag(start_x, start_y, end_x, end_y, duration)
This tool can drag something from (start_x, start_y) to (end_x, end_y).
''')

    def __init__(self):
        super(Drag, self).__init__()
        self.llm = None

    def __call__(self, start_x, start_y, end_x, end_y, duration=1):
        code = f'''moveTo({start_x}, {start_y})  
mouseDown()  
moveTo({end_x}, {end_y}, duration={duration})  
mouseUp()
'''
        return code


class AddGuideLine(BaseAction):
    name = "add_guideline"
    description = (
        '''
API: add_guideline(program_bbox, object)
This tool can add guide line in program panel for an object. program_bbox -> list, object -> str, indicating add guide line for which object in media asset, e.g., "man in white suit".
        ''')

    def __init__(self):
        super(AddGuideLine, self).__init__()
        self.build_model(['groundingdino'])

    def __call__(self, program_bbox, object_name):
        boxes, phrases, logits = self.predict_dino(self.screenshot_path, object_name)
        object_bbox = boxes[0].cpu().numpy().tolist()

        code = f'''program_bbox = {program_bbox}
object_bbox = {object_bbox}

ruler_center_x = program_bbox[0] + (program_bbox[2] - program_bbox[0]) / 2
ruler_center_y = program_bbox[1] + (program_bbox[3] - program_bbox[1]) / 2

click(ruler_center_x, ruler_center_y)

moveTo(program_bbox[0] + 10, ruler_center_y)
mouseDown()
moveTo(object_bbox[0], ruler_center_y, duration=0.2)
mouseUp()

moveTo(program_bbox[0] + 10, ruler_center_y)
mouseDown()
moveTo(object_bbox[2], ruler_center_y, duration=0.2)
mouseUp()

moveTo(ruler_center_x, program_bbox[1] + 60)
mouseDown()
moveTo(ruler_center_x, object_bbox[1], duration=0.2)
mouseUp()

moveTo(ruler_center_x, program_bbox[1] + 60)
mouseDown()
moveTo(ruler_center_x, object_bbox[3], duration=0.2)
mouseUp()'''
        return code


class MovePlayHead(BaseAction):
    name = "move_playhead"
    description = (
        '''
API: move_playhead(target_timestamp)
This tool can move playhead to specific position by typing the time in playhead.
:param target_timestamp -> str, can be in either one of the following format: 
1) absolute timestamp: "H:M:S:F", e.g., "00:00:00:03", indicating 0 seconds and 3 frames
2) relative timestamp: "start/middle/end", indicating the start/middle/end of the video
3) relative timestamp: "start/middle/end/current +/- H:M:S:F", indicating the start/middle/end/current of the video plus/minor the corresponding time
''')

    def __init__(self):
        super(MovePlayHead, self).__init__()
        self.llm = 'gpt-4-0613'

    def __call__(self, target_seconds):
        current, video_length = self.get_time_position()

        current_time = Time(current['name'].replace(" ", ""))
        end_time = Time(video_length['name'].replace(" ", ""))

        # use re to find the "00:00:00:03" like string in the target_seconds
        seconds = re.findall(r"\d{2}:\d{2}:\d{2}:\d{2}", target_seconds)
        # replace it with Time('00:00:00:03')
        if seconds:
            target_seconds = target_seconds.replace(seconds[0], f"Time('{seconds[0]}')")

        if "start" in target_seconds:
            target_seconds = target_seconds.replace("start", "Time('00:00:00:00')")
        elif "middle" in target_seconds:
            target_seconds = target_seconds.replace("middle", f"Time('{str(end_time / 2)}')")
        elif "end" in target_seconds:
            target_seconds = target_seconds.replace("end", f"Time('{str(end_time)}') - Time('00:00:00:01')")
        elif "current" in target_seconds:
            target_seconds = target_seconds.replace("current", f"Time('{str(current_time)}')")

        target_seconds = eval(target_seconds)

        code = self.set_parameters(current, target_seconds)

        return code

    def get_time_position(self):
        gui = [panel for panel in self.gui['panel'] if panel['name'] == 'Program'][0]
        required_information = "the box indicating the current timestamp; the box indicating the length of the video."
        response = self.get_gui_info(query=required_information, gui=gui)
        current, video_length = response[0], response[1]
        return current, video_length


class AlignObjectToGuideLine(BaseAction):
    name = "align_object_to_guideline"
    description = (
        '''
API: align_object_to_guideline(object_name)
:param object_name -> 'str', indicating align which object to the reference object.
This tool can directly adjust the position and scale of the frame to align the object to the guideline line. 
''')

    def __init__(self):
        super(AlignObjectToGuideLine, self).__init__()
        self.llm = 'gpt-4-0613'
        self.build_model(['groundingdino'])
        self.canvas_size = (1920, 1080)

    def __call__(self, object_name):
        canvas_x, canvas_y, scale = self.get_video_info()
        media_asset = self.get_media_asset_info()
        canvas_center = [float(canvas_x['name']), float(canvas_y['name'])]

        object_bbox = self.get_object_bbox(object_name, media_asset)
        print("object_bbox", object_bbox)
        guideline_bbox = self.get_guideline_bbox(media_asset)
        print("guideline_bbox", guideline_bbox)

        # calculate the new center and scale
        new_c_x, new_c_y, new_scale = self.transform_object_to_target_box(object_bbox, guideline_bbox, canvas_center, scale)
        print("new_c_x, new_c_y, new_scale", new_c_x, new_c_y, new_scale)

        # generate code for setting the new center and scale
        code = ""
        code += self.set_parameters(canvas_x, new_c_x)
        code += self.set_parameters(canvas_y, new_c_y)
        code += self.set_parameters(scale, new_scale)
        return code

    def get_object_bbox(self, object_name, media_asset):
        # detect object with a visual grounding model DINO
        media_asset_img_path = crop_panel(media_asset['rectangle'], self.screenshot_path, if_save=True)
        boxes, phrases, logits = self.predict_dino(media_asset_img_path, object_name)
        print("boxes:", boxes[0])
        object_bbox = boxes[0].cpu().numpy().tolist()
        object_bbox = self.transform_coordinates(video_size=self.canvas_size,
                                                 point_coord_media_coord=object_bbox, media_asset=media_asset)
        print("object bbox:", object_bbox)
        return object_bbox

    def get_guideline_bbox(self, media_asset):
        # TODO: move to gui parser
        # Convert the image to HSV color space
        image = crop_panel(media_asset['rectangle'], self.screenshot_path)

        # Redefine the detection function to use new thresholds
        def detect_cyan_lines_revised(image, threshold=0.8):
            # Redefine the lower and upper bounds for cyan color in HSV to be more restrictive
            lower_cyan = np.array([85, 150, 150])
            upper_cyan = np.array([95, 255, 255])

            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(image_hsv, lower_cyan, upper_cyan)
            vertical_lines = [col for col in range(mask.shape[1]) if np.sum(mask[:, col]) / 255 / mask.shape[0] >= threshold]
            horizontal_lines = [row for row in range(mask.shape[0]) if np.sum(mask[row, :]) / 255 / mask.shape[1] >= threshold]
            return vertical_lines, horizontal_lines

        # Detect the cyan lines in the uploaded image using revised function
        vertical_lines, horizontal_lines = detect_cyan_lines_revised(image)

        def merge_close_lines(lines, threshold=5):
            if not lines:
                return []

            # Sort the lines
            lines = sorted(lines)

            # Initialize merged lines list and start with the first line
            merged_lines = [lines[0]]

            for i in range(1, len(lines)):
                # If the line is close to the last merged line, replace it with their average
                if abs(lines[i] - merged_lines[-1]) <= threshold:
                    merged_lines[-1] = (merged_lines[-1] + lines[i]) // 2
                else:
                    # Otherwise, add it as a new line
                    merged_lines.append(lines[i])

            return merged_lines

        # Merge the detected vertical and horizontal lines
        vertical_lines = merge_close_lines(vertical_lines)
        horizontal_lines = merge_close_lines(horizontal_lines)

        guideline_box = [vertical_lines[0], horizontal_lines[0], vertical_lines[1], horizontal_lines[1]]
        guideline_box = self.transform_coordinates(video_size=self.canvas_size,
                                                   point_coord_media_coord=guideline_box, media_asset=media_asset)
        return guideline_box

    def transform_coordinates(self, video_size, point_coord_media_coord, media_asset):
        video_width, video_height = video_size

        media_asset_bbox = media_asset['rectangle']
        media_width = media_asset_bbox[2] - media_asset_bbox[0]
        media_height = media_asset_bbox[3] - media_asset_bbox[1]

        video_aspect_ratio = video_width / video_height
        program_aspect_ratio = media_width / media_height

        padding_x = 0
        padding_y = 0

        if video_aspect_ratio < program_aspect_ratio:
            scale = media_height / video_height
            padding_x = (media_width - (video_width * scale)) / 2
        else:
            scale = media_width / video_width
            padding_y = (media_height - (video_height * scale)) / 2

        if len(point_coord_media_coord) == 2:
            point_x, point_y = point_coord_media_coord
            video_x = (point_x - padding_x) / scale
            video_y = (point_y - padding_y) / scale
            return video_x, video_y
        elif len(point_coord_media_coord) == 4:
            x1, y1, x2, y2 = point_coord_media_coord
            video_x1 = (x1 - padding_x) / scale
            video_y1 = (y1 - padding_y) / scale
            video_x2 = (x2 - padding_x) / scale
            video_y2 = (y2 - padding_y) / scale
            return video_x1, video_y1, video_x2, video_y2

    def get_video_info(self):
        query = "1) the box indicates the x coordinate of the video position; 2) the box indicates the y coordinate of the video position; 3) the box indicates the scale of the video."
        gui = [panel for panel in self.gui['panel'] if panel['name'] == 'Effect Controls'][0]
        canvas_x, canvas_y, scale = self.get_gui_info(query, gui)
        return canvas_x, canvas_y, scale

    def get_media_asset_info(self):
        gui = [panel for panel in self.gui['panel'] if panel['name'] == 'Program'][0]
        for row in gui['elements']:
            for ele in row:
                if ele['name'] == 'media asset':
                    return ele

    @staticmethod
    def transform_object_to_target_box(object_bbox, target_bbox, canvas_center, scale):
        x1, y1, x2, y2 = object_bbox
        x1_t, y1_t, x2_t, y2_t = target_bbox
        c_x, c_y = canvas_center
        scale = float(scale['name']) / 100

        width_obj = x2 - x1
        width_target = x2_t - x1_t

        new_scale = scale * (width_target / width_obj)

        y1_after_scale = c_y + (y1 - c_y) * new_scale
        x1_after_scale = c_x + (x1 - c_x) * new_scale

        dy = y1_t - y1_after_scale
        dx = x1_t - x1_after_scale

        new_c_x = c_x + dx
        new_c_y = c_y + dy

        new_scale *= 100

        return new_c_x, new_c_y, new_scale


class MaskObject(BaseAction):
    name = "roto_brush"
    description = (
        '''
API: roto_brush(query)
:param query -> str, the name of the object you want to mask.
This tool can use roto brush tool in After Effects to mask object asked by query.
''')

    def __init__(self):
        super(MaskObject, self).__init__()
        self.llm = 'gpt-4-1106-preview'
        self.input_image = None
        self.build_model(['qwen_vl_chat'])  # 'sam', 'groundingdino',

    def __call__(self, query):
        # code = ''''''
        # """detect media assets in the screenshot"""
        print("start roto brush")
        query = self.process_query(query)
        print("start roto brush")
        boxes = self.predict_qwen(self.screenshot_path, query)
        code = self.code_generation(boxes)
        return code

    def process_query(self, query):
        prompt = f'''将我的输入（英文）中关键的物体提炼出来，然后转化成这样形式的问句（中文）。
以下为一个示例，请遵循输入输出格式：
Input:
Mask the boy with roto brush.
Output:
请给我框出男孩

Note that: Output不需要提及roto brush，只需要提及物体即可。

开始！
Input:
{query}
Output:
'''
        response = run_llm(llm=self.llm, prompt=prompt)
        return response

    def predict_qwen(self, screenshot_path, query):
        model = self.qwen_vl_chat_model
        tokenizer = self.qwen_vl_chat_tokenizer

        torch.manual_seed(1234)
        query = tokenizer.from_list_format([
            {'image': screenshot_path},
            {'text': query},
        ])
        response, history = model.chat(tokenizer, query=query, history=None)
        boxes = self.decode_box(response, history, tokenizer)
        return boxes

    def decode_box(self, response, history, tokenizer):
        boxes = tokenizer._fetch_all_box_with_ref(response)
        image = Image.open(self.screenshot_path)
        w, h = image.size[0], image.size[1]
        for box in boxes:
            x1, y1, x2, y2 = box['box']
            box['box'] = (int(x1 / 1000 * w), int(y1 / 1000 * h), int(x2 / 1000 * w), int(y2 / 1000 * h))
        return boxes

    def code_generation(self, boxes):
        # boxes = tokenizer._fetch_all_box_with_ref(response)
        # The bounding box coordinates
        x1, y1, x2, y2 = boxes[0]['box']

        # Calculate center points for the vertical and horizontal lines
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        th = 10
        # Calculate start and end points for the horizontal line (within the bounding box)
        horizontal_start = (x1 + th, center_y)
        horizontal_end = (x2 - th, center_y)

        # Calculate start and end points for the vertical line (within the bounding box)
        vertical_start = (center_x, y1 + th)
        vertical_end = (center_x, y2 - th)

        code = f'''PAUSE = 0.5

# Move to the start of the horizontal line and draw it
moveTo({horizontal_start[0]}, {horizontal_start[1]})
dragTo({horizontal_end[0]}, {horizontal_end[1]}, button='left', duration=1)

# Wait a bit before drawing the next line
time.sleep(1)

# Move to the start of the vertical line and draw it
moveTo({vertical_start[0]}, {vertical_start[1]})
dragTo({vertical_end[0]}, {vertical_end[1]}, button='left', duration=1)
'''
        return code


def transform_image(image) -> torch.Tensor:
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_transformed, _ = transform(image, None)
    return image_transformed
