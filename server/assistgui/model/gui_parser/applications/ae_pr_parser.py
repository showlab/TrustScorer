from assistgui.model.gui_parser.ui_text_detection import text_detection
from assistgui.model.gui_parser.utils import *
from assistgui.model.gui_parser.gui_parser_base import GUIParserBase
from ultralytics import YOLO


class AfterEffectParser(GUIParserBase):
    def __init__(self, cache_folder='.cache/'):
        # judge if the cache folder exists
        super(GUIParserBase, self).__init__()
        self.cache_folder = cache_folder
        self.task_id = get_current_time()
        self.action_type = ['moveTo', 'click']
        # YOLOv8 model that can detect 600 classes
        self.yolo_model = YOLO("yolov8n-oiv7.pt")

    def __call__(self, meta_data, screenshot_path, software_name=None):
        # only detect text within it
        self.software_name = software_name
        _, ocr = text_detection(screenshot_path, save_png=False)
        highlight_ocr = self.detect_highlight_with_ocr(screenshot_path)

        self.parsed_gui = {software_name: []}
        for window_name, window_meta_data in meta_data.items():
            if not window_meta_data:
                continue

            window_type = self.check_window_type(window_meta_data)

            if window_type == 'popup':
                self.parsed_gui[software_name] += self.parse_popup_window(window_meta_data, window_name)
            else:
                self.parsed_gui[software_name] += self.parse_main_window(window_meta_data, screenshot_path, ocr, highlight_ocr)
                self.parsed_gui[software_name] += self.parse_menu(window_meta_data)

        return self.parsed_gui

    @staticmethod
    def parse_menu(meta_data):
        # get menu item
        menu_item = []
        for item in meta_data:
            if item['properties']['friendly_class_name'] == 'Menu':
                # only keep one level children
                new_item = {'name': item['properties']['texts'][0],
                            'rectangle': item['properties']['rectangle'],
                            'type': ['moveTo', 'click'],
                            'elements': [[{'name': child['properties']['texts'][0],
                                           'rectangle': child['properties']['rectangle'],
                                            'type': ['moveTo', 'click'],
                                           } for child in item['children'] if child['properties']['texts']]]
                            }
                menu_item.append(new_item)
        return menu_item

    @staticmethod
    def check_window_type(window_meta_data):
        if window_meta_data[0]['properties']['friendly_class_name'] in ['MenuItem', 'Edit', 'Button']:
            return 'popup'
        else:
            return 'main'

    def parse_popup_window(self, meta_data, window_name):
        # Filter out dictionaries where 'texts' field is empty
        filtered_data = self.filter_data(meta_data)
        sorted_data_y = self.sort_data_by_y_coordinate(filtered_data)
        elements = self.organize_elements(sorted_data_y)

        panel_item = {'name': window_name, 'rectangle': find_compact_bounding_box(elements), 'elements': elements, 'type': self.action_type}
        return [panel_item]

    def parse_main_window(self, window_meta_data, screenshot_path, ocr, highlight_ocr):
        main_panel = []

        for raw_item in window_meta_data:
            if raw_item['properties']['friendly_class_name'] in ['Pane', 'Dialog']:
                if raw_item['properties']['friendly_class_name'] == 'Dialog': 
                    panel_name = raw_item['properties']['texts'][0]
                else:
                    panel_name = self.recognize_panel(raw_item, highlight_ocr, screenshot_path)

                panel_item = {'name': panel_name,
                              'rectangle': raw_item['properties']['rectangle'], 'type': self.action_type+['doubleClick', 'rightClick', 'dragTo']}

                print("processing", panel_item['name'])

                # call relevant parser for different panel
                temp = {}
                temp['editing_control'] = self.get_text(panel_item, ocr, screenshot_path)
                temp['timeline'] = self.get_timeline(panel_item, screenshot_path)
                if temp['timeline'] and panel_item['name'] == 'Accessory' and raw_item['properties']['friendly_class_name'] != 'Dialog':
                    panel_item['name'] = 'Timeline'
                temp['button'] = self.get_button(panel_item, screenshot_path)
                temp['search'] = self.get_search_bar(panel_item, raw_item)
                temp['asset_bar'] = self.get_asset_bar(panel_item)
                temp['scroll_bar'] = self.get_scroll(panel_item, temp['button'], ocr, screenshot_path)
                temp['media_asset'] = self.get_media_asset(panel_item, raw_item, screenshot_path)
                # merge all elements at the line
                panel_item['elements'] = self.merge_elements(temp)
                main_panel.append(panel_item)

        return main_panel

    @staticmethod
    def filter_data(meta_data):
        """ Filters out entries with empty 'texts' field. """
        return [entry for entry in meta_data if entry['properties']['texts'][0]]

    @staticmethod
    def sort_data_by_y_coordinate(data):
        """ Sorts data based on the y-coordinate of the 'rectangle' field. """

        def get_y_coordinate(entry):
            # Check if 'rectangle' is directly in the entry or inside 'properties'
            if 'rectangle' in entry:
                return entry['rectangle'][1]
            elif 'properties' in entry and 'rectangle' in entry['properties']:
                return entry['properties']['rectangle'][1]
            else:
                raise ValueError("Invalid entry format: 'rectangle' field not found")

        return sorted(data, key=get_y_coordinate)

    @staticmethod
    def sort_row_by_x_coordinate(row):
        """ Sorts a row of elements based on the x-coordinate. """

        def get_x_coordinate(element):
            # Check if 'rectangle' is directly in the element or inside 'properties'
            if 'rectangle' in element:
                return element['rectangle'][0]
            elif 'properties' in element and 'rectangle' in element['properties']:
                return element['properties']['rectangle'][0]
            else:
                raise ValueError("Invalid element format: 'rectangle' field not found")

        return sorted(row, key=get_x_coordinate)

    def organize_elements(self, sorted_data):
        """ Organizes elements into rows based on their y-coordinate. """
        elements = []
        current_row = []
        prev_y = None

        for entry in sorted_data:
            y = entry['properties']['rectangle'][1]

            if prev_y is not None and abs(y - prev_y) > 10:
                elements.append(self.sort_row_by_x_coordinate(current_row))
                current_row = []

            element_name = self.construct_element_name(entry)
            current_row.append({
                "name": element_name,
                "rectangle": entry['properties']['rectangle']
            })
            prev_y = y

        if current_row:
            elements.append(self.sort_row_by_x_coordinate(current_row))

        return elements

    @staticmethod
    def construct_element_name(entry):
        """ Constructs a name for an element based on its properties. """
        if entry['properties']['friendly_class_name'] in ['MenuItem', 'Button']:
            return entry['properties']['texts'][0]
        else:
            return f"{entry['properties']['texts'][0]}|{entry['properties']['friendly_class_name']}"




