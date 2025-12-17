import cv2
import numpy as np
import datetime
import importlib
import textdistance
from PIL import Image
from assistgui.model.base_module import BaseModule
from assistgui.model.gui_parser.applications.default_parser import DefaultParser
from assistgui.model.gui_parser.ui_text_detection import text_detection
from assistgui.model.gui_parser.button_detection import detect_button
from assistgui.model.gui_parser.utils import *


class GUIParser(BaseModule):
    name = "gui_parser"
    description = (
        '''
This tool can extract the information of screenshot.
Invoke command: gui_parser(query, visual[i])
:param query -> str, specific command. visual[i] -> image, the latest screenshot.
''')

    def __init__(self, cache_folder='.cache/'):
        # judge if the cache folder exists
        super(GUIParser, self).__init__()
        self.cache_folder = cache_folder
        self.task_id = get_current_time()
        self.parsers = {}
        self.load_parsers_from_config("assistgui/model/gui_parser/applications.config")

    def load_parsers_from_config(self, config_file):
        print("load parsers from config")
        prefix = "assistgui.model.gui_parser.applications."
        with open(config_file, 'r', encoding='utf-8') as file:
            for line in file:
                software_name, parser_class = line.strip().split(',')
                module_name, class_name = parser_class.rsplit('.', 1)
                module = importlib.import_module(prefix + module_name)
                parser = getattr(module, class_name)()
                self.register_parser(software_name, parser)

    def register_parser(self, software_name, parser):
        self.parsers[software_name] = parser

    def get_parser(self, software_name):
        return self.parsers.get(software_name, DefaultParser())

    def _run(self, software_name, meta_data, screenshot_path):
        parser = self.get_parser(software_name)
        print(parser.name)
        return parser(meta_data, screenshot_path, software_name)

