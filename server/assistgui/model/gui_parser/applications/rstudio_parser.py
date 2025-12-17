from assistgui.model.gui_parser.ui_text_detection import text_detection
from assistgui.model.gui_parser.utils import *
from assistgui.model.gui_parser.gui_parser_base import GUIParserBase


class RStudioParser(GUIParserBase):
    def __init__(self, cache_folder='.cache/'):
        # judge if the cache folder exists
        super(GUIParserBase, self).__init__()
        self.cache_folder = cache_folder
        self.task_id = get_current_time()

    def __call__(self, meta_data, screenshot_path, software_name=None):
        self.software_name = software_name
        # self.screenshot_paths.append(screenshot_path)
        self.parsed_gui = {software_name: []}
        self.exclude_class_name_list = ['GroupBox', 'ListBox', 'Custom', 'Menu', 'Pane', 'Toolbar', 'TabControl', 'DataItem', 'TabItem', 'MenuItem', 'Button', 'Document']

        self.parsed_gui = self.get_panel_uia(meta_data, screenshot_path)
        # Template matching for these applications
        for panel_item in self.parsed_gui[self.software_name]:
            if panel_item['name'] not in ['Title Bar', 'Navigation Bar']:
                button_box = self.get_button(panel_item, screenshot_path)
                panel_item['elements'] += button_box

        return self.parsed_gui

