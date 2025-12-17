import copy
from typing import Any
import random
import os
from explore_utils import (
    get_current_GUI,
    save_data,
    start_processs,
    stop_processs,
    get_projectname,
    get_softwarename,
)


class Explorer:
    name = "Explorer"
    description = """
This tool can create a lot of tasks based on the bottom-up method.
"""

    def __init__(self, SOFTWARENAME) -> None:
        self.done_button_name = []
        self.count_button_name = {}
        self.SOFTWARENAME = SOFTWARENAME
        (
            self.process_name,
            self.software_name_re,
            self.software_path,
            self.parser_softwarename,
        ) = get_softwarename(self.SOFTWARENAME)

    def __call__(self, project_path, output_folder_path, epoch=1, **kwds: Any) -> Any:
        ###########################
        # Main function: select one project; find X task trace (epoch = X）; the output will be saved in
        # the output_folder_path as a folder whose name is the project's name.
        project_result_path = os.path.join(
            output_folder_path, get_projectname(project_path, self.SOFTWARENAME)
        )
        count = 0
        while count < epoch:
            count += 1
            start_processs(self.software_path, project_path)
            code = self.start_explorer_singleTask(project_result_path)       #
            stop_processs(self.process_name)
            print("Single task result path: ", project_result_path)

        print("Exploring a project's result path: ", project_result_path)
        # return project_result_path
        return code

    def start_explorer_singleTask(self, folder_path):
        ## find single Task trace
        # input: folder_path: the path that the result will be stored in

        # information will be store:
        # X screenshot-X.png: X means the X-nd image
        # X metadata-X.json:  X means the X-nd json data

        # operation.json: A dict List: contains all the code and subtask description
        # eg.   "code": f"# Click on {button_name}\nclick({position_x}, {position_y})"
        #       "task": f"Subtask 1: Click the {button_name} button"

        gui_list, screenshot_path_list, operation_list = [], [], []
        code =''
        previous_gui = {}
        # flag = 0

        # start state:

        current_gui, screenshot_path = get_current_GUI(
            self.software_name_re, self.parser_softwarename
        )

        gui_list.append(current_gui)
        screenshot_path_list.append(screenshot_path)
        case_number, button = self.get_random_button(previous_gui, current_gui)
        while 1:
            # if (case_number == 3 and flag) or case_number == 2:
            if case_number == 3 or case_number == 2:
                # Case 2: Close several panels (end situation)
                break
            elif case_number == 1 or case_number == 4:
                # Case 1: Popup several panels
                # Case 4: other cases
                case_number, button = self.do_operation(
                    button,
                    gui_list,
                    screenshot_path_list,
                    operation_list,
                    self.software_name_re,
                    self.parser_softwarename,
                )

                # flag = 1
            else:
                # Case 3: Manipulate one panel
                case_number, button = self.do_operation(
                    button,
                    gui_list,
                    screenshot_path_list,
                    operation_list,
                    self.software_name,
                )
                # flag = 1
        save_data(folder_path, gui_list, screenshot_path_list, operation_list)
        #
        for operation in operation_list:
            act = operation['code']
            code = code + f'{act}'+'\n'
        return code
     
    # @staticmethod
    # def set_app_focus(target_window_name):
    #     ## set focus before operating the application
    #     app = Application(backend="uia").connect(title_re=target_window_name)
    #     all_windows = app.window(found_index=0)
    #     all_windows.set_focus()

    def do_operation(
        self,
        button,
        gui_list: list,
        screenshot_path_list: list,
        operation_list: list,
        softwarename,
        parser_softwarename,
    ):
        code=''
        action_list = button["type"]
        action = random.choice(action_list)
        button_name = button["name"]
        x1, y1, x2, y2 = button["rectangle"]
        position_x, position_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
        print("Doing action: ", action)

        if action == "click" or "moveto":
            # click case
            action = "click"
            # self.set_app_focus(softwarename)
            # pyautogui.click(position_x, position_y)
            code += f'pyautogui.click({position_x}, {position_y})'

        elif action == "rightClick":
            # self.set_app_focus(softwarename)
            # pyautogui.rightClick(position_x, position_y)
            code += f'pyautogui.rightClick({position_x}, {position_y})'
            
        elif action == "doubleClick":
            # self.set_app_focus(softwarename)
            # pyautogui.doubleClick(position_x, position_y)
            code += f'pyautogui.doubleClick({position_x}, {position_y})'
            
        else:
            assert "error: wrong button accessible type, check the gui_data"

        # GUI information
        previous_gui = gui_list[-1]
        current_gui, cur_screenshot_path = get_current_GUI(
            softwarename, parser_softwarename
        )

        # get the case_number and To-do button after operation
        case_number, button = self.get_random_button(previous_gui, current_gui)
        print("button: ", button)

        # record the operation
        operation = {
            "button_name": button_name,
            "case_number": case_number,
            # "code": f"# {action} on {button_name}\{action}({position_x}, {position_y})",
            "code": f"{action}({position_x}, {position_y})",
            "task": f"Subtask: {action} the {button_name} button",
        }

        self.done_button_name.append(button_name)
        gui_list.append(current_gui)
        screenshot_path_list.append(cur_screenshot_path)
        operation_list.append(operation)

        return case_number, button

    def get_random_button(self, previous_gui, current_gui):
        case = 1
        button_list = []
        # Start case: previous_gui = NULL
        if not previous_gui:
            for _, panel_list in current_gui.items():
                for _, sequence in enumerate(panel_list):
                    for item in sequence["elements"]:
                        if item:
                            button_list.extend(item)
            button = random.choice(button_list)
            case = 1
            return case, button
        panel_diff = self.find_panel_difference(previous_gui, current_gui)
        panel_diff_copy = copy.deepcopy(panel_diff)
        # Case 1: Popup several panels
        if len(panel_diff_copy["added"]) > 0 and len(panel_diff_copy["removed"]) == 0:
            print("# Case 1: Popup several panels")
            for diff_panel in panel_diff_copy["added"]:
                for _, value in diff_panel.items():
                    for item in value["elements"][0]:
                        if item:
                            button_list.append(item)
            case = 1

        # Case 2: Close several panels (end situation)
        elif len(panel_diff_copy["removed"]) > 0 and len(panel_diff_copy["added"]) == 0:
            print("# Case 2: Close several panels (end situation)")
            case, button = 2, {}

        # Case 3: Manipulate one panel
        elif (
            len(panel_diff_copy["added"]) == 0 and len(panel_diff_copy["removed"]) == 0
        ):
            print(" # Case 3: Manipulate one panel")
            for _, panel_list in current_gui.items():
                for _, sequence in enumerate(panel_list):
                    for item in sequence["elements"]:
                        if item:
                            button_list.extend(item)
            case = 3

        # Case 4: Other cases
        else:
            print("# Case 4: Other cases: popup and close some panels")
            if panel_diff_copy["added"]:
                for diff_panel in panel_diff_copy["added"]:
                    for _, value in diff_panel.items():
                        for item in value["elements"][0]:
                            if item:
                                button_list.append(item)

            else:
                for _, panel_list in current_gui.items():
                    for _, sequence in enumerate(panel_list):
                        for item in sequence["elements"]:
                            if item:
                                button_list.extend(item)

            case = 4
        button = {}
        while not button or not button["name"] or not button["type"]:
            button = random.choice(button_list)
        if button["name"] not in self.done_button_name:
            self.done_button_name.append(button["name"])
            return case, button
        else:
            return self.get_random_button(previous_gui, current_gui)

    def find_panel_difference(self, previous_gui, current_gui):
        results = {"added": [], "removed": []}

        # Create dictionaries for panel names to their contents
        panels1 = {}
        for _, panel_list in previous_gui.items():
            for _, sequence1 in enumerate(panel_list):
                panels1[sequence1["name"]] = sequence1

        panels2 = {}
        for _, panel_list in current_gui.items():
            for _, sequence2 in enumerate(panel_list):
                panels2[sequence2["name"]] = sequence2

        # Compare panel names for additions and removals
        added_panels = set(panels2) - set(panels1)
        removed_panels = set(panels1) - set(panels2)

        # Add new or removed panels to the results with their content
        for panel_name in added_panels:
            results["added"].append({panel_name: panels2[panel_name]})
        for panel_name in removed_panels:
            results["removed"].append({panel_name: panels1[panel_name]})

        # Check for 'children' additions or removals in panels present in both GUI data structures
        for panel_name in set(panels1) & set(panels2):
            children1 = panels1[panel_name].get("children", [])
            children2 = panels2[panel_name].get("children", [])

            # Comparing children by length here, you might need a more complex comparison
            if children1 != children2:
                if len(children1) < len(children2):
                    # A 'children' was added, include the panel content from gui_data2
                    results["added"].append({panel_name: panels2[panel_name]})
                else:
                    # A 'children' was removed, include the panel content from gui_data1
                    results["removed"].append({panel_name: panels1[panel_name]})

        return results


if __name__ == "__main__":
    explorer = Explorer("premiere")
    explorer(
        project_path="xxx.prproj",
        output_folder_path="xxx",
        epoch=3,
    )
