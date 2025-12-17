import os
import copy
import json
import copy
from typing import Any

import random
import numpy as np
from assistgui.model.command.utils import format_gui, compress_gui
from assistgui.model.utils import run_llm
from assistgui.model.base_module import BaseModule
from assistgui.model.command.action_wrapper import (
    Drag,
    UpdateGUI,
    Finish,
    AddGuideLine,
    MovePlayHead,
    AlignObjectToGuideLine,
    MaskObject,
)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)


# load and write JSON file of metadata
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data_json, file_path):
    with open(file_path, "w", encoding="utf-8") as f_out:
        json.dump(data_json, f_out, indent=4, ensure_ascii=False, cls=NumpyEncoder)
    print(f"Data has been saved to {file_path}")


class ActionExplorer(BaseModule):
    """Tool that adds the capability to locate image region with a natural language query."""

    name = "action_explorer"
    description = """
This tool can translate the user's language commands into Python code that can operate the keyboard and mouse.
Invoke command: action_explorer(query, visual[i])
:param query -> str, specific command. visual[i] -> image, the latest screenshot.
"""

    def __init__(self):
        super(ActionExplorer, self).__init__()
        self.llm = "gpt-4-0613"
        # TODO: add additional action wrapper API
        # TODO: add action wrapper conditioned on the software name
        self.available_api = {
            api.name: api for api in [Drag()]
        }  # UpdateGUI(),, AddGuideLine(), MovePlayHead(), AlignObjectToGuideLine(), MaskObject()]}
        self.available_api_illustration = "\n".join(
            [f"{api.description}" for _, api in self.available_api.items()]
        )
        self.done_button_name = []
        self.count_button_name = {}

    def __call__(
        self,
        gui,
        history=None,
        action_countdown=None,
        current_task=None,
        input_image=None,
        error_message=None,
        next_step=None,
        pre_act_success_flag=None,
        pre_act_resume_flag=None,
        software_name=None,
        **kwargs,
    ):
        """input"""
        print("Execute the step")
        # feed the screenshot to the all available api
        # self.update_gui(screenshot=input_image[0], gui=gui)
        cumulative_step = kwargs.get("step", None)
        print("=========print step number ========")
        print(cumulative_step)
        current_gui = gui

        print("============start print current_gui==============")
        print(current_gui)
        if history:
            previous_gui = history[-1]["gui"]  # add previous_gui initialization
        else:
            previous_gui = None
        # generation the code
        print("======start print previous_gui=======")

        code, current_task, case_number = self.query_to_interaction(  # 2024.2.8 wqc
            previous_gui, current_gui, history, cumulative_step
        )
        # if the trace should stop
        message = self.stop_justification(case_number, action_countdown)
        print("============message ============")
        print(f"case_number:{case_number}, message:{message}")
        return code, current_task, history, message

    def query_to_interaction(self, previous_gui, current_gui, history, cumulative_step):
        return self.do_operation(
            history, previous_gui, current_gui, cumulative_step
        )  # 2024.2.8 wqc add previous_gui

    def stop_justification(self, case_number, action_countdown):
        if (
            case_number == 1 or case_number == 4 or case_number == 3
        ):  # 2024. 2. 12 wqc debug
            return "Continue"
        elif case_number == 2 or action_countdown > 300:
            return "Stop"
        else:
            return "Continue"

    def do_operation(self, history, previous_gui, current_gui, cumulative_step):
        # get the case_number and To-do button after operation
        case_number, button = self.get_random_button(previous_gui, current_gui)
        # end state： 当出现case number为2时，直接返回
        command= None       # debug initialize
        current_task=None
        if cumulative_step > 20:
            return None, None, 2  # add cumulative step justification
        if case_number == 2:
            return None, None, case_number

        print("==============start processing button type===============")
        # print(button)  # visualize button
        ## TODO: 运用editable参数判断,区分case To be done
        action_list = button.get("type")
        print(action_list)
        action = random.choice(action_list)
        button_name = button.get("name")
        x1, y1, x2, y2 = button.get("rectangle")
        position_x, position_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
        print("Doing action: ", action)

        if action == "click" or action == "moveTo" or action == "typewrite":  #  debug wqc 2024. 2. 17
            # click case
            action = "click"
            # TODO:
            command = f"click({position_x}, {position_y})"

        elif action == "rightClick":
            command = f"rightClick({position_x}, {position_y})"

        elif action == "doubleClick":
            command = f"doubleClick({position_x}, {position_y})"

        else:
            assert "error: wrong button accessible type, check the gui_data"
        print("button: ", button)
        current_task = f"Subtask: {action} the {button_name} button"
        return command, current_task, case_number

    def get_random_button(self, previous_gui, current_gui):  #
        case = 1
        button_list = []
        # Start case: previous_gui = NULL
        if not previous_gui:
            for _, panel_list in current_gui.items():
                # print(panel_list)
                for _, sequence in enumerate(panel_list):
                    # print(sequence)
                    for item in sequence["elements"]:
                        if isinstance(item, list):
                            button_list.extend(item)
                        elif isinstance(item, dict):
                            button_list.append(item)
                        else:
                            continue
            button = random.choice(button_list)
            # while not button or not button["name"] or not button["type"]:
            while not button or not button.get("name") or "type" not in button:
                button = random.choice(button_list)
            case = 1
            # print('======= return button =========')
            # print(button)
            return case, button
        
        panel_diff = self.find_panel_difference(previous_gui, current_gui)
        panel_diff_copy = copy.deepcopy(panel_diff) #
        print("====== print difference =====")
        print(panel_diff_copy)
        # Case 1: Popup several panels
        if len(panel_diff_copy["added"]) > 0 and len(panel_diff_copy["removed"]) == 0:
            print("# Case 1: Popup several panels")
            for diff_panel in panel_diff_copy["added"]:
                for _, value in diff_panel.items():
                    for item in value["elements"]:
                        if isinstance(item, list):
                            for sub_item in item:
                                if 'type' in value:
                                    sub_item['type'] = value['type']
                                button_list.append(sub_item)
                        elif isinstance(item, dict):
                            if 'type' in value:
                                item['type'] = value['type']
                            button_list.append(item)
            case = 1

        # Case 2: Close several panels (end situation)
        elif len(panel_diff_copy["removed"]) > 0 and len(panel_diff_copy["added"]) == 0:
            print("# Case 2: Close several panels (end situation)")
            case = 2
            return case, {}

        # Case 3: Manipulate one panel
        elif (
            len(panel_diff_copy["added"]) == 0 and len(panel_diff_copy["removed"]) == 0
        ):
            print(" # Case 3: Manipulate one panel")
            for _, panel_list in current_gui.items():
                # print(panel_list)
                for _, sequence in enumerate(panel_list):
                    # print(sequence)
                    for item in sequence["elements"]:
                        if isinstance(item, list):
                            button_list.extend(item)
                        elif isinstance(item, dict):
                            button_list.append(item)
                        else:
                            continue
            case = 3

        # Case 4: Other cases
        else:
            print("# Case 4: Other cases: popup and close some panels")
            if panel_diff_copy["added"]:
                for diff_panel in panel_diff_copy["added"]:
                    for _, value in diff_panel.items():
                        for item in value["elements"]:
                            if isinstance(item, list):
                                button_list.extend(item)
                            elif isinstance(item, dict):
                                button_list.append(item)
                            else:
                                continue

            else:
                for _, panel_list in current_gui.items():
                    for _, sequence in enumerate(panel_list):
                        for item in sequence["elements"]:
                            if isinstance(item, list):
                                button_list.extend(item)
                            elif isinstance(item, dict):
                                button_list.append(item)
                            else:
                                continue

            case = 4
        print("=====current button list==========")
        print(button_list)
        button = random.choice(button_list)
        while not button or not button.get("name") or "type" not in button:
            button = random.choice(button_list)
        print("find button with type")
        if button["name"] == "Open Project ...":
            case = 2

        if button["name"] not in self.done_button_name:
            self.done_button_name.append(button["name"])
            return case, button
        else:
            return self.get_random_button(previous_gui, current_gui)

    def find_panel_difference(self, previous_gui, current_gui):
        results = {"added": [], "removed": []}

        # Create dictionaries for panel names to their contents
        panels1 = {}
        for previous_gui_ in previous_gui:  # 2024.2.11 wqc modify previous gui
            for _, panel_list in previous_gui_.items():
                for _, sequence1 in enumerate(panel_list):
                    panels1[sequence1["name"]] = sequence1

        panels2 = {}
        for _, panel_list in current_gui.items():
            for _, sequence2 in enumerate(panel_list):
                panels2[sequence2["name"]] = sequence2
        added_panels = set(panels2) - set(panels1)
        removed_panels = set(panels1) - set(panels2)

        for panel_name in added_panels:
            if panel_name:
                results["added"].append({panel_name: panels2[panel_name]})
        for panel_name in removed_panels:
            if panel_name:
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
