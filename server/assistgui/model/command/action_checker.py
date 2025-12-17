import re
import os
import json
import copy
from deepdiff import DeepDiff
from assistgui.model.utils import run_llm
from assistgui.model.command.utils import format_gui, compress_gui


class ActionChecker():
    name = "action_checker"
    description = (
        '''
This tool can check if the action is correctly executed.
Invoke command: action_checker(query, visual[i])
:param query -> str, specific command. visual[i] -> image, the latest screenshot.
''')

    def __init__(self):
        super(ActionChecker, self).__init__()
        self.llm = 'gpt-4-0613'

    def __call__(self, current_gui, history):
        # return True, None
        # query, previous_gui, current_gui, code, previous_code = ""
        task = history[-1]['task']
        previous_gui = history[-1]['gui'][-1]
        code = history[-1]['code'][-1]
        previous_code = "\n".join(history[-1]['code'][:-2]) if len(history[-1]['code']) > 1 else ""

        print("checking action ...")
        difference = self.find_difference(previous_gui, current_gui, code)
        success_flag, resume_flag, message, next_step = self.action_evaluation(task, previous_code, code, previous_gui, difference)
        return success_flag, resume_flag, message, next_step

    def find_difference(self, previous_gui, current_gui, code=""):
        panel_diff = self.find_panel_difference(previous_gui, current_gui)
        panel_diff_copy = copy.deepcopy(panel_diff)

        # Case 1: Popup several panels
        if len(panel_diff_copy['added']) > 0 and len(panel_diff_copy['removed']) == 0:
            diff = f"{len(panel_diff_copy['added'])} panel appears:\n"
            for diff_panel in panel_diff_copy['added']:
                diff += "\n".join(format_gui(compress_gui({'panel': [v for k, v in diff_panel.items()]})))
            return diff

        # Case 2: Close several panels
        elif len(panel_diff_copy['removed']) > 0 and len(panel_diff_copy['added']) == 0:
            diff = f"{len(panel_diff_copy['removed'])} panel disappears:\n"
            for diff_panel in panel_diff_copy['removed']:
                diff += "\n".join(format_gui(compress_gui({'panel': [v for k, v in diff_panel.items()]})))
            return diff

        # Case 3: Manipulate one panel
        elif len(panel_diff_copy['added']) == 0 and len(panel_diff_copy['removed']) == 0:
            pre_gui_manipulated_panel, pre_panel_name = self.get_action_panel(previous_gui, code)
            cur_gui_manipulated_panel, cur_panel_name = self.get_action_panel(current_gui, code)
            # use deepdiff to compare json
            if pre_panel_name == cur_panel_name:
                # Manipulate one panel
                diff = self.compare_gui(pre_gui_manipulated_panel, cur_gui_manipulated_panel)
            else:
                # This should not happen, but if it happens, we just compare the whole gui
                diff = self.compare_gui(previous_gui, current_gui)
                diff = diff.pretty()[:500]

        # Other cases
        else:
            diff = self.compare_gui(previous_gui, current_gui)

        print(f"Difference token number: {len(diff)}")
        return diff

    @staticmethod
    def find_panel_difference(previous_gui, current_gui):
        results = {'added': [], 'removed': []}

        # Create dictionaries for panel names to their contents
        panels1 = {panel['name']: panel for panel in previous_gui.get('panel', [])}
        panels2 = {panel['name']: panel for panel in current_gui.get('panel', [])}

        # Compare panel names for additions and removals
        added_panels = set(panels2) - set(panels1)
        removed_panels = set(panels1) - set(panels2)

        # Add new or removed panels to the results with their content
        for panel_name in added_panels:
            results['added'].append({panel_name: panels2[panel_name]})
        for panel_name in removed_panels:
            results['removed'].append({panel_name: panels1[panel_name]})

        # Check for 'children' additions or removals in panels present in both GUI data structures
        for panel_name in set(panels1) & set(panels2):
            children1 = panels1[panel_name].get('children', [])
            children2 = panels2[panel_name].get('children', [])

            # Comparing children by length here, you might need a more complex comparison
            if children1 != children2:
                if len(children1) < len(children2):
                    # A 'children' was added, include the panel content from gui_data2
                    results['added'].append({panel_name: panels2[panel_name]})
                else:
                    # A 'children' was removed, include the panel content from gui_data1
                    results['removed'].append({panel_name: panels1[panel_name]})

        return results

    @staticmethod
    def filter_noise(diff):
        pixel_change_th = 10
        items_to_delete = []  # List to hold keys of items that should be deleted
        for change_type, change_content in diff.items():
            if change_type == 'values_changed':
                for ele, change in change_content.items():
                    if (type(change["new_value"]) in [int, float]) & (type(change["old_value"]) in [int, float]):
                        if abs(change["new_value"] - change["old_value"]) < pixel_change_th:
                            items_to_delete.append((change_type, ele))  # Add the item key for deletion

        # Delete the items after the iteration is complete
        for change_type, ele in items_to_delete:
            del diff[change_type][ele]

    def get_action_panel(self, metadata, code=""):
        action_position = self.extract_last_position(code)
        if action_position:
            x, y = action_position

            for item in metadata["panel"]:
                panel_name = item["name"]
                x1, y1, x2, y2 = item["rectangle"]
                if x1 <= int(x) <= x2 and y1 <= int(y) <= y2:
                    return item, panel_name

            return metadata, "not in any panel"
        else:
            return metadata, "only keyboard operation"

    @staticmethod
    def extract_last_position(code_string):
        # Define a pattern to match the PyAutoGUI operations that include coordinates
        pattern = r"(moveTo|click|doubleClick|rightClick|dragTo|dragRel)\((\d+\.?\d*|\-\d+\.?\d*),\s*(\d+\.?\d*|\-\d+\.?\d*)"

        # Find all matches of the pattern in the string
        matches = re.findall(pattern, code_string)

        # If any matches are found, return the last one, otherwise return None
        if matches:
            last_match = matches[-1]
            # The coordinates are in the second and third group of the match
            position = [float(last_match[1]), float(last_match[2])]
            return position
        else:
            return None

    def compare_gui(self, pre_gui, cur_gui):
        diff = DeepDiff(pre_gui, cur_gui, ignore_order=True, view='text')
        self.filter_noise(diff)
        return diff

    def action_evaluation(self, task, pre_code, cur_code, cur_gui, difference):
        task = task.split(":")[1]
        if len(difference) < 25:
            gui_info = "Current GUI [The action may have already been completed; verify if the current GUI meets the results of the actions. If so, also consider it as Success]:\n" + \
                       "\n".join(format_gui(compress_gui(copy.deepcopy(cur_gui))))
        else:
            gui_info = "Difference:\n" + difference

        pre_code = self.extract_action_from_code(pre_code)
        pre_code = "Previous Finished Action:\n" + pre_code if pre_code else ""
        cur_code = self.extract_action_from_code(cur_code)
        cur_code = "Current Action:\n" + cur_code if cur_code else ""

        prompt = f'''An AI is operating the keyboard and mouse, aiding a user with tasks. 
Your role is to evaluate whether the AI's actions fulfill the user's request, as evidenced by changes in the GUI screenshots before and after the current action.

Responsibilities:
You are primarily responsible for determining the obvious success of the following actions:
If a popup window/panel appears or disappears as expected.
If parameters or typing content are modified.

Not your job:
You can only observe the text content of the GUI, not the visual appearance.
So, when certain button clicked, there indeed may not be any noticeable changes to the interface, like bold.
Therefore, by default, they should be considered as correct.

Output format: {{'Evaluation': str (Analysis of possible mistakes if action is wrong, in under 20 words.), 'Success': boolean (Current Action completion status), 'Finished': boolean (whether the main goal is achieved.),  'Next': str (What should be done next, if Finished is False)}}.
Main Goal:
{task}

{pre_code}

{cur_code}

{gui_info}

Note: Don't question the correctness of the Query.
If there is a popup window, please close it.

Output:'''

        # Previous Panel:
        # {pre_gui}
        response = run_llm(prompt, llm=self.llm, max_tokens=200, temperature=0, stop=[])

        # RE extract the answer of evaluation
        try:
            feedback = eval(response)
            return feedback['Success'], not feedback['Finished'], feedback['Evaluation'], feedback['Next']
        except:
            print("response of action evaluation is", response)
            assert ("The output format of action evaluation is wrong")

    @staticmethod
    def extract_action_from_code(code):
        # Extracting lines that start with a hash (#)
        # Removing the finish line. let model judge if the action is finished
        actions = [line for line in code.split('\n') if line.strip().startswith('#') and "finish" not in line]

        # Joining the comments back into a single string
        actions_text = '\n'.join(actions)
        return actions_text

