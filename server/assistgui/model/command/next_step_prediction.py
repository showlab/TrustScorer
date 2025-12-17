import os
import copy
import json
from assistgui.model.command.utils import format_gui, compress_gui
from assistgui.model.utils import run_llm
from assistgui.model.base_module import BaseModule
from assistgui.model.command.action_wrapper import Drag, UpdateGUI, Finish, AddGuideLine, MovePlayHead, \
    AlignObjectToGuideLine


class NextStepPrediction:
    """Tool that adds the capability to locate image region with a natural language query."""

    name = "command_to_interaction"
    description = (
        '''
This tool can translate the user's language commands into Python code that can operate the keyboard and mouse.
Invoke command: command_to_interaction(query, visual[i])
:param query -> str, specific command. visual[i] -> image, the latest screenshot.
''')

    def __init__(self):
        super(NextStepPrediction, self).__init__()
        self.llm = "gpt-4-0125-preview"
        self.available_api = {api.name: api for api in
                              [Drag()]}  #, AddGuideLine(), MovePlayHead(), AlignObjectToGuideLine()]}
        self.available_api_illustration = "\n".join([f"{api.description}" for _, api in self.available_api.items()])

    def __call__(self,
                 current_task,
                 gui,
                 input_image=None,
                 history=None,
                 error_message=None,
                 next_step=None,
                 pre_act_success_flag=None,
                 pre_act_resume_flag=None,
                 software_name=None,
                 **kwargs):
        """input """
        print("Execute the step")
        # feed the screenshot to the all available api
        self.update_gui(screenshot=input_image[0], gui=gui)

        # if message == "Success":
        #     code = ""
        # else:
            # translate query to interaction code
        code = self.query_to_interaction(current_task, gui, history, software_name)
        message = None

        return code, current_task, history, message

    def query_to_interaction(self, current_task, gui, history, software_name):
        """translate query to interaction"""
        compressed_gui = copy.deepcopy(gui)
        compressed_gui = compress_gui(compressed_gui)
        compressed_gui = "\n".join(format_gui(compressed_gui))

        task_name = current_task

        api = self.available_api_illustration

        software_hints = {'word': '''to select text, click one the left edge, hold it then drag to the right edge.
When selecting text, drag slowly, at least 3.5 seconds.
When typing font size: must press ctrl-A before typing.
When click dropdown of some button, must click the dropdown on the right side of the button.''',
                          "powerpoint": '''to select text, click one the left edge, hold it then drag to the right edge.
When selecting text, drag slowly, at least 3.5 seconds.
When typing font size: must press ctrl-A before typing.''',
                          "premiere": '''Don't move the playhead unless the task asks for it.
Press ctrl-A to select the parameter before changing it.
To change a setting, click on its value, not the name. For example, click "1000" instead of "Position" in "Position 1000."
Use the Tab key instead of pressing Enter after entering parameters.
For create new things, right-click on some panel, MUST right click on the empty space of it, better to be bottom of the panel. 
                          '''}

        hints = software_hints.get(software_name, "")

        prompt = f'''
Please, based on the GUI content provided below, use pyautogui and the following API to control the computer's mouse and keyboard to complete the specified task.
GUI: [Note that: element format is "name [its position]", separate with comma]
{compressed_gui}

Available API:
the pyautogui API imported

{api}

Current Task: {task_name}

Notes:
Don't write an algorithm to search on the GUI data, directly fill the coordinates in the corresponding API.
You must add one line "# finish" at the end of the code.
Please prioritize using our provided custom API.
{hints}
When you are required to click on a blank area, do the calculation, e.g., 
# the righest element is (100, 100), leave a margin (200, 0)
click(100 + 200, 0)

Now let's complete the code to achieve the command(only output the code and comment, Must Not output ```python):
from pyautogui import click, write, hotkey, press, scroll, keyDown, keyUp, moveTo, doubleClick
'''
        # MUST NOT click on elements not shown in the GUI, use update_gui() to wait gui refresh.
        # run llm
        code = run_llm(prompt, llm=self.llm, max_tokens=200, temperature=0, stop=["update_gui"])
        if "# finish" not in code:
            code += "\nupdate_gui()"

        # post-process the code
        # 1) Delete annotation
        # 2) replace customized api with the corresponding code
        code = code.split("\n")
        out = []
        for line in code:
            if not line.startswith("#"):
                for _, api in self.available_api.items():
                    if api.name in line:
                        line = line.replace(api.name, f"self.available_api['{api.name}']")
                        line = eval(line)  # exec("line = " + line, locals())
                        continue
            out.append(line)
        return "\n".join(out)

    def update_gui(self, screenshot, gui):
        # update screenshot for each api
        for _, api in self.available_api.items():
            api.update_screenshot(screenshot=screenshot, gui=gui)

    @staticmethod
    def check_resume(history):
        history_code = "\n".join(history[-1]['code']) if history else "# finish"
        if "# finish" in history_code:
            return False
        else:
            return True

    def get_code_history_for_current_task(self, history):
        # keep previous four steps
        finished_tasks, code = "", ""
        if history:
            if self.check_resume(history):
                # select self.history from -5 index to -1 index, needs to check length
                finished_tasks = [x['task'] for x in history[-5:-1]]
                code = "\n".join(history[-1]['code'])
            else:
                finished_tasks = [x['task'] for x in history[-4:]]
        return {"finished_tasks": finished_tasks, "code": code}
