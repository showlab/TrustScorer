import os
import re
import copy
import json
from assistgui.model.command.utils import format_gui, compress_gui
from assistgui.model.utils import run_llm
from assistgui.model.base_module import BaseModule
from assistgui.model.command.action_wrapper import Drag, UpdateGUI, Finish, AddGuideLine, MovePlayHead, \
    AlignObjectToGuideLine, MaskObject


class CommandToInteraction:
    """Tool that adds the capability to locate image region with a natural language query."""

    name = "command_to_interaction"
    description = (
        '''
This tool can translate the user's language commands into Python code that can operate the keyboard and mouse.
Invoke command: command_to_interaction(query, visual[i])
:param query -> str, specific command. visual[i] -> image, the latest screenshot.
''')

    def __init__(self):
        super(CommandToInteraction, self).__init__()
        self.llm = "gpt-4-1106-preview"
        # TODO: add additional action wrapper API
        # TODO: add action wrapper conditioned on the software name
        self.available_api = {api.name: api for api in
                              [Drag()]}  #UpdateGUI(), AddGuideLine(), MovePlayHead(), AlignObjectToGuideLine(), MaskObject()]}
        self.available_api_illustration = "\n".join([f"{api.description}" for _, api in self.available_api.items()])

    def __call__(self,
                 current_task,
                 gui,
                 task_id=None,
                 use_gt=False,
                 gt_path=None,
                 ace_path=None,
                 gtwgui_path=None,
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
        message = ""

        if type(current_task) != str:
            # adjust the plan: next, continue, re-try, or finish
            current_task, message, history = self.next_task(current_task, error_message, next_step, pre_act_success_flag, pre_act_resume_flag, history)

        if message == "Success":
            code = ""
            trust_score = 1
        else:
            # translate query to interaction code
            code, trust_score = self.query_to_interaction(current_task, gui, history, software_name, task_id=task_id, use_gt=use_gt, gt_path=gt_path, ace_path=ace_path, gtwgui_path=gtwgui_path, **kwargs)

        return code, current_task, history, message, trust_score

    def next_task(self, current_task, error_message, next_step, success_flag, resume_flag, history):
        if success_flag:
            if resume_flag is True:
                # if the previous task didn't finish, just continue, no need to move to the next task
                if "# finish" not in history[-1]['code'][-1]:
                    history[-1]['code'][-1] += "\n# Previous codes are successfully executed. Continue.\n"
                    if len(history[-1]['code']) >= 5:
                        current_task = current_task.next()
                        server_message = "Force Next"
                    else:
                        server_message = "Continue"
                else:
                    # The actor previously finished the task, but the critic think it was not completed.
                    history[-1]['code'][-1] = history[-1]['code'][-1].replace("# finish", "")
                    history[-1]['code'][-1] += f"\n# Previous codes are successfully executed." \
                                               f"\n# Next, {next_step} " \
                                               f"\n# Continue.\n"

                    server_message = "Continue"
            elif resume_flag is False:
                # if the previous task is correctly executed, move to the next task
                current_task = current_task.next()
                server_message = "Next"
            elif resume_flag is None:
                # if the previous task didn't finish, just continue, no need to move to the next task
                if "# finish" not in history[-1]['code'][-1]:
                    history[-1]['code'][-1] += "\n# Previous codes are successfully executed. Continue.\n"
                    if len(history[-1]['code']) >= 5:
                        current_task = current_task.next()
                        server_message = "Force Next"
                    else:
                        server_message = "Continue"
                else:
                    current_task = current_task.next()
                    server_message = "Next"
        else:
            # if not, re-plan the following task\
            history[-1]['code'][-1] += f"# Previous actions are unsuccessful executed based on the updated gui. \n" \
                                       f"# Suggested possible reasons: {error_message}\n" \
                                       f"# Let's adjust the code.\n"
            if len(history[-1]['code']) >= 5:
                current_task = current_task.next()
                server_message = "Force Next"
            else:
                server_message = "Retry"

        if not current_task:
            server_message = "Success"
        elif not ("子任务" in current_task.name or "Subtask" in current_task.name):
            # only execute the subtask, if the current task is not a subtask, move to the next task
            # TODO: add to task manager
            current_task = current_task.next()

        return current_task, server_message, history

    def query_to_interaction(self, current_task, gui, history, software_name, task_id, use_gt=False, gt_path=None, ace_path=None, gtwgui_path=None):
        """translate query to interaction"""
        compressed_gui = copy.deepcopy(gui)
        compressed_gui = compress_gui(compressed_gui)
        compressed_gui = "\n".join(format_gui(compressed_gui))

        if type(current_task) != str:
            task_name = current_task.name
            main_goal = current_task.parent.name
        else:
            task_name = current_task
            main_goal = current_task

        summarized_history = self.get_code_history_for_current_task(history)
        finished_tasks = "\n".join(summarized_history['finished_tasks'])
        code_history = summarized_history['code']

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

Previous Finished Tasks: {finished_tasks}
Main Goal: {main_goal}
Current Task: {task_name}

Notes:
Don't write an algorithm to search on the GUI data, directly fill the coordinates in the corresponding API.
When using our provided custom API, do NOT need to import anything as it has been imported.
Please prioritize using our provided custom API (all positional arguments MUST be filled directly, rather than by using variable names, e.g., you cannot output drag(start_x, start_y, end_x, end_y, duration).)
Follow exactly the instructions in the task description.
MUST NOT click on elements not shown in the GUI, use update_gui() to wait gui refresh.
Add one line "# finish" at the end of the code.
You Must write the specific position for pyautogui function.
The screen resolution is 3840*2160.

{hints}

Now let's complete the code to achieve the command:
from pyautogui import click, write, hotkey, press, scroll, keyDown, keyUp
{code_history}
'''

        if not use_gt:
            # run llm
            code = run_llm(prompt, llm=self.llm, max_tokens=500, temperature=0, stop=["update_gui"])
            code = self.extract_code(code)
            if ace_path is not None:
                self.save_ace_code(task_id, current_task, code, compressed_gui, gt_path, ace_path)
        else:
            # load from gt json file. also feed 'current_task' for identify the index
            print(compressed_gui)
            print("Previous Finished Tasks: " + finished_tasks)
            print("Main Goal: " + main_goal)
            print("Current Task: " + task_name)
            code = self.load_gt_code(task_id, gt_path, current_task)
            if gtwgui_path is not None:
                self.save_gt_gui(task_id, current_task, compressed_gui, gt_path, gtwgui_path)

        if "# finish" not in code:
            code += "\nupdate_gui()"

        # post-process the code
        # 1) Delete annotation
        # 2) replace customized api with the corresponding code
        code = code.split("\n")
        out = []
        # if no score
        trust_score = "0.98765"
        for line in code:
        # if have score
        # trust_score = code[-1]
        # for line in code[:-2]:
            if not line.startswith("#"):
                for _, api in self.available_api.items():
                    if api.name in line:
                        # print(line)
                        line = line.replace(api.name, f"self.available_api['{api.name}']")
                        # print(line)
                        try:
                            line = eval(line)  # exec("line = " + line, locals())
                        except Exception as e:
                            print("An error occurred:", e)
                            pass
                        continue
            out.append(line)
        return "\n".join(out), trust_score

    @staticmethod
    def extract_code(input_string):
        # Regular expression to extract content starting from '```python' until the end if there are no closing backticks
        pattern = r'```python(.*?)(```|$)'
        # Extract content
        matches = re.findall(pattern, input_string, re.DOTALL)  # re.DOTALL allows '.' to match newlines as well
        # Return the first match if exists, trimming whitespace and ignoring potential closing backticks
        return matches[0][0].strip() if matches else input_string

    @staticmethod
    def load_gt_code(task_id, gt_path, current_task):
        # task_id = gt_path.split("/")[-2].split(".")[0]
        subtask_name = current_task.name
        task_name = current_task.parent.name
        if os.path.exists(gt_path):
            print("load gt action from " + gt_path)
            gt = json.load(open(gt_path, "r"))
            action = gt[task_id]["action"]
            parent_id = f"task_{int(task_name.split(':')[0].split(' ')[1]):02d}"
            sub_id = f"subtask_{int(subtask_name.split(':')[0].split(' ')[1]):02d}"
            gt_action = action[parent_id][sub_id]
        return gt_action

    @staticmethod
    def save_ace_code(task_id, current_task, code, compressed_gui, gt_path, ace_path):
        task_name = current_task.parent.name
        subtask_name = current_task.name
        parent_id = f"task_{int(task_name.split(':')[0].split(' ')[1]):02d}"
        sub_id = f"subtask_{int(subtask_name.split(':')[0].split(' ')[1]):02d}"
        if not os.path.exists(ace_path):
            gt = json.load(open(gt_path, "r"))
            ace = {}
            ace[task_id] = {}
            ace[task_id]["video"] = gt[task_id]["video"]
            ace[task_id]["query"] = gt[task_id]["query"]
            ace[task_id]["software"] = gt[task_id]["software"]
            ace[task_id]["project_file"] = gt[task_id]["project_file"]
            ace[task_id]["code"] = gt[task_id]["code"]
            ace[task_id]["plan"] = gt[task_id]["plan"]
            with open(ace_path, 'w', encoding='utf-8') as outfile:
                json.dump(ace, outfile, indent=4, ensure_ascii=False)

        ace = json.load(open(ace_path, "r"))
        ace_new = ace
        if "llm" not in ace_new[task_id]:
            ace_new[task_id]["llm"] = {}
        if parent_id not in ace_new[task_id]["llm"]:
            ace_new[task_id]["llm"][parent_id] = {}
        ace_new[task_id]["llm"][parent_id][sub_id] = code

        if "gui" not in ace_new[task_id]:
            ace_new[task_id]["gui"] = {}
        if parent_id not in ace_new[task_id]["gui"]:
            ace_new[task_id]["gui"][parent_id] = {}
        ace_new[task_id]["gui"][parent_id][sub_id] = compressed_gui

        if "flag" not in ace_new[task_id]:
            ace_new[task_id]["flag"] = {}
        if parent_id not in ace_new[task_id]["flag"]:
            ace_new[task_id]["flag"][parent_id] = {}
        ace_new[task_id]["flag"][parent_id][sub_id] = 1

        with open(ace_path, 'w', encoding='utf-8') as outfile:
            json.dump(ace_new, outfile, indent=4, ensure_ascii=False)

    @staticmethod
    def save_gt_gui(task_id, current_task, compressed_gui, gt_path, gtwgui_path):
        task_name = current_task.parent.name
        subtask_name = current_task.name
        parent_id = f"task_{int(task_name.split(':')[0].split(' ')[1]):02d}"
        sub_id = f"subtask_{int(subtask_name.split(':')[0].split(' ')[1]):02d}"
        if not os.path.exists(gtwgui_path):
            gt = json.load(open(gt_path, "r"))
            gtwgui = {}
            gtwgui[task_id] = {}
            gtwgui[task_id]["video"] = gt[task_id]["video"]
            gtwgui[task_id]["query"] = gt[task_id]["query"]
            gtwgui[task_id]["software"] = gt[task_id]["software"]
            gtwgui[task_id]["project_file"] = gt[task_id]["project_file"]
            gtwgui[task_id]["code"] = gt[task_id]["code"]
            gtwgui[task_id]["plan"] = gt[task_id]["plan"]
            gtwgui[task_id]["action"] = gt[task_id]["action"]
            with open(gtwgui_path, 'w', encoding='utf-8') as outfile:
                json.dump(gtwgui, outfile, indent=4, ensure_ascii=False)

        gtwgui = json.load(open(gtwgui_path, "r"))
        gtwgui_new = gtwgui
        if "gtgui" not in gtwgui_new[task_id]:
            gtwgui_new[task_id]["gtgui"] = {}
        if parent_id not in gtwgui_new[task_id]["gtgui"]:
            gtwgui_new[task_id]["gtgui"][parent_id] = {}
        gtwgui_new[task_id]["gtgui"][parent_id][sub_id] = compressed_gui
        print(f"add: {task_id}_gtgui_{parent_id}_{sub_id}.")

        with open(gtwgui_path, 'w', encoding='utf-8') as outfile:
            json.dump(gtwgui_new, outfile, indent=4, ensure_ascii=False)

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
