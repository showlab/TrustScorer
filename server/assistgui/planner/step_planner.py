import pickle
from assistgui.model import *
from assistgui.inspector.memory_manager import MemoryManager
from assistgui.model.gui_parser.gui_parser import GUIParser


class AssistGUI:
    def __init__(self, llm='gpt-4-1106-preview', cache_folder=".cache/", if_example=True):
        self.query = ""
        self.maximum_step = 50
        self.llm = llm
        self.concise_history = ""
        self.cache_folder = cache_folder

        self.tools = self.load_models()
        self.models = {tool.name: tool for tool in self.tools}
        self.memory_manager = MemoryManager()

        self.step = 0
        self.plan = None
        self.current_task = None
        self.root_task = None
        self.history = []   # [{task: ,code, ocr(json), screenshot_path, gui,  }]
        self.screenshots = []
        self.action_countdown = 100
        self.task_id = None
        self.video_name = None
        self.software_name = None

        self.current_state = self.get_current_state()

    def reset(self):
        self.current_state = self.get_current_state()

    def get_current_state(self, key=None, value=None):
        if key:
            self.current_state[key] = value
            return self.current_state
        else:
            return {
                "in_progress": False,
                "tasks": [],
                "current_task": -1,
                "current_step": None,
                "code": None
            }

    def get_current_task(self, is_retry, delete_prefix=True):
        if is_retry:
            print("retry current subtask")
            current_task = self.current_task.name
        else:
            print("move to next subtask")
            self.current_task = self.current_task.next()
            self.get_current_state("current_task", self.current_state['current_task'] + 1)

            if self.current_task is None:
                current_task = "Finished"
            else:
                if 'Subtask' not in self.current_task.name:
                    self.current_task = self.current_task.next()

                    # self.get_current_state("current_task", self.current_state['current_task'] + 1)
                current_task = self.current_task.name

        # format: Subtask i: task
        if delete_prefix:
            return current_task.split(": ")[1]
        else:
            return current_task

    def get_plan(self, query, software):
        self.get_current_state("in_progress", True)
        self.get_current_state("current_step", "Planning")

        if not self.plan:
            steps = self.models['query_to_steps'](query, software)
            self.plan = steps['plan']
            self.current_task = steps['current_task']
            self.root_task = steps['root_task']

            print(self.plan)
        format_plan = self.format_plan()
        self.get_current_state("in_progress", False)
        self.get_current_state("tasks", format_plan)
        self.get_current_state("current_step", "Planning Finished")
        return format_plan

    def format_plan(self):
        formated_plan = []
        for task, subtasks in self.plan['Root'].items():
            formated_plan.append(task)
            for subtask in subtasks:
                formated_plan.append(subtask)

        subtasks_only = [task for task in formated_plan if task.startswith('Subtask')]
        return subtasks_only

    def run_step(self, query, meta_data, screenshot_path, **kwargs):
        self.get_current_state("in_progress", True)
        self.get_current_state("current_step", "GUI Parsing")

        # if receive a new query
        if self.step > self.maximum_step:
            return {"message": 'exceed maximum step'}

        parsed_tasks, server_message = None, ""

        # load the screenshot and video
        self.update_visual_input(screenshot_path, **kwargs)

        # Observe: Execute the steps
        print("Parsing GUI")
        software_name = kwargs.get('software_name', None)
        self.current_stage = 'gui_parsing'
        gui = self.models['gui_parser'](meta_data=meta_data,
                                        screenshot_path=screenshot_path,
                                        software_name=software_name)

        # Actor: run
        self.get_current_state("current_step", "Generating Action")
        screenshot, _ = self.get_visual_input(-1)
        code, current_task, history, message = self.models['command_to_interaction'](current_task=query,
                                                                                     input_image=screenshot,
                                                                                     history=self.history,
                                                                                     error_message="",
                                                                                     next_step="",
                                                                                     pre_act_success_flag=True,
                                                                                     pre_act_resume_flag=True,
                                                                                     gui=gui,
                                                                                     software_name=self.software_name)
        server_message += str(message)
        # self.update_history(history, code, message, gui, current_task, success_flag, resume_flag)
        self.step += 1
        self.get_current_state("in_progress", False)
        self.get_current_state("current_step", "None")
        self.get_current_state("code", code)
        return {"code": code, "plan": parsed_tasks, "message": server_message, "current_task": current_task}

    def update_history(self, history, code, message, gui, current_task, sucess_flag, resume_flag):
        self.history = history
        self.current_task = current_task
        if message in ['Continue', 'Re-try']:
            # the task doesn't change, so only append the code and gui
            self.history[-1]['code'].append(code)
            self.history[-1]['gui'].append(gui)
            self.history[-1]['success_flag'].append(sucess_flag)
            self.history[-1]['resume_flag'].append(resume_flag)
        else:
            self.history.append({"task": current_task.name if current_task else None,
                                 "code": [code],
                                 "gui": [gui],
                                 "success_flag": [sucess_flag],
                                 "resume_flag": [resume_flag]})
        pickle.dump(self.history, open(f"{self.cache_folder}/history.pkl", "wb"))

    def update_visual_input(self, screenshot_path, **kwargs):
        instructional_video_path = kwargs.get('instructional_video_path', None)
        if instructional_video_path:
            self.load_media(instructional_video_path, user_comment=f"user provided video at step-{self.step}")

        # self.screenshots.append(screenshot_path)
        self.load_media(screenshot_path, user_comment=f"screenshot at step-{self.step}")

    def load_media(self, file_name, user_comment="user provided video"):
        self.memory_manager.append_visual_input(file_name, user_comment=user_comment)

    @staticmethod
    def load_models():
        print("==Loading models...==")
        tools = [
            NextStepPrediction(),
            VideoToSteps(),
            QueryToSteps(),
            GUIParser(),
            ActionChecker(),
        ]
        return tools

    def get_visual_input_summary(self, index):
        summary = self.memory_manager.get_description(index)
        return summary

    def get_visual_input(self, action_input):
        """Get the visual input for the action input."""
        image, video = self.memory_manager.get_visual_input(action_input)
        return image, video

    def reset(self):
        self.memory_manager.reset()
        self.current_task = None
        self.step = 0
        self.history = []
        self.screenshots = []
        self.software_name = None
        self.plan = None

