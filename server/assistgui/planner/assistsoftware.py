import pickle
from assistgui.model import *
from assistgui.inspector.memory_manager import MemoryManager
from assistgui.model.gui_parser.gui_parser_v4 import GUIParser


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

        self.current_task = None
        self.step = 0
        self.history = []
        self.screenshots = []
        self.plan = None

        self.task_id = None
        self.video_name = None
        self.software_name = None

    def run_step(self, query, meta_data, screenshot_path, **kwargs):
        # if receive a new query
        if self.step > self.maximum_step:
            return {"message": 'exceed maximum step'}

        parsed_tasks, server_message = None, ""

        # load the screenshot and video
        self.update_visual_input(screenshot_path, **kwargs)

        # Observe: Execute the steps
        print("Parsing GUI")
        gui = self.models['gui_parser'](meta_data=meta_data,
                                        screenshot_path=screenshot_path,
                                        software_name=self.software_name)

        if not self.current_task and self.step == 0:
            # Plan: initialize the task
            _, instructional_video = self.get_visual_input(-2)
            screenshot, _ = self.get_visual_input(-1)
            plan = self.models['video_to_steps'](query=query,
                                                 gui=gui,
                                                 input_image=screenshot,
                                                 input_video=instructional_video,
                                                 history=self.concise_history,
                                                 instructional_video_path=kwargs["instructional_video_path"],
                                                 video_name=self.video_name,  # TODO: update video_name
                                                 software_name=self.software_name)
            self.current_task, self.plan = plan['current_task'], plan['plan']
            success_flag, resume_flag, error_message, next_step = True, False, "", ""
            server_message += "Plan the task into steps based on the video.\n"
        else:
            # Critic: check if the previous task is correctly executed
            print("check if the previous task is correctly executed")
            success_flag, resume_flag, error_message, next_step = self.models['action_checker'](current_gui=gui,
                                                                                                history=self.history)
        # Actor: run
        screenshot, _ = self.get_visual_input(-1)
        code, current_task, history, message = self.models['command_to_interaction'](current_task=self.current_task,
                                                                                     input_image=screenshot,
                                                                                     history=self.history,
                                                                                     error_message=error_message,
                                                                                     next_step=next_step,
                                                                                     pre_act_success_flag=success_flag,
                                                                                     pre_act_resume_flag=resume_flag,
                                                                                     gui=gui,
                                                                                     software_name=self.software_name)
        server_message += message
        self.update_history(history, code, message, gui, current_task, success_flag, resume_flag)

        self.step += 1

        return {"code": code, "plan": parsed_tasks, "message": server_message, 'current_step': current_task.name}

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
            CommandToInteraction(),
            VideoToSteps(),
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

