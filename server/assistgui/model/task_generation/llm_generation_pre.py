import os
import json
import imageio
import torch
from assistgui.model.utils import run_llm
from assistgui.model.base_module import BaseModule
from assistgui.planner.task_manager import parse_tasks, ordered_dict_to_tasks


class QueryToSteps(BaseModule):
    name = "query_to_steps"
    description = (
        '''Can transfer an query into detailed steps.
Invoke command: narration_reason(query, visual[i])'''
    )

    def __init__(self, llm="gpt-4-1106-preview", model=None, use_ocr=False):
        super(QueryToSteps, self).__init__()
        self.llm = llm

    def __call__(self, query, software):
        plan = self.query_to_steps(query, software)

        parsed_plan, current_task, root_task = self.turn_text_steps_to_iter(plan)
        return {'plan': parsed_plan, 'current_task': current_task, 'root_task': root_task}

    def query_to_steps(self, query, software):
        prompt = f'''As an expert in using {software}, please generate a procedure to complete the user's command. The output should be structured as follows:

Task 1: [Rephrase the User's Command]

Subtask 1: Click the "New Composition" button
Subtask 2: ...

Note:
1. Each subtask must detail specific actions within the GUI. 
2. There should be only one main task, but it may include multiple subtasks.
3. The Application is already open, so you do not need to open it.
4. The window is already properly focused and located.
5. Don't involve any locate action in your output, click action can automatically locate.

Example request:
{query}'''

        steps = run_llm(prompt, llm=self.llm, max_tokens=1000, temperature=0)

        return steps

    @staticmethod
    def turn_text_steps_to_iter(plan):
        parsed_tasks = parse_tasks(plan)
        root_task = ordered_dict_to_tasks(parsed_tasks)
        # The first task is "root", so move to the next task
        current_task = root_task.next()
        return parsed_tasks, current_task, root_task

    def video_to_steps(self, input_video):
        subtitle = input_video.subtitle
        prompt = f"{subtitle}\n\n" \
                 f"The above is the subtitle of an instructional video for achieving a kind of effect. " \
                 f"Please extract the procedure (only the procedure, delete unrelated part) for achieving the desired effect into the following format.\n " \
                 f"Note that subtitle is extracted by model, so it might contain noise, please correct them while output the content:\n" \
                 f"Task 1: Creat a new composition \nSubtask 1: Click New Composition button\nSubtask 2: ...\nTask 2: ..." \
                 f"\n\nLet's start:"

        # run llm
        steps = run_llm(prompt, llm=self.llm, max_tokens=1000, temperature=0)
        return steps

    def refine_steps(self, query, raw_steps, gui, video_name, software_name, **kwargs):
        gui_summary = self.get_gui_summary(gui)

        prompt = f'''The user is currently utilizing {software_name}. 
Please modify or delete unnecessary steps according to the user's unique requirements: 
Modifications could include:  
1) Delete unnecessary parts. for example, remove the importing footage step if the user's video has already been added to the track.
2) Change the content. For example, the video is about achieving an effect on the text "hello", but the user wants to generate "world".

Steps in Instructional Video for {video_name}:
{raw_steps}

User Query: {query}

Note that: 
1. The project file is already opened, no need to open it gain.
2. In your subtask, you must specify which button to click, which footage to manipulate, according to user query.

Refined steps:'''
        refined_steps = run_llm(prompt, llm=self.llm, max_tokens=1000, temperature=0)
        self.save_plan(query, refined_steps, **kwargs)
        return refined_steps

    @staticmethod
    def get_gui_summary(gui):
        # TODO: get gui summary
        return gui

    @staticmethod
    def save_plan(query, steps, **kwargs):
        plan_path = kwargs["instructional_video_path"].replace(".mp4", f"-{query}-plan.json")
        json.dump(steps, open(plan_path, "w", encoding="utf-8"), ensure_ascii=False)

    def load_plan_from_cache(self, query, **kwargs):
        plan_path = kwargs["instructional_video_path"].replace(".mp4", f"-{query}-plan.json")
        if os.path.exists(plan_path):
            print("load the plan from cache.")
            plan = json.load(open(plan_path, "r"))
            parsed_plan, current_task, root_task = self.turn_text_steps_to_iter(plan)
            return {'plan': parsed_plan, 'current_task': current_task}
        else:
            return None

