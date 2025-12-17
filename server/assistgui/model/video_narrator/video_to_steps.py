import os
import re
import json
import imageio
import torch
from assistgui.model.utils import run_llm
from assistgui.model.video_narrator import BaseNarrator
from assistgui.planner.task_manager import parse_tasks, ordered_dict_to_tasks


class VideoToSteps(BaseNarrator):
    name = "video_to_steps"
    description = (
        '''Can transfer an instructional video into detailed steps.
Invoke command: narration_reason(query, visual[i])'''
    )

    def __init__(self, llm="gpt-4-1106-preview", model=None, use_ocr=False):
        super(VideoToSteps, self).__init__()
        self.llm = llm

    def _run(self, query, input_image, input_video, task_id=None, use_gt=False, gt_path=None, **kwargs) -> dict:
        # loading gt
        if use_gt:
            plan = self.load_plan_from_gt(query, task_id, gt_path, **kwargs)
            if plan:
                return plan

        # loading cache
        plan = self.load_plan_from_cache(query, **kwargs)
        if plan:
            return plan

        # segment video with subtitle
        raw_plan = self.video_to_steps(input_video[0], **kwargs)

        plan = self.refine_steps(query, raw_plan, **kwargs)

        parsed_plan, current_task, root_task = self.turn_text_steps_to_iter(plan)
        return {'plan': parsed_plan, 'current_task': current_task, 'root_task': root_task}

    @staticmethod
    def turn_text_steps_to_iter(plan):
        parsed_tasks = parse_tasks(plan)
        root_task = ordered_dict_to_tasks(parsed_tasks)
        # The first task is "root", so move to the next task
        current_task = root_task.next()
        return parsed_tasks, current_task, root_task

    def video_to_steps(self, input_video, video_name, software_name, **kwargs):
        subtitle = input_video.subtitle
        prompt = f'''{subtitle}
        
The text above is the subtitle from an instructional video about {software_name}, titled "{video_name}". 
Its format includes one line for the time span and one line for the narration.

Please extract the procedure (only the procedure, omitting unrelated parts) to achieve the desired goal, and format it as follows:
```plan
Task 1: Create a new composition
Subtask 1: Click the 'New Composition' button
Subtask 2: ...
Task 2: ...
```

Note:
1. The subtitle was extracted by a model, so it may contain errors. Please correct them while outputting the content.
2. If the video does not provide specific steps, use your knowledge to fill in the gaps.

Let's begin:'''

        # run llm
        steps = run_llm(prompt, llm=self.llm, max_tokens=1000, temperature=0)

        steps = self.extract_plan(steps)
        return steps

    def refine_steps(self, query, raw_steps, gui, video_name, software_name, **kwargs):
        gui_summary = self.get_gui_summary(gui)

        prompt = f'''The user is currently utilizing {software_name}. 
Please modify or delete unnecessary steps according to the user's unique requirements: 
Modifications could include:  
1) Delete unnecessary parts. for example, remove the importing footage step if the user's video has already been added to the track.
2) Change the content. For example, the video is about achieving an effect on the text "hello", but the user wants to generate "world".

Steps in Instructional Video with title {video_name}:
{raw_steps}

User Query: {query}

Output format: 
```plan
Task 1: ...
Subtask 1: ...
Subtask 2: ...
Task 2: ...
```

Note that: 
1. The project file is already opened, no need to open it gain.
2. In your subtask, you must specify which button to click, which footage to manipulate, according to user query.

Refined steps:'''
        refined_steps = run_llm(prompt, llm=self.llm, max_tokens=1000, temperature=0)
        refined_steps = self.extract_plan(refined_steps)
        self.save_plan(query, refined_steps, **kwargs)
        return refined_steps

    @staticmethod
    def extract_plan(input_string):
        # Regular expression to extract content from '```python ... ```'
        pattern = r'```plan(.*?)```'
        # Extract content
        matches = re.findall(pattern, input_string, re.DOTALL)  # re.DOTALL allows '.' to match newlines as well
        # Return the first match if exists, else original
        return matches[0].strip() if matches else input_string

    @staticmethod
    def get_gui_summary(gui):
        return gui

    @staticmethod
    def save_plan(query, steps, **kwargs):
        if '16_create-animation-presets.1280x720at2000_h264_1_clipped.mp4' in kwargs["instructional_video_path"]:
            print("****** 113 ******")
            plan_path = kwargs["instructional_video_path"].replace(".mp4", f"-113-plan.json")
        else:
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

    @staticmethod
    def gt_plan_dict2str(data):
        output = ""

        task_number = 1
        for task_key in sorted(data['plan'].keys()):
            task = data['plan'][task_key]
            output += f"Task {task_number}: {task['task_name']}\n"
            subtask_number = 1
            for subtask_key in sorted(task.keys()):
                if subtask_key != 'task_name':
                    output += f"Subtask {subtask_number}: {task[subtask_key]}\n"
                    subtask_number += 1
            task_number += 1

        return output

    def load_plan_from_gt(self, query, task_id, gt_path, **kwargs):
        if os.path.exists(gt_path):
            print("load gt plan from " + gt_path)
            gt = json.load(open(gt_path, "r"))[task_id]
            plan = self.gt_plan_dict2str(gt)
            parsed_plan, current_task, root_task = self.turn_text_steps_to_iter(plan)
            return {'plan': parsed_plan, 'current_task': current_task}
        else:
            return None

