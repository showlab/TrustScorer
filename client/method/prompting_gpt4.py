import os
import sys
import json
from utils import run_llm

llm = "gpt-4-1106-preview"

input_folder_path = '../TrustBench/ace'

method_name = 'prompting_gpt4'
output_all_path = f'../results/{method_name}_all.json'
output_score_path = f'../results/{method_name}_scores.json'

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

extracted_data = {}
scores_data = {}
i = 1

for filename in os.listdir(input_folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(input_folder_path, filename)

        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

            for key, value in data.items():
                if 'llm' in value:
                    for task_key, task_value in value['llm'].items():
                        for subtask_key, subtask_value in task_value.items():
                            new_key = f"{key}_{task_key}_{subtask_key}"
                            if new_key not in extracted_data:
                                extracted_data[new_key] = {}

                            if "query" not in extracted_data[new_key]:
                                extracted_data[new_key]['query'] = {}
                            if "task" not in extracted_data[new_key]:
                                extracted_data[new_key]['task'] = {}
                            if "subtask" not in extracted_data[new_key]:
                                extracted_data[new_key]['subtask'] = {}
                            if "llm" not in extracted_data[new_key]:
                                extracted_data[new_key]['llm'] = {}
                            if "gui" not in extracted_data[new_key]:
                                extracted_data[new_key]['gui'] = {}
                            if "score" not in extracted_data[new_key]:
                                extracted_data[new_key]['score'] = {}
                            if "reason" not in extracted_data[new_key]:
                                extracted_data[new_key]['reason'] = {}

                            extracted_data[new_key]['query'] = data[key]['query']
                            extracted_data[new_key]['task'] = data[key]['plan'][task_key]['task_name']
                            extracted_data[new_key]['subtask'] = data[key]['plan'][task_key][subtask_key]
                            extracted_data[new_key]['llm'] = data[key]['llm'][task_key][subtask_key]
                            extracted_data[new_key]['gui'] = data[key]['gui'][task_key][subtask_key]

                            prompt = f"""
                            Please, based on the content provided below, choose a confidence score listed below, to show the possibility that the provided code can complete the current task described below. Must choose the score number only, i.e., ONLY the number before -. If you think the code has errors such as the positional arguments of function are not correct or the code has redundant or inadequate steps to achieve the current task, you should choose a score no larger than 0.5.
                            
                            0.0 - No Confidence Indicates an absolute lack of certainty, suggesting complete doubt or the absence of any belief in the outcome.
                            0.1 - Extremely Low Confidence Indicates almost complete uncertainty with minimal, if any, confidence in the outcome.
                            0.2 - Very Low Confidence Demonstrates very little confidence, signifying strong doubt and minimal trust in the outcome.
                            0.3 - Low Confidence Shows a low level of confidence, indicating significant uncertainty and limited trust in the outcome.
                            0.4 - Somewhat Low Confidence Suggests a slightly low confidence level, with uncertainty still prevailing over trust.
                            0.5 - Moderate Confidence Represents a moderate level of confidence, balancing doubt and trust equally.
                            0.6 - Somewhat High Confidence Implies a moderately high level of confidence, where trust begins to outweigh doubt.
                            0.7 - High Confidence Indicates a high level of confidence, showing strong belief in the outcome with minimal doubt.
                            0.8 - Very High Confidence Demonstrates a very high level of confidence, suggesting almost complete certainty in the outcome.
                            0.9 - Extremely High Confidence Shows nearly absolute certainty, with almost no doubt in the outcome.
                            1.0 - Absolute Confidence Represents complete certainty and absolute belief in the outcome, indicating no doubt whatsoever.
                            
                            The final goal is: {extracted_data[new_key]['query']}. We want to achieve this goal by several steps and each step includes one or more tasks.
                            
                            The current step is: {extracted_data[new_key]['task']}.
                            
                            The current task is : {extracted_data[new_key]['subtask']}.
                            
                            The python code is: {extracted_data[new_key]['llm']}. This code is generated based on the GUI content provided below, aiming to use pyautogui and the custom API to control the computer's mouse and keyboard to the current task.
                            GUI: [Note that: element format is "name [its position]", separate with comma].
                            {extracted_data[new_key]['gui']}.
                            
                            You MUST return the selected score in the first row and a brief explanation of the scoring (a few sentences) in the second row. Do not say anything else in you reply except the above two. 
                            """

                            response = run_llm(prompt, llm=llm, max_tokens=500, temperature=0, stop=None)

                            score = float(response.split('\n')[0])
                            reason = response.split('\n')[1]

                            extracted_data[new_key]['score'] = "{:.2f}".format(score)
                            extracted_data[new_key]['reason'] = reason

                            if new_key not in scores_data:
                                scores_data[new_key] = "{:.2f}".format(score)

                            print(f"{i} {new_key}'s score is: {extracted_data[new_key]['score']}")
                            print(f"{i} {new_key}'s reason is: {extracted_data[new_key]['reason']}")
                            i += 1


save_json(extracted_data, output_all_path)
print("save all info to {}".format(output_all_path))
save_json(scores_data, output_score_path)
print("save scores to {}".format(output_score_path))

