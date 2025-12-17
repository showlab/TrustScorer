import sys
import os
import numpy as np

# path normalization
pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
)
sys.path.insert(0, pythonpath)

import assistgui.config.openai_api

# if you are in China, need import proxy
# import proxy

import json
from assistgui.model.gui_parser.gui_parser import GUIParser
from assistgui.explore.assistgpt_client import GUICapture
import cv2
import subprocess
import time
import re
import yaml

import numpy as np


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


def convert_int32(data):
    if isinstance(data, dict):
        return {k: convert_int32(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_int32(element) for element in data]
    elif isinstance(data, np.int32):  # Check if the data is int32
        return int(data)  # Convert int32 to int
    else:
        return data


def get_current_GUI(software_name, parser_softwarename):
    capture = GUICapture(target_window=software_name)
    gui_parser = GUIParser()
    meta_data, screenshot_path = capture.run("None", send_data=False)
    current_gui = gui_parser._run(parser_softwarename, meta_data, screenshot_path)
    write_json(current_gui, "parsed_data.json")
    return current_gui, screenshot_path


def get_softwarename(SOFTWARENAME):
    # Output: process_name, software_name, software_path
    # Note: Please Modify the software path Firstly in software_path.yaml
    print("Note: Please Modify the software path in software_path.yaml firstly")
    with open(
        os.path.expanduser("assistgui\explore\software_path.yaml"),
        "r",
        encoding="utf-8",
    ) as config:
        cfg = yaml.safe_load(config)
    if SOFTWARENAME == "premiere":
        return "Adobe Premiere Pro.exe", "Adobe Premiere Pro", cfg["PR"], "premiere"
    elif SOFTWARENAME == "word":
        return "WINWORD.EXE", ".* - Word", cfg["word"], "word"
    elif SOFTWARENAME == "after_effect":
        return "AfterFX.exe", "Adobe After Effects*", cfg["AE"], "after_effect"
    # elif SOFTWARENAME == "Acrobat":
    #     return "Acrobat.exe", ".* - Adobe Acrobat*", cfg["Acrobat"]
    elif SOFTWARENAME == "excel":
        return "EXCEL.EXE", ".* - Excel", cfg["excel"], "excel"
    else:
        assert (
            "Error: Not correctly set the global variable 'SOFTWARENAME' in exploer.py"
        )


def save_data(base_folder_path, task_id, screen_shot, meta, screenshot_path_list, operation_list):
    if not os.path.exists(base_folder_path):
        os.makedirs(base_folder_path)

    existing_folders = os.listdir(base_folder_path)
    existing_numbers = [
        int(re.search(r"\d+", folder).group())
        for folder in existing_folders
        if re.search(r"\d+", folder)
    ]
    max_number = max(existing_numbers) if existing_numbers else 0

    new_folder_path = os.path.join(base_folder_path, task_id, f"{max_number + 1}")
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    screen_shot.save(os.path.join(new_folder_path,f'/screenshot-0.png'))
    write_json(meta, os.path.join(new_folder_path,f'/metadata-1.json'))


def start_processs(software_path, project_path):
    command = f'start "" "{software_path}" "{project_path}"'
    process = subprocess.Popen(command, shell=True)
    print(process)
    time.sleep(15)
    print("Successfully start the process!")


def stop_processs(process_name="Adobe Premiere Pro.exe"):
    command = f'taskkill /F /IM "{process_name}"'
    subprocess.run(command, shell=True)


def get_projectname(project_path, SOFTWARENAME):
    start_index = project_path.rfind("\\") + 1
    if SOFTWARENAME == "premiere":
        end_index = project_path.rfind(".prproj")
    elif SOFTWARENAME == "word":
        end_index = project_path.rfind(".docx")
    elif SOFTWARENAME == "after_effect":
        end_index = project_path.rfind(".aep")
    elif SOFTWARENAME == "Acrobat":
        end_index = project_path.rfind(".pdf")
    elif SOFTWARENAME == "excel":
        end_index = project_path.rfind(".xlsx")
    end_index = project_path.rfind(".prproj")
    project_name = project_path[start_index:end_index]
    return project_name
