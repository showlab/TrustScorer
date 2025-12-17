import json
import uiautomation as auto
from pywinauto import Application, Desktop
import os
import re
import datetime
import requests
import time


def get_control_properties(control, properties_list):
    prop_dict = {}
    for prop in properties_list:
        # justify if prop is a method of the control.
        if not hasattr(control, prop):
            continue
        else:
            prop_dict[prop] = getattr(control, prop)()
            if prop == "rectangle":
                rect = prop_dict[prop]
                prop_dict[prop] = [rect.left, rect.top, rect.right, rect.bottom]
    return prop_dict


class GUICapture:
    def __init__(
        self,
        target_window="Adobe Premiere Pro",
        file_explorer_name=None,
        cache_folder=".cache/",
    ):
        # judge if the cache folder exists
        self.task_id = self.get_current_time()
        self.target_window_name = target_window
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)

        self.cache_folder = os.path.join(cache_folder + self.task_id)
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)

        self.current_step = 0
        self.history = []
        self.file_explorer_name = file_explorer_name

    def run(
        self,
        query,
        send_data=False,
        run_code=False,
        task_id=None,
        reset=False,
        software=None,
    ):
        start = time.time()
        if self.target_window_name in ["Calculator", "Taskbar", "任务栏"]:
            self.app = Desktop(backend="uia").window(
                title=self.target_window_name, visible_only=False
            )
            meta_data = self.get_gui_meta_data_cal()
        elif self.target_window_name in ["file explorer"]:
            self.app = Desktop(backend="uia").window(
                title=self.file_explorer_name, visible_only=False
            )
            meta_data = self.get_gui_meta_data_cal()
        else:
            self.app = Application(backend="uia").connect(
                title_re=self.target_window_name
            )
            meta_data = self.get_gui_meta_data()
        screenshot_path = self.capture_screenshot()
        self.current_step += 1
        self.history.append((meta_data, screenshot_path))
        print("Time used: ", time.time() - start)
        start = time.time()

        if send_data:
            print("Sending data to server...")
            response = self.send_data(
                query, task_id=task_id, reset=reset, software=software
            )
            print("Time used: ", time.time() - start)
            code = response.json()["code"]
            print(response.json()["current_step"])
            print(code)

            if run_code:
                if response.json()["message"] == "success":
                    print("Success!")
                    return "Success"
                else:
                    exec(post_process_code(code))
                    self.run("None", send_data=True, run_code=True)
            else:
                return response.json()

        else:
            return meta_data, screenshot_path

    def send_data(self, query, reset=False, task_id=None, software=None):
        meta_data, screenshot_path = self.history[-1]
        files = {"image": open(screenshot_path, "rb")}

        response = requests.post(
            "http://localhost:6004/api/upload",
            data={
                "data": json.dumps(meta_data),
                "query": json.dumps(query),
                "task_id": json.dumps(task_id),
                "reset": json.dumps(reset),
                "software": json.dumps(software),
            },
            files=files,
        )
        if response.status_code == 200:
            print("Upload successfully!")
        return response

    def get_gui_meta_data_cal(self):
        # Connect to the application
        # Initialize data storage
        control_properties_list = ["friendly_class_name", "texts", "rectangle"]

        def recurse_controls(control, current_depth=0):
            children = control.children()
            child_data = []
            for child in children:
                child_data.append(
                    {
                        "properties": get_control_properties(
                            child, control_properties_list
                        ),
                        "children": recurse_controls(child, current_depth + 1),
                    }
                )
            return child_data

        meta_data = {}

        # Traverse the control tree
        self.app.set_focus()
        meta_data[self.target_window_name] = recurse_controls(self.app)

        # Save the meta data to a JSON file
        with open(f"{self.cache_folder}/metadata-{self.current_step}.json", "w") as f:
            json.dump(meta_data, f, indent=4)

        return meta_data

    def get_gui_meta_data(self):
        # Connect to the application
        # Initialize data storage
        control_properties_list = ["friendly_class_name", "texts", "rectangle"]

        def recurse_controls(control, current_depth=0):
            children = control.children()
            child_data = []
            for child in children:
                child_data.append(
                    {
                        "properties": get_control_properties(
                            child, control_properties_list
                        ),
                        "children": recurse_controls(child, current_depth + 1),
                    }
                )
            return child_data

        all_windows = self.app.windows()
        window_names = [window.window_text() for window in all_windows]
        meta_data = {}
        for window_name in window_names:
            if window_name:
                target_window = self.app.window(title=window_name)
                target_window.set_focus()

                # Traverse the control tree
                meta_data[window_name] = recurse_controls(target_window)

        # Save the meta data to a JSON file
        with open(f"{self.cache_folder}/metadata-{self.current_step}.json", "w") as f:
            json.dump(meta_data, f, indent=4)

        return meta_data

    def capture_screenshot(self):
        # save screenshot and return path
        screenshot_path = os.path.join(
            self.cache_folder, f"screenshot-{self.current_step}.png"
        )
        screenshot = auto.GetRootControl().ToBitmap()
        screenshot.ToFile(screenshot_path)
        return screenshot_path

    @staticmethod
    def get_current_time():
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def post_process_code(code):
    out = ["from pyautogui import *", "from time import sleep"]
    for line in code.split("\n"):
        if "#" not in line[:1] and line and "update_gui()" not in line:
            out.append(line + "\nsleep(1)\n")
    out = "\n".join(out)
    return out


if __name__ == "__main__":
    capture = GUICapture(".*Power BI*")
    meta_data, screenshot_path = capture.run("None", send_data=False)
