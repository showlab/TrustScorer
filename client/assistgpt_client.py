import json
import uiautomation as auto
import time
import subprocess
import pygetwindow as gw
from pywinauto import Application, Desktop
import os
import sys
import re
import datetime
import requests
import time
from termcolor import colored
import subprocess
import win32gui
import win32process
import psutil
from pywinauto.findwindows import find_windows
from pyautogui import click


def get_control_properties(control, properties_list):
    prop_dict = {}
    for prop in properties_list:
        # justify if prop is a method of the control.
        if not hasattr(control, prop):
            continue
        else:
            try:
                prop_dict[prop] = getattr(control, prop)()
            except Exception as e:
                continue
            if prop == 'rectangle':
                rect = prop_dict[prop]
                prop_dict[prop] = [rect.left, rect.top, rect.right, rect.bottom]
    return prop_dict


class GUICapture:
    """
    A class to capture and interact with a GUI of a specified application.
    """
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    OTHER = '\033[95m'
    RED = '\033[91m'
    END = '\033[0m'

    def __init__(self, cache_folder='.cache/'):
        """
        Initialize the GUI Capture instance.
        """
        self.task_id = self.get_current_time()
        self.ensure_directory_exists(cache_folder)
        self.cache_folder = os.path.join(cache_folder, self.task_id)
        self.ensure_directory_exists(self.cache_folder)
        self.current_step = 0
        self.history = []
        self.port = 6006

    @staticmethod
    def ensure_directory_exists(path):
        """
        Ensure that a directory exists; if not, create it.
        """
        if not os.path.exists(path):
            os.makedirs(path)

    def auto_start_application(self, software_path, project_path):
        # TODO: xiangwu
        command = f'start "" "{software_path}" "{project_path}"'
        process = subprocess.Popen(command, shell=True)
        print(process)
        time.sleep(15)  # Waiting for the app to launch…
        print("Successfully start the process!")

    def stop_application(self):
        pass

    def record(self):
        """
        Record the screen.
        """
        pass

    def run(self, query, send_data=False, run_code=False, task_id=None, reset=False, software=None, port=6006, save_path=None):
        """
        Execute the capture process.
        """
        start = time.time()
        time.sleep(3)
        self.connect_to_application(software)
        meta_data = self.get_gui_meta_data(software)
        with open(os.path.join(self.cache_folder, f"meta-data-{self.current_step}.json"), "w") as f:
            json.dump(meta_data, f)
        screenshot_path = self.capture_screenshot(save_path)
        self.current_step += 1
        self.history.append((meta_data, screenshot_path))
        print(f"Time used: {time.time() - start}")
        start = time.time()

        if send_data:
            print("Sending data...")
            response = self.send_data(query, task_id=task_id, reset=reset, software=software, port=port)
            # TODO: start recording (before 1 second)
            if run_code:
                self.handle_response(response, start, run_code, software, task_id, port=port)
            else:
                return response
        else:
            return meta_data, screenshot_path

    def send_data(self, query, reset=False, task_id=None, software=None, is_retry=False, port=6006):
        meta_data, screenshot_path = self.history[-1]
        files = {'image': open(screenshot_path, 'rb')}

        response = requests.post(
            f'http://localhost:{port}/api/upload',
            data={'data': json.dumps(meta_data),
                  'query': json.dumps(query),
                  'task_id': json.dumps(task_id),
                  'is_plan': json.dumps(reset),
                  'is_retry': json.dumps(is_retry),
                  'software': json.dumps(software)},
            files=files
        )
        if response.status_code == 200:
            print('Upload succeed!')
        else:
            print('Upload failed!')
        return response

    def connect_to_application(self, software_name):
        """
        Connect to the target application.
        """

        if software_name in ["explorer", "project"]:
            software_name = get_explorer_windows()[0]

        try:
            self.app = Application(backend="uia").connect(title_re=f".*{software_name}*")
        except Exception as e:
            print(f"Error connecting to application: {e}")
            try:
                print("Try to connect to the application by using the window name.")
                self.app = self.detect_duplicate_name_windows(software_name)
            except Exception as e:
                print(f"Error connecting to application: {e}")

    def detect_duplicate_name_windows(self, software_name):
        window_handles = find_windows(title_re=f".*{software_name}*", visible_only=False)

        if window_handles:
            first_window_handle = window_handles[1]

            app = Application(backend="uia").connect(handle=first_window_handle)

            return app

        else:
            print("Unable to find a matching window.")
            return None

    def handle_response(self, response, start_time, run_code, software, task_id, port):
        """
        Handle the response from data sending operation.
        """
        print(response.json())
        print(f"{self.OTHER}==Plan==\n{self.END}")
        print(response.json()['plan'])
        message = response.json().get('message', '')
        print(f"{self.BLUE}==Time used=={self.END} {time.time() - start_time}")
        print(f"{self.RED}==Message==\n{self.END}" + message)
        if response.json()['message'] == 'Success':
            return "Success"

        if 'code' in response.json():
            code = response.json()['code']
            trust_score = response.json()['trust_score']
            print(f"{self.GREEN}==Current Step==\n{self.END}" + response.json()['current_step'])
            print(f"{self.CYAN}==Code==\n{self.END}" + code)
            print(f"{self.OTHER}==Trust Score==\n{self.END}" + trust_score)
        else:
            return response.json()

        if run_code:
            # Caution: Using exec. Ensure code is safe to execute.
            try:
                exec(post_process_code(code))
            except Exception as e:
                print("An error occurred:", e)
                pass
            # stop recording (after 3 seconds)
            self.run("None", send_data=True, run_code=True, software=software, task_id=task_id, port=port)
        else:
            return response.json()

    def get_gui_meta_data(self, software):
        # Connect to the application
        # Initialize data storage
        control_properties_list = ['friendly_class_name', 'texts', 'rectangle', 'automation_id']
        software_th = {"Calculator": 100,
                       'after_effects': 3,
                       'Word': 6,
                       'Excel': 6,
                       'Edge': 100}
        th = software_th.get(software, 7)

        def recurse_controls(control, current_depth=0):
            children = control.children()
            # remove excel location indicator
            children = [ctrl for ctrl in children if ctrl.friendly_class_name() != 'ComboBox']

            child_data = []
            if current_depth > th:
                return []
            for child in children:
                child_data.append({
                    'properties': get_control_properties(child, control_properties_list),
                    'children': recurse_controls(child, current_depth + 1)
                })

            return child_data

        all_windows = self.app.windows()
        window_names = [window.window_text() for window in all_windows]
        window_names.reverse()
        meta_data = {}
        for window_name in window_names:
            if window_name:
                target_window = self.app.window(title=window_name)

                # Traverse the control tree
                meta_data[window_name] = recurse_controls(target_window)

        return meta_data

    def capture_screenshot(self, save_path):
        # save screenshot and return path
        if save_path:
            screenshot_path = save_path
        else:
            screenshot_path = os.path.join(self.cache_folder, f'screenshot-{self.current_step}.png')

        screenshot = auto.GetRootControl().ToBitmap()
        screenshot.ToFile(screenshot_path)
        return screenshot_path

    @staticmethod
    def get_current_time():
        return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def post_process_code(code):
    def extract_python_code(input_string):
        # Regular expression to extract content from '```python ... ```'
        pattern = r'```python(.*?)```'
        # Extract content
        matches = re.findall(pattern, input_string, re.DOTALL)  # re.DOTALL allows '.' to match newlines as well
        # Return the first match if exists, else original
        return matches[0].strip() if matches else input_string

    code = extract_python_code(code)
    out = ["from pyautogui import *", "from time import sleep"]
    for line in code.split("\n"):
        if "```" in line:
            continue
        stripped_line = line.lstrip()  # Remove left whitespaces to check the content
        # Add `sleep(1)` after each line of actual code that isn't a comment or 'update_gui()' call
        if not stripped_line.startswith("#") and "update_gui()" not in stripped_line:
            indent = len(line) - len(stripped_line)  # Calculate the leading spaces (indentation)
            out.append(" " * indent + stripped_line)  # Maintain the same indentation
            if not stripped_line.startswith("if") and not stripped_line.startswith("for") and not stripped_line.startswith("while"):
                out.append(" " * indent + "sleep(0.1)")  # Add sleep with the same indentation
    out = "\n".join(out)
    out += "\nsleep(3)"
    return out


def get_all_windows():
    all_windows = gw.getAllWindows()
    all_windows_name = [win.title for win in all_windows if win.title]
    all_windows_name = simplify_window_names(all_windows_name)
    return all_windows_name


def simplify_window_names(names):
    simplified_names = []
    for name in names:
        # Split the name by '-' and strip whitespace
        parts = [part.strip() for part in name.split('-')]
        # Use the part after the last '-' if available, otherwise the original name
        simplified_name = parts[-1] if len(parts) > 1 else name
        simplified_names.append(simplified_name)
    return simplified_names


def open_software(software_name):
    windows = gw.getWindowsWithTitle(software_name)

    name2exe = {"calculator": "calc.exe"}

    if windows:
        print("Calculator is already open.")
        for window in windows:
            window.close()  # Close each window that matches
        time.sleep(2)  # Wait for the software to close completely
    else:
        print("Calculator is not open, opening now.")

    subprocess.Popen(name2exe[software_name.lower()])
    time.sleep(2)

    maximize_window(software_name)


def maximize_window(title):
    """Maximize the window"""
    windows = gw.getWindowsWithTitle(title)
    if windows:
        window = windows[0]  # Assume the first window is target window
        if not window.isMaximized:
            window.maximize()
            print(f"Window '{title}' has been maximized.")
        else:
            print(f"Window '{title}' is already maximized.")
    else:
        print(f"No window with the title '{title}' found.")


def web_collector(capture, url, save_folder="website"):
    save_path = os.path.join(save_folder, url_to_filename(url))
    response = capture.run("None", software='Chrome', send_data=True, run_code=False, reset=True,
                           port=6006, save_path=f"{save_path}.png")
    json.dump(response.json()['gui'], open(f"{save_path}.json", "w"))

    print(f"The meta data is saved in {save_path}.json & {save_path}.png")


def url_to_filename(url):
    # Remove illegal characters for filenames
    # You might also want to remove or replace HTTP protocols and such to make it more readable
    filename = url.replace('http://', '').replace('https://', '').replace('www.', '')
    # These include: \ / : * ? " < > |
    # Replace illegal characters with underscores or another preferred character
    filename = re.sub(r'[\\/:*?"<>|]', '_', filename)
    # Shorten the filename or split if necessary to avoid overly long filenames
    if len(filename) > 255:  # Typical max length for file systems
        filename = filename[:255]
    return filename


def get_explorer_windows():
    explorer_windows = []

    def enum_window_callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd) and len(win32gui.GetWindowText(hwnd)) > 0:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            try:
                process = psutil.Process(pid)
                if process.name() == "explorer.exe":
                    explorer_windows.append(win32gui.GetWindowText(hwnd))
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

    win32gui.EnumWindows(enum_window_callback, None)
    return explorer_windows


if __name__ == '__main__':
    capture = GUICapture('Adobe Premiere Pro')
    meta_data, screenshot_path = capture.run("None", send_data=False)

