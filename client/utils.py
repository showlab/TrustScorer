import time
import subprocess
import pygetwindow as gw

def open_software(software_name):
    calc_windows = gw.getWindowsWithTitle(software_name)
    
    name2exe = {"calculator": "calc.exe"}
    
    if calc_windows:
        print("Calculator is already open.")
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