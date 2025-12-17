import os
import time
import io
import openai
import requests

openai.api_key = "xxxx"  # xxxx is your api

def text_completion(prompt, llm, max_tokens=256, temperature=0, stop=None):
    response = openai.Completion.create(
        engine=llm,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop
    )
    return response['choices'][0]['text']


def gpt4v_completion(prompt, llm, max_tokens=256, temperature=0, stop=None):
    print(llm, max_tokens, temperature, stop)
    # from IPython.core.debugger import Pdb
    # Pdb().set_trace()
    response = openai.ChatCompletion.create(
        model=llm,
        messages=[
            # {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens, 
    )
    usage = response['usage']
    total_tokens = usage['total_tokens']
    print(f"\nTotal tokens used: {total_tokens}")
    return response['choices'][0]['message']['content']


def chat_completion(prompt, llm, max_tokens=256, temperature=0, stop=None):
    print(llm, max_tokens, temperature, stop)
    # from IPython.core.debugger import Pdb
    # Pdb().set_trace()
    response = openai.ChatCompletion.create(
        model=llm,
        messages=[
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop
    )
    return response['choices'][0]['message']['content']


def azure_completion(prompt, llm=None, max_tokens=256, temperature=0, stop=None):
    response = openai.Completion.create(
        engine="gpt-35-turbo",
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=stop)
    return response['choices'][0]['text']


def azure_chat(prompt, llm=None, max_tokens=256, temperature=0, stop=None):
    print("using azure chatgpt")
    # time.sleep(1)
    response = openai.ChatCompletion.create(
        engine="gpt-4",
        messages=[
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=1,
        presence_penalty=1,
        stop=stop)
    return response['choices'][0]['message']['content']


def run_llm(prompt, llm='gpt-3.5-turbo', max_tokens=256, temperature=0, stop=None):
    print(f"========Input for LLM===================================")
    if llm in ['text-davinci-003']:
        out = text_completion(prompt, llm, max_tokens, temperature, stop)
    elif llm in ["gpt-4-vision-preview"]:
        out = gpt4v_completion(prompt, llm, max_tokens, temperature, stop)
    elif llm in ['gpt-4', 'gpt-4-0314', 'gpt-4-0613', 'gpt-3.5-turbo', 'gpt-3.5-turbo-0301', "gpt-4-1106-preview"]:
        out = chat_completion(prompt, llm, max_tokens, temperature, stop)
    elif llm in ['gpt-4-azure']:
        out = azure_chat(prompt, llm, max_tokens, temperature, stop)
    elif llm in ['gpt-35-turbo']:
        out = azure_completion(prompt, llm, max_tokens, temperature, stop)
    else:
        raise ValueError(f'Invalid llm: {llm}')
    print(f"========Output for LLM=======\n{out}\n============================")
    return out


def format_gui(data, indent=0, in_elements=False, inner_elements=False):
    lines = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key == 'elements':
                lines.append(' ' * indent + str(key) + ':')
                lines.extend(format_gui(value, indent + 2, True))
            elif key in ['rectangle', 'position']:
                lines.append(' ' * indent + str(key) + ': ' + str(value))
            else:
                lines.append(' ' * indent + str(key) + ':')
                lines.extend(format_gui(value, indent + 2))
    elif isinstance(data, list):
        if in_elements:
            for value in data:
                lines.extend(format_gui(value, indent, False, True))
        elif inner_elements:
            element_line = []
            for element in data:
                name = element.get('name', '')
                rectangle = element.get('rectangle', [])
                position = element.get('position', [])
                if position:
                    element_line.append(f"{name} {position}")
                else:
                    element_line.append(f"{name} {rectangle}")
            lines.append(' ' * indent + ', '.join(element_line))
        else:
            for value in data:
                lines.extend(format_gui(value, indent))
    else:
        return [' ' * indent + str(data)]
    return lines


def compress_gui(com_gui):
    # compress gui
    for panel_item in com_gui.get('panel', []):
        if panel_item.get('name') in ['']:
            continue

        for row in panel_item.get("elements", []):
            if type(row) is list:
                for element in row:
                    element['position'] = [int((element['rectangle'][0] + element['rectangle'][2]) / 2),
                                           int((element['rectangle'][1] + element['rectangle'][3]) / 2)]
                    del element['rectangle']
            elif type(row) is dict:
                row['position'] = [int((row['rectangle'][0] + row['rectangle'][2]) / 2),
                                   int((row['rectangle'][1] + row['rectangle'][3]) / 2)]
                del row['rectangle']

    for panel_item in com_gui.get('menu', []):
        for row in panel_item.get("elements", []):
            row['position'] = [int((row['rectangle'][0] + row['rectangle'][2]) / 2),
                               int((row['rectangle'][1] + row['rectangle'][3]) / 2)]
            del row['rectangle']

    return com_gui

