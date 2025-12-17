import os
import time
import io
import openai
import requests
from bs4 import BeautifulSoup
from google.cloud import vision
from openai import OpenAI

client = OpenAI(api_key=os.environ.get('OPENAI_KEY'))


def is_url_or_filepath(input_string):
    # Check if input_string is a URL
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    if url_pattern.match(input_string):
        return 'URL'

    # Check if input_string is a file path
    file_path = os.path.abspath(input_string)
    if os.path.exists(file_path):
        return 'File path'

    return 'Invalid'


def text_completion(prompt, llm, max_tokens=256, temperature=0, stop=None):
    response = openai.Completion.create(
        engine=llm,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop
    )
    return response['choices'][0]['text']


def chat_completion(prompt, llm, max_tokens=256, temperature=0, stop=None):
    print(llm, max_tokens, temperature, stop)
    response = client.chat.completions.create(
        model=llm,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop
    )
    return response.choices[0].message.content #response['choices'][0]['message']['content']


def multimodal_chat_completion(prompt, llm, max_tokens=256, temperature=0, stop=None):
    response = client.chat.completions.create(
        model=llm,
        messages=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
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
    print(f"========Input for LLM=======\n{prompt}\n============================")
    if llm in ['text-davinci-003']:
        out = text_completion(prompt, llm, max_tokens, temperature, stop)
    elif llm in ['gpt-4', 'gpt-4-0314', 'gpt-4-0613', 'gpt-3.5-turbo', 'gpt-3.5-turbo-0301', "gpt-4-1106-preview", "gpt-4-0125-preview"]:
        out = chat_completion(prompt, llm, max_tokens, temperature, stop)
    elif llm in ['gpt-4-vision-preview']:
        out = multimodal_chat_completion(prompt, llm, max_tokens, temperature, stop)
    elif llm in ['gpt-4-azure']:
        out = azure_chat(prompt, llm, max_tokens, temperature, stop)
    elif llm in ['gpt-35-turbo']:
        out = azure_completion(prompt, llm, max_tokens, temperature, stop)
    else:
        raise ValueError(f'Invalid llm: {llm}')
    print(f"========Output for LLM=======\n{out}\n============================")
    return out


def get_video_title(video_source, llm='gpt-35-turbo'):
    source_type = is_url_or_filepath(video_source)
    if source_type == 'URL':
        url = video_source
        # video_id = url.split("=")[-1]
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        title_tag = soup.find("title")
        video_title = title_tag.text.strip()
    elif source_type == 'File path':
        video_title = video_source
    else:
        video_title = f'Invalid video source: {video_source}'
    return video_title


def get_ocr(self, image_path):
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Perform OCR using the API
    response = self.client.text_detection(image=image)
    texts = response.text_annotations

    # Extract the full text from the response
    if texts:
        full_text = texts[0].description
        return f" Also some texts on the frame ({full_text})"
    else:
        return ""
