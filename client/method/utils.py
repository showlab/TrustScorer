import os
import time
import io
import openai
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

os.environ["OPENAI_KEY"] = "" # your_api
client = OpenAI(api_key=os.environ.get('OPENAI_KEY'))


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
    response = client.chat.completions.create(
        model=llm,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop
    )
    return response.choices[0].message.content


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
    return out

