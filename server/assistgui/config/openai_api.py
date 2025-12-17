import os
import openai

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./omniqa-xxxxxx.json"  # a json file looks like omniqa-example.json
os.environ["OPENAI_KEY"] = "xxxx"  # xxxx is your api
openai.api_key = os.environ.get('OPENAI_KEY')
