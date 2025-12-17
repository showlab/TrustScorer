import os
import json
import imageio
import torch
import re
from assistgui.model.utils import run_llm
from assistgui.model.base_module import BaseModule
from assistgui.planner.task_manager import parse_tasks, ordered_dict_to_tasks
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer,util

from langchain.document_loaders import UnstructuredWordDocumentLoader,PyPDFium2Loader,DirectoryLoader,PyPDFLoader,TextLoader
from langchain.text_splitter import CharacterTextSplitter


class QueryToSteps(BaseModule):
    name = "query_to_steps"
    description = (
        '''Can transfer an query into detailed steps.
Invoke command: narration_reason(query, visual[i])'''
    )

    def __init__(self, llm="gpt-4-1106-preview", emb_model='sentence-transformers/sentence-t5-base', use_ocr=False):
        super(QueryToSteps, self).__init__()
        self.llm = llm
        self.emb_model= emb_model   # emb_model = 'sentence-transformers/sentence-t5-base'
        # self.threshold=0.85


    def __call__(self, query, software):
        #######################################################################################
        # support: Youtube, Word, PPT, Excel, DBS, Amazon, Google Chrome, Twitter, WeChat, AE, PR
        #######################################################################################

        print(f'Currently use {software}')
        plan= self.load_plan_from_database(query=query, software=software)

        parsed_plan, current_task, root_task = self.turn_text_steps_to_iter(plan)
        return {'plan': parsed_plan, 'current_task': current_task, 'root_task': root_task}

    def query_to_steps(self, query, software):
        prompt = f'''As an expert in using {software}, please generate a procedure to complete the user's command. The output should be structured as follows:

Task 1: [Rephrase the User's Command]

Subtask 1: Click the "New Composition" button
Subtask 2: ...

Note:
1. Each subtask must detail specific actions within the GUI. 
2. There should be only one main task, but it may include multiple subtasks.
3. The Application is already open, so you do not need to open it.
4. The window is already properly focused and located.
5. Don't involve any locate action in your output, click action can automatically locate.

Example request:
{query}'''

        steps = run_llm(prompt, llm=self.llm, max_tokens=1000, temperature=0)

        return steps

    @staticmethod
    def turn_text_steps_to_iter(plan):
        parsed_tasks = parse_tasks(plan)
        root_task = ordered_dict_to_tasks(parsed_tasks)
        # The first task is "root", so move to the next task
        current_task = root_task.next()
        return parsed_tasks, current_task, root_task

    def video_to_steps(self, input_video):
        subtitle = input_video.subtitle

        prompt = f"{subtitle}\n\n" \
                 f"The above is the subtitle of an instructional video for achieving a kind of effect. " \
                 f"Please extract the procedure (only the procedure, delete unrelated part) for achieving the desired effect into the following format.\n " \
                 f"Note that subtitle is extracted by model, so it might contain noise, please correct them while output the content:\n" \
                 f"Task 1: Creat a new composition \nSubtask 1: Click New Composition button\nSubtask 2: ...\nTask 2: ..." \
                 f"\n\nLet's start:"

        # run llm
        steps = run_llm(prompt, llm=self.llm, max_tokens=1000, temperature=0)
        return steps

    def refine_steps(self, query, raw_steps, gui, software_name, **kwargs):
        gui_summary = self.get_gui_summary(gui)

        prompt = f'''The user is currently utilizing {software_name}. 
Please modify or delete unnecessary steps according to the user's unique requirements: 
Modifications could include:  
1) Delete unnecessary parts. for example, remove the importing footage step if the user's video has already been added to the track.
2) Change the content. For example, the video is about achieving an effect on the text "hello", but the user wants to generate "world".

Raw steps are:
{raw_steps}

User Query: {query}

Note that: 
1. The Application is already open, so you do not need to open it.
2. In your subtask, you must specify which button to click, which footage to manipulate, according to user query.
3. Don't add unnecessary steps.
4. Import the file by click it instead of typing in the file name. 
5. Follow the format of raw steps Task 1: ...\nSubtask 1:...\nSubtask 2:...

Refined steps:'''
        refined_steps = run_llm(prompt, llm=self.llm, max_tokens=1000, temperature=0)

        return refined_steps

    @staticmethod
    def get_gui_summary(gui):
        # TODO: get gui summary
        return gui

    @staticmethod
    def save_plan(query, steps, **kwargs):
        plan_path = kwargs["instructional_video_path"].replace(".mp4", f"-{query}-plan.json")
        json.dump(steps, open(plan_path, "w", encoding="utf-8"), ensure_ascii=False)

    def load_plan_from_cache(self, query, **kwargs):
        pass

    def load_plan_from_database(self, query, software):  # qinchen wu

        text_splitter = CharacterTextSplitter(chunk_size=1600, chunk_overlap=0)
        loader = DirectoryLoader(f'/home/aiassist/{software}', glob='**/*.txt')
        doc = loader.load()
        split_docs = text_splitter.split_documents(documents=doc)

        model = HuggingFaceEmbeddings(model_name=self.emb_model)
        vector_hf = Chroma.from_documents(documents=split_docs, embedding=model, collection_name='huggingface_embed')
        response = vector_hf.similarity_search(query=query, k=1)

        plan = response[0].page_content
        print("retrived_docs: ", plan)
        return plan

    def llm_judge(self, retrived_plan, query, software):
        prompt = f'''The user is currently utilizing {software}. 
Please justify whether the provided steps is suitable for complete the task.
If so, just response '1', if not suitable, just response '0'.
The task is: {query}
The provided steps are :{retrived_plan} 

Note that: 
There should be a tolerance because the object name in the provided steps might be different.
Your response: '''
        response = run_llm(prompt, llm=self.llm, max_tokens=100, temperature=0)
        if '0' in response:
            return False
        if '1' in response:
            return True
