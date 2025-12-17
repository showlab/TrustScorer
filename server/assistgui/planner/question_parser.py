from assistgui.prompt_library.prompt_library import react_instruction, reason_instruction, question_parse_prefix, react_okvqa, react_nextqa
from assistgui.prompt_library.optimize_prompt import retry_prompt
from assistgui.callback_manager.utils import colored
from sentence_transformers import SentenceTransformer
import torch
import pickle
import re
from assistgui.model.utils import run_llm


class QuestionParser:
    def __init__(self,
                 tools,
                 demonstration_manager,
                 llm="gpt-3.5-turbo",
                 mode="react"):
        self.llm = llm
        self.mode = mode   # react, one run prediction
        self.tools = tools
        self.reload = False

        self.max_tokens_for_get_prompt = 2000
        self.max_tokens_for_split_question = 100
        self.temperature = 0

        self.demonstration_manager = demonstration_manager
        self.cache_task_intro = {}

    def parse_question(self, question, demonstrations=None, history="", previous_try=None, if_explanation=True,
                       if_output_format=True, if_examples=True):
        #
        if self.cache_task_intro.get(question, None):
            task_intro = self.cache_task_intro[question]
        else:
            task_intro = self.get_task_introduction(question, demonstrations, if_explanation, if_output_format, if_examples)
            self.cache_task_intro[question] = task_intro
            print(colored(task_intro, "introduction"))

        prompt = self.get_previous_try(previous_try)

        prompt += "\n\nLet's begin:"
        prompt += f"\nQuestion: {question}\n{history}\n"

        print(colored(prompt, "prompt"))

        out = run_llm(task_intro+prompt, llm=self.llm, temperature=self.temperature, stop=["\nObservation", "\n\tObservation"])

        print(colored(out, "output"))
        return out, self.parse_result(out)

    def get_task_introduction(self, question, demonstrations=None, if_explanation=True,
                              if_output_format=True, if_examples=True):
        # Prefix
        prompt = question_parse_prefix

        # Tool explanation
        if if_explanation:
            prompt += self.get_tool_explanation(self.tools)

        # Output format
        if if_output_format:
            prompt += self.get_reason_format(self.mode, self.tools)

        # Examples
        if if_examples:
            prompt += self.get_demonstrations(question, demonstrations)

        return prompt

    @staticmethod
    def get_tool_explanation(tools):
        return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

    @staticmethod
    def get_reason_format(mode, tools):
        if mode == "reason":
            output_format = reason_instruction
        elif mode == "react":
            output_format = react_instruction
        elif mode == "react_okvqa":
            output_format = react_okvqa
        elif mode == "react_nextqa":
            output_format = react_nextqa
        else:
            raise Exception(f"Unknown prompt model: {mode}")

        tool_names = ", ".join([tool.name for tool in tools])
        output_format = output_format.replace("TOOL_NAMES", tool_names)
        return output_format

    def get_demonstrations(self, question, demonstrations=None):
        demonstrations = demonstrations if demonstrations else self.demonstration_manager.retrieve_demonstration(question)
        out = f'''\nFor example: '''
        for demonstration in demonstrations:
            out += f'''\n{demonstration}'''  # load the demonstration prompts
        return out

    @staticmethod
    def get_previous_try(previous_try):
        if previous_try:
            previous_try = "\n\n".join(previous_try)
            return f'''\n{retry_prompt}\nPrevious try: {previous_try}'''
        else:
            return ""

    def parse_result(self, input_string):
        patterns = ["Thought:", "Action:", "Final Answer:"]
        parsed_prompt = {}

        for pattern in patterns:
            match = re.search(f'{pattern}(.*)', input_string)
            if match:
                parsed_prompt[pattern[:-1]] = match.group(1).strip()  # remove trailing colon and get the content

        if "Action" in parsed_prompt:
            parsed_prompt["Action"] = self.parse_action(parsed_prompt["Action"])

        return parsed_prompt

    @staticmethod
    def parse_action(input_string):
        pattern = r"(\w+)\((.*),\s*visual\[(.*)\]\)"
        match = re.search(pattern, input_string)

        if match:
            module_name = match.group(1)
            query = match.group(2)
            if match.group(3):
                i_list = [int(x) for x in match.group(3).split(",")]
            else:
                i_list = []
            return module_name, query.strip('\'"'), i_list
        else:
            print("String not in expected format")

    def reset_cache(self):
        self.cache_task_intro = {}

