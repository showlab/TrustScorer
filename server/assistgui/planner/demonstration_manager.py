import os
import json
import torch
import pickle
import openai
import datetime

from assistgui.demonstration.imageqa import imageqa_demonstrations
from assistgui.prompt_library.prompt_library import question_split_prompt
from assistgui.prompt_library.optimize_prompt import feedback_evaluation_prompt
from sentence_transformers import SentenceTransformer
from assistgui.model.utils import run_llm


class DemonstrationManager:
    def __init__(self, file_path=None, mode="reason", llm="gpt-3.5-turbo", save_every=100):
        self.raw_data = None
        self.mode = mode
        self.file_path = file_path
    
        self.llm = llm
        self.nli = SentenceTransformer('bert-base-nli-mean-tokens')

        self.added_count = 0
        self.save_every = save_every
        # load demonstration embeddings
        self.demonstrations, self.candidate_questions, self.query_types, self.demonstration_version = self.load_demonstration(file_path)
        self.query_type_embedding, self.rest_embedding = self.load_demonstration_embedding()

    @staticmethod
    def load_demonstration(file_path=None):
        if file_path:
            if os.path.exists(file_path):
                raw_data = json.load(open(file_path))
                demonstrations = raw_data['demonstrations']
                candidate_questions = list(demonstrations.keys())
                query_types = raw_data['type2question']
                demonstration_version = raw_data['version']
            else:
                raise FileNotFoundError(f"Cannot find demonstration file at {file_path}")
        else:
            demonstrations = imageqa_demonstrations['demonstrations']
            query_types = imageqa_demonstrations['type2question']
            demonstration_version = imageqa_demonstrations['version']
            candidate_questions = list(demonstrations.keys())

        return demonstrations, candidate_questions, query_types, demonstration_version

    def retrieve_demonstration(self, question):
        '''Two-step selection: first select query type, then select rest
            :return: a list of relevant questions for demonstrations
        '''
        query_type, rest = self.split_question(question)

        # query type selection
        query_embedding = self.nli.encode(query_type, convert_to_tensor=True)
        scores = torch.matmul(query_embedding.unsqueeze(0), torch.transpose(self.query_type_embedding, 0, 1))
        _, topk_idx = torch.topk(scores, k=1)
        topk_idx = topk_idx.squeeze(0).tolist()
        retrieved_query_types = [list(self.query_types.keys())[idx] for idx in topk_idx]

        # rest part selection
        rest_embedding = self.nli.encode(rest, convert_to_tensor=True)
        selected_demonstration = []
        for q_type in retrieved_query_types:
            rest_ques = self.query_types[q_type]
            rest_embed = self.rest_embedding[q_type]

            scores = torch.matmul(rest_embedding.unsqueeze(0), torch.transpose(rest_embed, 0, 1))
            _, topk_idx = torch.topk(scores, k=1)
            topk_idx = topk_idx.squeeze(0).tolist()

            for idx in topk_idx:
                ques = " ".join([q_type, rest_ques[idx]]).lower()
                selected_demonstration += [self.format_demonstration(ques)]

        return selected_demonstration

    def format_demonstration(self, question):
        prompt = self.demonstrations[question]
        if self.mode == "reason":
            # "filter Thought and observation in demonstration"
            prompt = self.filter_demonstration(prompt, ["Thought", "Observation", "Input"])
        elif self.mode == "react":
            prompt = self.filter_demonstration(prompt, [])
        else:
            raise Exception(f"Unknown prompt model: {self.mode}")

        demonstration = "\n".join(prompt)
        return f'{demonstration}'

    @staticmethod
    def filter_demonstration(demonstration, keys):
        filtered_demonstration = []
        for act in demonstration:
            act = act.split("\n")
            filtered_act = "\n".join([elem for elem in act if not any([key in elem for key in keys])])
            if filtered_act:
                filtered_demonstration.append(filtered_act)

        return filtered_demonstration

    def split_question(self, question):
        max_tokens = 100
        temperature = 0

        output = run_llm(question_split_prompt + question, self.llm, max_tokens, temperature)
        query_type, rest = None, None
        for line in output.split("\n"):
            if "Query Type:" in line:
                query_type = line.split("Query Type:")[1].strip()
            if "Rest:" in line:
                rest = line.split("Rest:")[1].strip()
        return query_type, rest

    def recompute_embedding(self):
        self.query_type_embedding, self.rest_embedding = self.load_demonstration_embedding(reload=True)

    def load_demonstration_embedding(self, reload=False):
        query_embed_path = f"cache/query_embeddings-{self.demonstration_version}.pt"
        rest_embed_path = f"cache/rest_embeddings-{self.demonstration_version}.pkl"
        if os.path.exists(query_embed_path) and not reload:
            print("loading query type embedding...")
            query_type_embedding = torch.load(query_embed_path)
        else:
            print("computing query type embedding...")
            query_types = list(self.query_types.keys())
            query_type_embedding = self.get_text_embedding(query_types)  # num_demonstration, 768
            print("saving query type embedding...")
            torch.save(query_type_embedding, query_embed_path)

        rest_embedding = {}
        if os.path.exists(rest_embed_path) and not reload:
            # load the rest embedding
            print("loading rest embedding...")
            rest_embedding = pickle.load(open(rest_embed_path, "rb"))
        else:
            print("computing rest embedding...")
            for query_type, candidates in self.query_types.items():
                rest_embedding[query_type] = self.get_text_embedding(candidates)   # num_demonstration, 768
            # save the rest embedding
            pickle.dump(rest_embedding, open(rest_embed_path, "wb"))

        return query_type_embedding, rest_embedding

    def get_text_embedding(self, text):
        demon_embed = self.nli.encode(text, convert_to_tensor=True)
        return demon_embed

    def add_demonstration(self, question, demonstration):
        # parse the question
        query_type, rest = self.split_question(question)

        # update the demonstration
        self.demonstrations[question] = demonstration
        self.candidate_questions.append(question)
        self.query_types[query_type].append(rest)

        self.query_type_embedding = torch.cat([self.query_type_embedding, self.get_text_embedding([query_type])], dim=0)
        self.rest_embedding[query_type] = torch.cat([self.rest_embedding[query_type], self.get_text_embedding([rest])], dim=0)

        self.added_count += 1

        if self.added_count % self.save_every == 0:
            self.save_demonstration()

    def save_demonstration(self):
        # Get current time
        now = datetime.datetime.now()
        # Convert time to string
        time_str = now.strftime("%H:%M:%S")

        suffix = f"{time_str}-{self.added_count}"
        out = {'demonstrations': self.demonstrations,
               "type2question": self.query_types,
               "version": f"{self.demonstration_version}-{suffix}"}

        # Get the file name without the extension, add the suffix to the file name
        file_name = os.path.splitext(os.path.basename(self.file_path))[0] + suffix

        # Combine the new file name with the directory and extension to get the new file path
        new_json_path = os.path.join(os.path.dirname(self.file_path), file_name + ".json")

        print(f"save demonstration at {new_json_path}")
        pickle.dump(out, open(new_json_path, "wb"))

