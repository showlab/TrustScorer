import os
import sys
import json
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from datasets import concatenate_datasets
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import InputExample


########################### ScoreDataset Class ###########################
class ScoreDataset(Dataset):
    def __init__(self, data=None, neg_ratio=5):
        self.data = data
        self.neg_ratio = neg_ratio

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        code = sample['code']

        query = (
            f"Final Goal is: {sample['goal']}\n"
            f"Current Step is: {sample['step']}\n"
            f"Current Task is: {sample['task']}\n"
            f"Current GUI is: {sample['gui']}"
        )

        label = sample['label']

        return InputExample(
            texts=[
                code,
                query,
            ],
            label=label,
        )


############################### Load json data ###############################
def get_data_split(data_dir, split_file):
    # Create a list of file paths for all JSON files in the specified directory
    json_files = [os.path.join(data_dir, split_file, f) for f in os.listdir(os.path.join(data_dir, split_file)) if f.endswith('.json')]
    json_files = sorted(json_files)

    all_samples = []

    # Iterate over each file
    for file_path in json_files:
        # Open and read the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        # Assume each top-level key's value is a dictionary representing a sample
        # Add these samples to the all_samples list
        for name, sample in data.items():
            # Add the top-level key as 'name' in the sample dictionary
            sample_with_name = sample  # Start with the original sample
            sample_with_name['name'] = name  # Add the 'name' key
            all_samples.append(sample_with_name)

    return all_samples
