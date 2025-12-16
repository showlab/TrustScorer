import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import sys
import torch
import argparse
import json
import logging
import pdb
from tqdm import tqdm

import torch
from dataloader import ScoreDataset, get_data_split
from model import CrossEncoder
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

argparser = argparse.ArgumentParser()
argparser.add_argument("--model_path", type=str, default="ckpt")
argparser.add_argument("--model_name", type=str, default="trustmodel")
argparser.add_argument("--data_path", type=str, default="data")
argparser.add_argument("--split_file", type=str, default="test")
argparser.add_argument("--batch_size", type=int, default=30)
argparser.add_argument("--show_progress_bar", type=bool, default=True)
argparser.add_argument("--max_seq_length", type=int, default=512)
argparser.add_argument("--output_dir", type=str, default="output")


def main():
    args = argparser.parse_args()
    logger.info(f"Use model {args.model_path}")

    test_data = get_data_split(args.data_path, args.split_file)

    model_input = [[sample['code'],
                    (f"Final Goal is: {sample['goal']}\n"
                    f"Current Step is: {sample['step']}\n"
                    f"Current Task is: {sample['task']}\n"
                    f"Current GUI is: {sample['gui']}")
                    ] for sample in test_data]

    model_input_name = [sample['name'] for sample in test_data]

    logger.info(f"Use device {'gpu' if torch.cuda.is_available() else 'cpu'}")
    model = CrossEncoder(
        os.path.join(args.model_path, args.model_name, "best"),
        num_labels=1,
        max_length=args.max_seq_length,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )

    with torch.no_grad():
        pred_scores = model.predict(
            model_input,
            convert_to_numpy=True,
            show_progress_bar=args.show_progress_bar,
            batch_size=args.batch_size,
        )

        print(pred_scores)

        assert len(model_input_name) == len(pred_scores), "List lengths do not match"

        pred_scores_rounded = ["{:.2f}".format(score) for score in pred_scores]

        model_scores_dict = dict(zip(model_input_name, pred_scores_rounded))

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        output_file_path = f"{args.output_dir}/{args.model_name}_ace_scores.json"

        with open(output_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(model_scores_dict, json_file, indent=4, ensure_ascii=False)

        print(f"Saved to {output_file_path}")


if __name__ == "__main__":
    main()

