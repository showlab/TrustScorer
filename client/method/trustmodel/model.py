from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from typing import Dict, Type, Callable, List, Optional
import util
import logging
import os
import sys
import torch.distributed as dist
from typing import Callable, Dict, Type

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
import numpy as np
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange

logger = logging.getLogger(__name__)


class CrossEncoder(CrossEncoder):
    def __init__(
            self,
            model_name: str,
            num_labels: int = None,
            max_length: int = None,
            device: torch.device('cuda:0') = None,
            tokenizer_args: Dict = {},
            automodel_args: Dict = {},
            revision: Optional[str] = None,
            default_activation_function=None,
            classifier_dropout: float = None,
    ):
        self.config = AutoConfig.from_pretrained(model_name, revision=revision)
        classifier_trained = True
        if self.config.architectures is not None:
            classifier_trained = any(
                [arch.endswith("ForSequenceClassification") for arch in self.config.architectures]
            )

        if classifier_dropout is not None:
            self.config.classifier_dropout = classifier_dropout

        if num_labels is None and not classifier_trained:
            num_labels = 1

        if num_labels is not None:
            self.config.num_labels = num_labels

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=self.config, revision=revision, **automodel_args
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, **tokenizer_args)
        self.max_length = max_length

        self._target_device = device

        if default_activation_function is not None:
            self.default_activation_function = default_activation_function
            try:
                self.config.sbert_ce_default_activation_function = util.fullname(self.default_activation_function)
            except Exception as e:
                logger.warning(
                    "Was not able to update config about the default_activation_function: {}".format(str(e))
                )
        elif (
                hasattr(self.config, "sbert_ce_default_activation_function")
                and self.config.sbert_ce_default_activation_function is not None
        ):
            self.default_activation_function = util.import_from_string(
                self.config.sbert_ce_default_activation_function
            )()
        else:
            self.default_activation_function = nn.Sigmoid() if self.config.num_labels == 1 else nn.Identity()


    def _eval_during_training(self, val_dataloader, output_path, save_best_model, epoch, steps):
        """Runs evaluation during the training"""
        scores_list = self.val(val_dataloader, output_path=output_path, epochs=epoch, steps=steps)

        labels_list = []
        with tqdm(val_dataloader) as t:
            for features, labels in t:
                labels = labels.view(-1)
                labels_list.extend(labels)

        # Initialize counters for the statistics
        one_all = 0
        one_true = 0
        zero_all = 0
        zero_true = 0

        for score, label in zip(scores_list, labels_list):
            if label == 1:
                one_all += 1
                if score > 0.5:
                    one_true += 1
            elif label == 0:
                zero_all += 1
                if score <= 0.5:
                    zero_true += 1

        one_accuracy = one_true / one_all * 100
        zero_accuracy = zero_true / zero_all * 100

        accuracy = (one_accuracy + zero_accuracy) / 2
        print(f"Current Accuracy: {accuracy:.2f}% (epoch {epoch}, one_accuracy: {one_accuracy:.2f}%, zero_accuracy: {zero_accuracy:.2f}%)")
        # sys.exit(0)
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            print(f"Best Accuracy: {self.best_accuracy:.2f} (epch {epoch}). Will save this model.")
            output_path_best = os.path.join(output_path, "best")
            if save_best_model:
                self.save(output_path_best)

    def save(self, path: str) -> None:
        """
        Saves all model and tokenizer to path
        """
        if path is None:
            return

        logger.info("Save model to {}".format(path))
        self.model.module.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def fit(
        self,
        train_sampler: torch.utils.data.distributed.DistributedSampler = None,
        train_dataloader: DataLoader = None,
        val_dataloader: DataLoader = None,
        epochs: int = 1,
        loss_fct=None,
        activation_fct=nn.Identity(),
        scheduler: str = "WarmupLinear",
        warmup_steps: int = 10000,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params: Dict[str, object] = {"lr": 2e-5},
        weight_decay: float = 0.01,
        evaluation_steps: int = 0,
        evaluation_epochs: int = 0,
        output_path: str = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        callback: Callable[[float, int, int], None] = None,
        show_progress_bar: bool = True,
        gradient_accumulation_steps: int = 1,
        eval: bool = True,
    ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        :param train_dataloader: DataLoader with training InputExamples
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param loss_fct: Which loss function to use for training. If None, will use nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()
        :param activation_fct: Activation function applied on top of logits output of model.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        :param show_progress_bar: If True, output a tqdm progress bar
        """
        train_dataloader.collate_fn = self.smart_batching_collate

        if use_amp:
            from torch.cuda.amp import autocast

            scaler = torch.cuda.amp.GradScaler()

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        self.best_accuracy = -1
        num_train_steps = int(len(train_dataloader) * epochs)

        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(
                optimizer,
                scheduler=scheduler,
                warmup_steps=warmup_steps,
                t_total=num_train_steps,
            )

        if loss_fct is None:
            loss_fct = (
                nn.BCEWithLogitsLoss()
                if self.config.num_labels == 1
                else nn.CrossEntropyLoss()
            )

        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            self.model.zero_grad()
            self.model.train()

            # This is important for DDP training (i.e., enabling shuffling)
            train_sampler.set_epoch(epoch)

            with tqdm(train_dataloader, desc="train", smoothing=0.05, disable=not show_progress_bar) as t:
                for features, labels in t:
                    if use_amp:
                        with autocast():
                            model_predictions = self.model(**features, return_dict=True)
                            logits = activation_fct(model_predictions.logits)
                            if self.config.num_labels == 1:
                                logits = logits.view(-1)
                            loss_value = loss_fct(logits, labels)
                        scaler.scale(loss_value).backward()
                    else:
                        model_predictions = self.model(**features, return_dict=True)
                        logits = activation_fct(model_predictions.logits)
                        if self.config.num_labels == 1:
                            logits = logits.view(-1)
                        loss_value = loss_fct(logits, labels)
                        loss_value.backward()

                    t.set_postfix(loss=loss_value.item())

                    if (
                        training_steps + 1
                    ) % gradient_accumulation_steps == 0 or training_steps + 1 == len(
                        train_dataloader
                    ):
                        if use_amp:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), max_grad_norm
                            )
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), max_grad_norm
                            )
                            optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()

                    training_steps += 1

                    if (
                        eval and evaluation_steps > 0
                        and training_steps % evaluation_steps == 0
                    ):
                        self._eval_during_training(
                            val_dataloader,
                            output_path,
                            save_best_model,
                            epoch,
                            training_steps,
                        )

                        self.model.zero_grad()
                        self.model.train()

            if eval and epoch % evaluation_epochs == 0:
                self._eval_during_training(
                    val_dataloader, output_path, save_best_model, epoch, -1
                )

    def val(
        self,
        val_dataloader: DataLoader,
        output_path: str = None,
        save_best_model: bool = True,
        epochs: int = 0,
        steps: int = 0,
        show_progress_bar: bool = False,
        num_workers: int = 0,
        activation_fct=None,
        apply_softmax=False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
    ):
        """
        Performs predicts with the CrossEncoder on the given sentence pairs.

        :param batch_size: Batch size for encoding
        :param show_progress_bar: Output progress bar
        :param num_workers: Number of workers for tokenization
        :param activation_fct: Activation function applied on the logits output of the CrossEncoder. If None, nn.Sigmoid() will be used if num_labels=1, else nn.Identity
        :param convert_to_numpy: Convert the output to a numpy matrix.
        :param apply_softmax: If there are more than 2 dimensions and apply_softmax=True, applies softmax on the logits output
        :param convert_to_tensor: Convert the output to a tensor.
        :return: Predictions for the passed sentence pairs
        """
        val_dataloader.collate_fn = self.smart_batching_collate

        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )

        iterator = val_dataloader
        if show_progress_bar:
            iterator = tqdm(val_dataloader, desc="val")

        if activation_fct is None:
            activation_fct = self.default_activation_function

        pred_scores = []
        self.model.eval()
        self.model.cuda()
        with torch.no_grad():
            for features, labels in iterator:
                model_predictions = self.model(**features, return_dict=True)
                logits = activation_fct(model_predictions.logits)

                if apply_softmax and len(logits[0]) > 1:
                    logits = torch.nn.functional.softmax(logits, dim=1)
                pred_scores.extend(logits)

        if self.config.num_labels == 1:
            pred_scores = [score[0] for score in pred_scores]

        if convert_to_tensor:
            pred_scores = torch.stack(pred_scores)
        elif convert_to_numpy:
            pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])

        return pred_scores
