import os
import json

import logging
import pdb
import random
from datetime import timedelta
import hydra
import torch
from torch import nn
from dataloader import ScoreDataset, get_data_split
from hydra.core.hydra_config import HydraConfig
from model import CrossEncoder
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.backends import cudnn

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"Use model {cfg.model.model_name_or_path}")
    output_dir = HydraConfig.get().runtime.output_dir

    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=300), init_method='env://')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    print(local_rank)

    cudnn.benchmark = True
    torch.manual_seed(2025)

    train_data = get_data_split(
        cfg.data.data_path, cfg.data.train_split_file
    )
    train_dataset = ScoreDataset(train_data, neg_ratio=cfg.train.neg_ratio)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=cfg.train.batch_size, shuffle=(train_sampler is None)
    )

    val_data = get_data_split(
        cfg.data.data_path, cfg.data.val_split_file
    )
    val_dataset = ScoreDataset(val_data)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_dataloader = DataLoader(
        val_dataset, sampler=val_sampler, batch_size=cfg.val.batch_size, shuffle=False
    )

    if local_rank == 0:
        logger.info(f"Use device {'gpu' if torch.cuda.is_available() else 'cpu'}")
        logger.info(f"Use batch size {cfg.train.batch_size}")
        logger.info(f"Training data size {len(train_dataset)}")
        logger.info(f"Validating data size {len(val_dataset)}")

    device = torch.device('cuda:%d' % local_rank)

    # Initialize model
    model = CrossEncoder(
        cfg.model.model_name_or_path,
        num_labels=1,
        max_length=cfg.model.max_seq_length,
        device=device,
    )

    model.model = model.model.to(device)
    model.model = DDP(model.model, device_ids=[local_rank, ], output_device=0, find_unused_parameters=False)
    model.model = nn.SyncBatchNorm.convert_sync_batchnorm(model.model)

    warmup_steps = int(len(train_dataloader) * cfg.train.warmup_steps)
    if local_rank == 0:
        logger.info(f"Warmup steps {warmup_steps}")

    model.fit(
        optimizer_params={"lr": cfg.train.learning_rate},
        train_sampler=train_sampler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=cfg.train.epoch,
        use_amp=cfg.train.use_amp,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        eval=cfg.val.eval,
        evaluation_steps=cfg.val.evaluation_steps,
        evaluation_epochs=cfg.val.evaluation_epochs,
    )

    if local_rank == 0:
        model.save(output_dir)


if __name__ == "__main__":
    main()
