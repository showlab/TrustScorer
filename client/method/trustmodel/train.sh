CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node=4 --master_port=12347 mtrain.py
#CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.run --nproc_per_node=1 --master_port=12346 mtrain.py
