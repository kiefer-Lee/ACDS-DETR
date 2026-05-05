import os

import torch
import torch.distributed as dist


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        args.distributed = True
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.distributed = False
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def reduce_dict(input_dict):
    if not (dist.is_available() and dist.is_initialized()):
        return input_dict
    names = []
    values = []
    for k in sorted(input_dict.keys()):
        names.append(k)
        values.append(input_dict[k])
    values = torch.stack(values, dim=0)
    dist.all_reduce(values)
    values /= dist.get_world_size()
    return {k: v for k, v in zip(names, values)}

