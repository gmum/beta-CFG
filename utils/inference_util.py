import os

import torch
import torch.multiprocessing as mp
from utils.log_util import set_seed

from latent_diffusion import get_solver
from torchvision.utils import save_image
from utils.calculate_metrics import CLIP

def run_exp(solver_config, exp_dir, prompts, start, stop, gpu_id):
    device = torch.device(f"cuda:{gpu_id}")
    solver = get_solver(
        solver_config.method, solver_config=solver_config, device=device
    )
    set_seed(solver_config.seed)
    os.makedirs(os.path.join(exp_dir, "images"), exist_ok=True)
    for prompt_idx, prompt in enumerate(prompts[start:stop], start):
        result = solver.sample(
            prompt=["", prompt], cfg_guidance=solver_config.cfg_scale, callback_fn=None
        )
        save_image(
            result,
            str(
                os.path.join(
                    exp_dir,
                    "images",
                    f"{prompt_idx}.png",
                )
            ),
            normalize=True,
        )
    torch.cuda.empty_cache()

def run_clip(exp_dir, prompts, clip_results, logger, start, stop, gpu_id):
    device = torch.device(f"cuda:{gpu_id}")
    metric = CLIP(exp_dir, prompts, logger, device, start, stop, 128)
    result = metric.compute()
    clip_results.put(result)

def run_parallel(func, gpus, **args):
    out = {}
    batch_size = int(len(args["prompts"]) / gpus)
    processes = []
    for gpu_id in range(gpus):
        start = gpu_id * batch_size
        stop = start + batch_size
        process_args = args.copy()
        process_args.update({"gpu_id": gpu_id, "start": start, "stop": stop})

        p = mp.Process(
            target=func,
            kwargs=process_args,
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    return processes, out
