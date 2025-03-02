import json
import os
import argparse

import numpy as np

from utils.log_util import set_seed

from munch import munchify
from utils.data_util import load_json
from latent_diffusion import get_solver
from latent_sdxl import get_solver as get_solver_sdxl
from PIL import Image
from utils.callback_util import ComposeCallback
from pathlib import Path

if __name__ == '__main__':
    RANK = int(os.environ.get("RANK", "0"))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--method", type=str, default='ddim')
    parser.add_argument("--output_dir", type=str, default="out")
    parser.add_argument("--prompts", type=str, default="data/val10k.json")
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--images_per_prompt", type=int, default=1)
    parser.add_argument("--model", type=str, default='sd15', choices=["sd15", "sdxl"])
    parser.add_argument("--a", type=float, default=0.0)
    parser.add_argument("--b", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--callback", action="store_true")
    args = parser.parse_args()

    prompts = load_json(args.prompts)
    prompts = [item["prompt"] for item in prompts.values()]

    solver_config = {
        "method": args.method,
        "a": args.a,
        "b": args.b,
        "gamma": args.gamma,
        "cfg_scale": args.cfg_scale,
        "model": args.model, 
        "version": "v1",
        "num_sampling": 50
    }

    config_name = "-".join([str(f"{k}_{v}") for k, v in solver_config.items()])

    exp_dir = os.path.join(args.output_dir, config_name)

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "samples"), exist_ok=True)

    solver_config = munchify(solver_config)
    if args.model == "sdxl":
        solver = get_solver_sdxl(
            args.method, solver_config=solver_config, device='cuda:0'
        )
    else:
        solver = get_solver(
            args.method, solver_config=solver_config, device='cuda:0'
        )
    

    with open(str(os.path.join(exp_dir, "config.json")), "w") as f:
        json.dump(solver_config, f)

    for idx, prompt in enumerate(prompts):
        for i in range(args.images_per_prompt):
            image_id = idx * args.images_per_prompt + i

            if image_id % WORLD_SIZE != RANK:
                continue

            img_filename = f"{image_id:>06}.jpg"
            img_path = os.path.join(exp_dir, 'samples', img_filename)
            workdir = os.path.join(exp_dir, 'results', f"{image_id:>06}")

            if os.path.exists(img_path):
                continue
            if args.callback:
                callback = ComposeCallback(workdir=Path(workdir),
                               frequency=1,
                               callbacks=["history", "draw_noisy", 'draw_tweedie'])
            else:
                callback = None
            set_seed(image_id)
            if args.model == "sdxl":
                result = solver.sample(prompt1=["", prompt],
                                prompt2=["", prompt],
                                cfg_guidance=solver_config.cfg_scale,
                                target_size=(1024, 1024),
                                callback_fn=callback)
            else:
                result = solver.sample(
                    prompt=["", prompt], cfg_guidance=solver_config.cfg_scale, callback_fn=callback
                )
            result = np.clip(result.permute(0, 2, 3, 1).numpy() * 255, 0, 255).round().astype(np.uint8)[0]
            result_image = Image.fromarray(result)
            result_image.save(img_path, format='JPEG', quality=90, optimize=True)
