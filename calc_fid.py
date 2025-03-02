import os
import argparse
import json
import numpy as np
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import (
    calculate_activation_statistics,
    calculate_frechet_distance,
)
from utils.data_util import load_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="out")
    parser.add_argument("--prompts", type=str, default="data/val10k.json")
    parser.add_argument("--coco", type=str, default="coco10k")
    parser.add_argument("--coco_cache", type=str, default="coco_cache.npz")
    args = parser.parse_args()

    device = "cuda:0"

    # Load prompts
    config = load_json(args.prompts)
    coco_files = [os.path.join(args.coco, item["filename"]) for item in config.values()]

    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    if not os.path.exists(args.coco_cache):
        m1, s1 = calculate_activation_statistics(coco_files, model, 1, dims, device, 4)
        np.savez_compressed(args.coco_cache, mu=m1, sigma=s1)
    else:
        with np.load(args.coco_cache) as f:
            m1, s1 = f["mu"][:], f["sigma"][:]

    for config_name in os.listdir(args.output_dir):
        exp_dir = os.path.join(args.output_dir, config_name)
        samples_dir = [
            os.path.join(exp_dir, "samples", item)
            for item in os.listdir(os.path.join(exp_dir, "samples"))
        ]
        fid_dir = os.path.join(exp_dir, "fid")

        if len(samples_dir) != 10_000:
            print(f"Found only {len(samples_dir)}. Skipped")
            continue
        if os.path.exists(os.path.join(fid_dir, "result.json")):
            continue

        m2, s2 = calculate_activation_statistics(samples_dir, model, 1, dims, device, 4)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)

        os.makedirs(fid_dir, exist_ok=True)
        with open(str(os.path.join(fid_dir, "result.json")), "w") as f:
            json.dump({"result": fid_value}, f)
