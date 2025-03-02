import json
import os
import argparse

import ImageReward as RM

from tqdm import tqdm
from utils.data_util import load_json
from torchvision.io import read_image

if __name__ == "__main__":
    RANK = int(os.environ.get("RANK", "0"))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--prompts", type=str, default="data/val10k.json")
    parser.add_argument("--images_per_prompt", type=int, default=1)
    args = parser.parse_args()

    prompts = load_json(args.prompts)
    prompts = [item["prompt"] for item in prompts.values()]

    print("Loading IR model")
    model = RM.load("ImageReward-v1.0")

    print("Loading completed")

    for exp in os.listdir(args.output_dir):
        exp_dir = os.path.join(args.output_dir, exp)

        samples_dir = os.path.join(exp_dir, "samples")

        if len(os.listdir(samples_dir)) != 10000:
            print(f"{exp_dir} Found only {len(samples_dir)}. Skipped")

        os.makedirs(os.path.join(exp_dir, "image_reward"), exist_ok=True)

        if os.path.exists(os.path.join(exp_dir, "image_reward", f"result.json")):
            continue

        pbar = tqdm(total=args.images_per_prompt*len(prompts))

        scores = {}
        for idx, prompt in enumerate(prompts):
            for i in range(args.images_per_prompt):
                image_id = idx * args.images_per_prompt + i

                if image_id % WORLD_SIZE != RANK:
                    pbar.update(1)
                    continue

                # T2I Comp bench format
                img_filename = f"{image_id:>06}.jpg"
                img_path = os.path.join(exp_dir, "samples", img_filename)
                score = model.score(prompt, img_path)
                scores[img_filename] = score
                pbar.update(1)

        with open(os.path.join(exp_dir, "image_reward", f"{RANK}.json"), "w") as f:
            json.dump(scores, f)

        pbar.close()
