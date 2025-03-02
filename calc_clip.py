import json
import os
import argparse

from tqdm import tqdm
from utils.data_util import load_json
from utils.calculate_metrics import CLIPScore
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

    print("Loading clip model")
    clip_fn = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(
        "cuda:0"
    )
    print("Loading completed")

    for exp in os.listdir(args.output_dir):
        exp_dir = os.path.join(args.output_dir, exp)

        samples_dir = os.path.join(exp_dir, "samples")

        if len(os.listdir(samples_dir)) != 10000:
            print(f"{exp_dir} Found only {len(samples_dir)}. Skipped")

        os.makedirs(os.path.join(exp_dir, "clip"), exist_ok=True)

        if os.path.exists(os.path.join(exp_dir, "clip", f"result.json")):
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
                img = read_image(img_path).to("cuda:0")
                score = clip_fn(img, prompt)
                scores[img_filename] = score.item()
                pbar.update(1)

        with open(os.path.join(exp_dir, "clip", f"{RANK}.json"), "w") as f:
            json.dump(scores, f)

        pbar.close()
