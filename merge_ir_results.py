import os
import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument('-m', "--metric", type=str, choices=["clip", "image_reward"], default="clip")
    args = parser.parse_args()

    for exp in os.listdir(args.out_dir):
        exp_dir = os.path.join(args.out_dir, exp)
        clip_dir = os.path.join(exp_dir, args.metric)
        os.makedirs(clip_dir, exist_ok=True)
        if os.path.exists(os.path.join(clip_dir, "result.json")):
            continue
        else:
            merged_data = {}
            files = [os.path.join(clip_dir, item) for item in os.listdir(clip_dir)]
            for file in files:
                with open(str(file), "r") as f:
                    data = json.load(f)
                    merged_data.update(data)

            if len(merged_data) != 10000:
                print(f"{clip_dir} has only {len(merged_data)} items")
                continue

            with open(os.path.join(clip_dir, "report.json"), "w") as f:
                json.dump(merged_data, f, indent=4)

            clip_score = sum(v for v in merged_data.values()) / len(merged_data)

            with open(os.path.join(clip_dir, "result.json"), "w") as f:
                json.dump({"result": clip_score}, f)