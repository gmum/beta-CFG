import json
import os
import pandas as pd
from tqdm import tqdm

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="out")
    args = parser.parse_args()

    results = []

    for exp in tqdm(os.listdir(args.output_dir)):
        exp_dir = os.path.join(args.output_dir, exp)

        exp_sub_dirs = [f for f in os.listdir(exp_dir) if f not in ['config.json', 'samples' , 'tmp_results_clip']]

        metrics = dict()
        metrics['name'] = exp

        config_path = os.path.join(exp_dir, 'config.json')

        if not os.path.exists(config_path):
            print(f"Config {config_path} not exist")

        with open(config_path, 'r') as f:
            config = json.load(f)
            metrics.update(config)

        for exp_metric_dir in exp_sub_dirs:
            metrics_result_file = os.path.join(exp_dir, exp_metric_dir, 'result.json')

            if not os.path.exists(metrics_result_file):
                print(f"{metrics_result_file} not exist")
                metrics[exp_metric_dir] = None
                continue

            with open(metrics_result_file, 'r') as f:
                metrics[exp_metric_dir] = json.load(f)['result']
        results.append(metrics)

    df = pd.DataFrame(results)


    # print(df)
    df.to_csv('table.csv')





