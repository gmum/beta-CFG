import json
import hashlib


def read_captions(captions_dir, start: int = 0, stop: int = 25):
    text_list = []
    with open(captions_dir, 'r') as f:
        lines = f.readlines()
        for line in lines:
            stripped_line = line.strip()
            if stripped_line:  # Only add non-empty lines
                text_list.append(stripped_line)
    return text_list[start:stop] # Test for 10k MS-COCO validation


def load_json(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data

def get_md5_of_string(input_string):
    return hashlib.md5(input_string.encode()).hexdigest()