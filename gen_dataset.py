import argparse
parser = argparse.ArgumentParser(prog="gen_dataset")
parser.add_argument("input_dir", help="folder, where classes are stored")
parser.add_argument("output_dir", help="output folder")
args = parser.parse_args()

import math
from shutil import copyfile

from tqdm import tqdm
import pandas as pd
import os

input_dir = args.input_dir
output_dir = args.output_dir

labels = [f for f in os.listdir(input_dir)]

malignant = ['MEL', 'BCC', 'SCC']

p = 0.2
# split_size = 5000

os.makedirs(f"{output_dir}/train/malignant")
os.makedirs(f"{output_dir}/test/benign")
os.makedirs(f"{output_dir}/train/benign")
os.makedirs(f"{output_dir}/test/malignant")

for label in labels:
    files = [f for f in os.listdir(f"{input_dir}/{label}")]
    test_size = math.ceil(len(files) * p)

    train_files = files[:-test_size]
    test_files = files[-test_size:]

    for train_file in train_files:
        src = f"{input_dir}/{label}/{train_file}"

        if label in malignant:
            dst = f"{output_dir}/train/malignant/{train_file}"
        else:
            dst = f"{output_dir}/train/benign/{train_file}"

        copyfile(src, dst)

    for test_file in test_files:
        src = f"{input_dir}/{label}/{test_file}"

        if label in malignant:
            dst = f"{output_dir}/test/malignant/{test_file}"
        else:
            dst = f"{output_dir}/test/benign/{test_file}"

        copyfile(src, dst)

