# ruff: noqa: T201

import os
import sys
import zipfile

from training.datasets.convert_grandstaff import grandstaff_train_index
from training.datasets.convert_lieder import lieder_train_index
from training.datasets.convert_primus import primus_train_index

# Setup paths
script_location = os.path.dirname(os.path.realpath(__file__))
git_root = os.path.abspath(os.path.join(script_location, "..", ".."))

# Index CSV files
index_files = [primus_train_index, grandstaff_train_index, lieder_train_index]

for index_file in index_files:
    index_path = os.path.join(git_root, index_file)
    if not os.path.exists(index_path):
        print(f"Error: Index file missing -> {index_path}")
        sys.exit(1)

    # Collect files for this dataset
    dataset_files = [index_path]  # include the index file itself

    with open(index_path, newline="") as csvfile:
        lines = csvfile.readlines()
        for line in lines:
            for cell in line.split(","):
                rel_path = cell.strip()
                abs_path = os.path.join(git_root, rel_path)

                if not os.path.exists(abs_path):
                    print(f"Error: Missing file -> {abs_path}")
                    sys.exit(1)

                dataset_files.append(abs_path)

    # Name the zip after the folder containing the index file
    dataset_name = os.path.basename(os.path.dirname(index_path))
    zip_path = os.path.join(f"{dataset_name}.zip")

    # Create normal zip (no password, standard deflate compression)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fpath in dataset_files:
            arcname = os.path.relpath(fpath, git_root)
            zf.write(fpath, arcname)

    print(f"✅ Created zip file: {zip_path}")
