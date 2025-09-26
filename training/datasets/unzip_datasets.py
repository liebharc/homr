# ruff: noqa: T201

import os
import sys
import zipfile

# Setup paths
script_location = os.path.dirname(os.path.realpath(__file__))
git_root = os.path.abspath(os.path.join(script_location, "..", ".."))

# Find all zips in current directory
zip_files = [f for f in os.listdir(".") if f.endswith(".zip")]

if not zip_files:
    print("Error: No zip files found in current directory")
    sys.exit(1)

for zip_path in zip_files:
    print(f"📦 Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            target_path = os.path.join(git_root, member)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with zf.open(member) as src, open(target_path, "wb") as dst:
                dst.write(src.read())
    print(f"✅ Extracted {zip_path}")

print(f"\n🎉 All archives extracted to git root: {git_root}")
