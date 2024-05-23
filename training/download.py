import os
from pathlib import Path

from homr.download_utils import download_file, untar_file, unzip_file
from homr.simple_logging import eprint

script_location = os.path.dirname(os.path.realpath(__file__))
git_root = Path(script_location).parent.absolute()
dataset_root = os.path.join(git_root, "datasets")


def download_cvs_musicma() -> str:
    dataset_path = os.path.join(dataset_root, "CvcMuscima-Distortions")
    if os.path.exists(dataset_path):
        return dataset_path
    eprint(
        "Downloading Staff Removal set from http://pages.cvc.uab.es/cvcmuscima/index_database.html"
    )
    download_url = "http://datasets.cvc.uab.es/muscima/CVCMUSCIMA_SR.zip"
    download_path = os.path.join(dataset_root, "CVCMUSCIMA_SR.zip")
    download_file(download_url, download_path)
    eprint("Extracting download")
    unzip_file(download_path, script_location)
    eprint("Download complete")
    return dataset_path


def download_deep_scores() -> str:
    dataset_path = os.path.join(dataset_root, "ds2_dense")
    if os.path.exists(dataset_path):
        return dataset_path
    eprint("Downloading deep DeepScoresV2 Dense from https://zenodo.org/records/4012193")
    download_url = "https://zenodo.org/records/4012193/files/ds2_dense.tar.gz?download=1"
    download_path = os.path.join(dataset_root, "ds2_dense.tar.gz")
    download_file(download_url, download_path)
    eprint("Extracting download")
    untar_file(download_path, script_location)
    eprint("Download complete")
    return dataset_path
