import os

from homr.download_utils import download_file, untar_file, unzip_file

script_location = os.path.dirname(os.path.realpath(__file__))


def download_cvs_musicma() -> str:
    dataset_path = os.path.join(script_location, "CvcMuscima-Distortions")
    if os.path.exists(dataset_path):
        return dataset_path
    print(
        "Downloading Staff Removal set from http://pages.cvc.uab.es/cvcmuscima/index_database.html"
    )
    download_url = "http://datasets.cvc.uab.es/muscima/CVCMUSCIMA_SR.zip"
    download_path = os.path.join(script_location, "CVCMUSCIMA_SR.zip")
    download_file(download_url, download_path)
    unzip_file(download_path, script_location)
    print("Download complete")
    return dataset_path


def download_deep_scores() -> str:
    dataset_path = os.path.join(script_location, "ds2_dense")
    if os.path.exists(dataset_path):
        return dataset_path
    print("Downloading deep DeepScoresV2 Dense from https://zenodo.org/records/4012193")
    download_url = "https://zenodo.org/records/4012193/files/ds2_dense.tar.gz?download=1"
    download_path = os.path.join(script_location, "ds2_dense.tar.gz")
    download_file(download_url, download_path)
    untar_file(download_path, script_location)
    print("Download complete")
    return dataset_path
