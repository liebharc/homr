import argparse
import sys

from homr.simple_logging import eprint
from training import download
from training.segmentation import train
from training.transformer import train as transformer

parser = argparse.ArgumentParser(description="Train a model")
parser.add_argument("model_name", type=str, help="The name of the model to train")
args = parser.parse_args()

model_type = args.model_name

if model_type == "segnet":
    dataset = download.download_deep_scores()
    train.train_segnet()
elif model_type == "unet":
    dataset = download.download_cvs_musicma()
    train.train_unet()
elif model_type == "transformer":
    transformer.train_transformer()
else:
    eprint("Unknown model: " + model_type)
    sys.exit(1)
