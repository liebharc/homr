import argparse
import os
import sys

import tensorflow as tf

from homr.simple_logging import eprint
from training import download
from training.run_id import get_run_id
from training.segmentation import train
from training.segmentation.model_utils import save_model
from training.transformer.train import train_transformer


def get_segmentation_model_path(model_name: str) -> str:
    model_path = os.path.join(script_location, "..", "homr", "segmentation")
    run_id = get_run_id()
    return os.path.join(model_path, f"{model_name}_{run_id}")


script_location = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description="Train a model")
parser.add_argument("model_name", type=str, help="The name of the model to train")
parser.add_argument(
    "--fp32",
    action="store_true",
    help="Only applicable for the transformer: Trains with fp32 accuracy",
)
args = parser.parse_args()

model_type = args.model_name

if model_type == "segnet":
    dataset = download.download_deep_scores()
    model = train.train_model(dataset, data_model=model_type, steps=1500, epochs=15)
    filename = get_segmentation_model_path(model_type)
    meta = {
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
    }
    save_model(model, meta, filename)
    eprint("Model saved as " + filename)
elif model_type == "unet":
    dataset = download.download_cvs_musicma()
    model = train.train_model(dataset, data_model=model_type, steps=1500, epochs=10)
    filename = get_segmentation_model_path(model_type)
    meta = {
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
    }
    save_model(model, meta, filename)
    eprint("Model saved as " + filename)
elif model_type in ["unet_from_checkpoint", "segnet_from_checkpoint"]:
    model = tf.keras.models.load_model(
        "seg_unet.keras", custom_objects={"WarmUpLearningRate": train.WarmUpLearningRate}
    )
    model_name = model_type.split("_")[0]
    filename = get_segmentation_model_path(model_name)
    meta = {
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
    }
    save_model(model, meta, filename)
    eprint("Model saved as " + filename)
elif model_type == "transformer":
    train_transformer(fp32=args.fp32)
else:
    eprint("Unknown model: " + model_type)
    sys.exit(1)
