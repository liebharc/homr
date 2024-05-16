import json
import os
from typing import Any

from training.segmentation.types import Model


def save_model(model: Model, metadata: dict[str, Any], model_path: str) -> None:
    """Save model and metadata"""
    model.export(model_path)  # Creates a folder with the model, we now add metadata
    write_text_to_file(
        model.to_json(), os.path.join(model_path, "arch.json")
    )  # Save model architecture for documentation
    write_text_to_file(json.dumps(metadata), os.path.join(model_path, "meta.json"))


def load_model(model_path: str) -> tuple[Model, dict[str, Any]]:
    """Load model and metadata"""
    import tensorflow as tf

    model = tf.saved_model.load(model_path)
    with open(os.path.join(model_path, "meta.json")) as f:
        metadata = json.loads(f.read())
    return model, metadata


def write_text_to_file(text: str, path: str) -> None:
    with open(path, "w") as f:
        f.write(text)
