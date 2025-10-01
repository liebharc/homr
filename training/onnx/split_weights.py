import os
from typing import Any

import torch

from homr.transformer.configs import FilePaths
from training.architecture.transformer.staff2score import load_model_weights


def split_weights(input_path: str) -> None:
    """
    Splits weights between Encoder and Decoder
    Args:
        input_path(str): Path to complete weights
    """
    # Load model
    state_dict = load_model_weights(input_path)

    encoder_state_dict = {}
    decoder_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("encoder."):
            # Remove encoder starting
            new_key = key.replace("encoder.", "")
            # And add it to a seperate dict
            encoder_state_dict[new_key] = value

        elif key.startswith("decoder."):
            # Remove decoder starting
            new_key = key.replace("decoder.", "")
            # And add it to a seperate dict
            decoder_state_dict[new_key] = value
        else:
            raise ValueError("Unknown key " + key)

    # Encoder weights can be saved directly
    torch.save(encoder_state_dict, "encoder_weights.pt")

    # Decoder weights need to be processed once more
    torch.save(remove_score_decoder_weights(decoder_state_dict), "decoder_weights.pt")


def remove_score_decoder_weights(full_state_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Since get_decoder_onnx() is not using ScoreDecoder() we need to change the weights
    so they are directly put into ScoreTransformerWrapper().

    Args:
        full_state_dict (dict): Model dictionary
    """
    # Remove the net. starting
    transformer_state_dict = {}
    for key, value in full_state_dict.items():
        if key.startswith("net."):
            new_key = key.replace("net.", "")
            transformer_state_dict[new_key] = value

    return transformer_state_dict


if __name__ == "__main__":
    split_weights(os.path.join("transformer", FilePaths().checkpoint))
