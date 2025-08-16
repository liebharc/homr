import torch
import os

from homr.segmentation.config import segnet_path_torch
from training.architecture.segmentation.model import create_segnet

from homr.transformer.configs import Config
from training.architecture.transformer.decoder import get_decoder_onnx
from training.architecture.transformer.encoder import get_encoder


class DecoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, rhythms, pitchs, lifts, context):
        result = self.model(rhythms, pitchs, lifts, mask=None, context=context)
        return result


def convert_encoder():
    """
    Converts the encoder to onnx
    """
    config = Config()

    dir_path = os.path.dirname(config.filepaths.checkpoint)
    filename = os.path.splitext(os.path.basename(config.filepaths.checkpoint))[0]
    path_out = os.path.join(dir_path, f"encoder_{filename}.onnx")

    # Get Encoder
    model = get_encoder(config)

    # Load weights
    model.load_state_dict(
                    torch.load(
                                r"encoder_weights.pt",
                                weights_only=True,
                                map_location=torch.device("cpu")
                    ),
                    strict=True)

    # Set eval mode
    model.eval()

    # Prepare input tensor
    input_tensor = torch.randn(1, 1, 128, 1280).float()

    # Export to onnx
    torch.onnx.export(
        model,
        input_tensor,
        path_out,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"])
    
    return path_out

def convert_decoder():
    """
    Converts the decoder to onnx.
    """
    config = Config()
    model = get_decoder_onnx(config)
    model.eval()

    dir_path = os.path.dirname(config.filepaths.checkpoint)
    filename = os.path.splitext(os.path.basename(config.filepaths.checkpoint))[0]
    path_out = os.path.join(dir_path, f"decoder_{filename}.onnx")

    model.load_state_dict(
                        torch.load(r"decoder_weights.pt", weights_only=True, map_location=torch.device("cpu")),
                        strict=True
                        )

    # Using a wrapper model with a custom forward() function
    wrapped_model = DecoderWrapper(model)
    wrapped_model.eval()

    # Create input data
    # Mask is not used since it caused problems with the tensor size
    rhythms = torch.randint(0, config.num_rhythm_tokens, (1, 10)).long()
    pitchs = torch.randint(0, config.num_pitch_tokens, (1, 10)).long()
    lifts = torch.randint(0, config.num_lift_tokens, (1, 10)).long()
    context = torch.randn((1, 641, 312)).float()

    dynamic_axes = {
        "rhythms": {0: "batch_size", 1: "input_seq_len"},
        "pitchs": {0: "batch_size", 1: "input_seq_len"},
        "lifts": {0: "batch_size", 1: "input_seq_len"},
        "context": {0: "batch_size"},
        "out_rhythms": {0: "batch_size", 1: "output_seq_len"},
        "out_pitchs": {0: "batch_size", 1: "output_seq_len"},
        "out_lifts": {0: "batch_size", 1: "output_seq_len"},
    }

    torch.onnx.export(
        wrapped_model,
        (rhythms, pitchs, lifts, context),
        path_out,
        input_names=["rhythms", "pitchs", "lifts", "context"],
        output_names=[
            "out_rhythms",
            "out_pitchs",
            "out_lifts",
        ],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        export_params=True
    )
    return path_out

def convert_segnet():
    """
    Converts the segnet model to onnx.
    """
    model = create_segnet()
    model.load_state_dict(torch.load(segnet_path_torch, weights_only=True), strict=True)
    model.eval()

    # Input dimension is 1x3x320x320
    sample_inputs = torch.randn(1, 3, 320, 320)

    torch.onnx.export(model,
                    sample_inputs,
                    f"{os.path.splitext(segnet_path_torch)[0]}.onnx",
                    opset_version=17,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    # dyamic axes are required for dynamic batch_size
                    dynamic_axes={
                                'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}
                                }
                    )
    return f"{os.path.splitext(segnet_path_torch)[0]}.onnx"