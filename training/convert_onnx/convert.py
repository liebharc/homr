import torch
from transformer.encoder import get_encoder
from transformer.decoder import get_decoder_onnx
from transformer.configs import Config
from segmentation.model import create_segnet
from segmentation.config import segnet_path

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

    # Get Encoder
    model = get_encoder(Config())

    # Load weights
    model.load_state_dict(
                    torch.load(
                        r"encoder_weights.pt", weights_only=True, map_location=torch.device("cpu")
                    ),
                    strict=True)
    
    # Set eval mode
    model.eval()

    # Prepare input tensor
    input = torch.randn(1, 1, 128, 1280).float()

    # Export to onnx 
    torch.onnx.export(
        model,
        input,
        "tromr_encoder.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"])
    
def convert_segnet():
    """
    Converts the segnet model to onnx
    """
    model = create_segnet()
    model.load_state_dict(torch.load(segnet_path, weights_only=True), strict=True)
    model.eval()

    # Input dimension is 1x3x320x320
    sample_inputs = torch.randn(1, 3, 320, 320)

    torch.onnx.export(model, 
                    sample_inputs, 
                    "segnet.onnx",
                    opset_version=17,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output']) 

def convert_decoder():
    """
    Converts the decoder to onnx.
    """
    config = Config()
    model = get_decoder_onnx(config)
    model.eval()

    model.load_state_dict(
                    torch.load(
                        r"decoder_weights.pt", weights_only=True, map_location=torch.device("cpu")), strict=True)

    # Using a wrapper model with a custom forward() function
    wrapped_model = DecoderWrapper(model)
    wrapped_model.eval()

    # Create input data
    # Mask is not used since it caused problems with the tensor size
    rhythms = torch.randint(0, config.num_rhythm_tokens, (1, 10)).long()
    pitchs = torch.randint(0, config.num_pitch_tokens, (1, 10)).long()
    lifts = torch.randint(0, config.num_lift_tokens, (1, 10)).long()
    context = torch.randn((1, 641, 256)).float()

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
        "tromr_decoder.onnx",
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