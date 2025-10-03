import json
import os
from typing import Any

from homr.transformer.vocabulary import Vocabulary

workspace = os.path.join(os.path.dirname(__file__))
root_dir = os.getcwd()


class FilePaths:
    def __init__(self) -> None:
        model_name = "pytorch_model_236-7bb416e48b8165ab161bb63aee88fad66f646c79"
        self.encoder_path = os.path.join(
            workspace,
            f"encoder_{model_name}.onnx",
        )  # noqa: E501
        self.decoder_path = os.path.join(
            workspace,
            f"decoder_{model_name}.onnx",
        )  # noqa: E501
        self.checkpoint = os.path.join(
            root_dir,
            "training",
            "architecture",
            "transformer",
            f"{model_name}.pth",
        )

        self.rhythmtokenizer = os.path.join(workspace, "tokenizer_rhythm.json")
        self.lifttokenizer = os.path.join(workspace, "tokenizer_lift.json")
        self.pitchtokenizer = os.path.join(workspace, "tokenizer_pitch.json")
        self.notetokenizer = os.path.join(workspace, "tokenizer_note.json")

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "rhythmtokenizer": self.rhythmtokenizer,
            "lifttokenizer": self.lifttokenizer,
            "pitchtokenizer": self.pitchtokenizer,
            "notetokenizer": self.notetokenizer,
        }

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class DecoderArgs:
    def __init__(self) -> None:
        self.attn_on_attn = True
        self.cross_attend = True
        self.ff_glu = True
        self.rel_pos_bias = False
        self.use_scalenorm = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "attn_on_attn": self.attn_on_attn,
            "cross_attend": self.cross_attend,
            "ff_glu": self.ff_glu,
            "rel_pos_bias": self.rel_pos_bias,
            "use_scalenorm": self.use_scalenorm,
        }

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class Config:
    def __init__(self) -> None:
        self.vocab = Vocabulary()
        self.filepaths = FilePaths()
        self.channels = 1
        self.patch_size = 16
        self.max_height = 256
        self.max_width = 1280
        self.max_seq_len = 608
        self.pad_token = 0
        self.bos_token = 1
        self.eos_token = 2
        self.nonote_token = 0
        self.num_rhythm_tokens = len(self.vocab.rhythm)
        self.num_pitch_tokens = len(self.vocab.pitch)
        self.num_lift_tokens = len(self.vocab.lift)
        self.num_articulation_tokens = len(self.vocab.articulation)
        self.num_position_tokens = len(self.vocab.position)
        self.num_state_tokens = len(self.vocab.state)
        self.encoder_structure = "hybrid"
        self.encoder_depth = 8
        self.backbone_layers = [3, 4, 6, 3]
        self.encoder_dim = 312
        self.encoder_heads = 8
        self.decoder_dim = 312
        self.decoder_depth = 8
        self.decoder_heads = 8
        self.temperature = 0.01
        self.decoder_args = DecoderArgs()
        self.lift_vocab = self.vocab.lift
        self.pitch_vocab = self.vocab.pitch
        self.rhythm_vocab = self.vocab.rhythm
        self.articulation_vocab = self.vocab.articulation
        self.position_vocab = self.vocab.position
        self.state_vocab = self.vocab.state

    def to_dict(self) -> dict[str, Any]:
        return {
            "filepaths": self.filepaths.to_dict(),
            "channels": self.channels,
            "patch_size": self.patch_size,
            "max_height": self.max_height,
            "max_width": self.max_width,
            "max_seq_len": self.max_seq_len,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "nonote_token": self.nonote_token,
            "encoder_structure": self.encoder_structure,
            "encoder_depth": self.encoder_depth,
            "backbone_layers": self.backbone_layers,
            "encoder_dim": self.encoder_dim,
            "encoder_heads": self.encoder_heads,
            "num_rhythm_tokens": self.num_rhythm_tokens,
            "decoder_dim": self.decoder_dim,
            "decoder_depth": self.decoder_depth,
            "decoder_heads": self.decoder_heads,
            "temperature": self.temperature,
            "decoder_args": self.decoder_args.to_dict(),
        }

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# Initialize the Config class
default_config = Config()
