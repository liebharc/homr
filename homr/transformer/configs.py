import json
import os
from typing import Any

workspace = os.path.join(os.path.dirname(__file__))


class FilePaths:
    def __init__(self) -> None:
        self.checkpoint = os.path.join(
            workspace, "pytorch_model_101-ba12ebef4606948816a06f4a011248d07a6f06da.pth"
        )
        self.rhythmtokenizer = os.path.join(workspace, "tokenizer_rhythm.json")
        self.lifttokenizer = os.path.join(workspace, "tokenizer_lift.json")
        self.pitchtokenizer = os.path.join(workspace, "tokenizer_pitch.json")
        self.rhythmtokenizer = os.path.join(workspace, "tokenizer_rhythm.json")
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
        self.filepaths = FilePaths()
        self.channels = 1
        self.patch_size = 16
        self.max_height = 128
        self.max_width = 1280
        self.max_seq_len = 256
        self.pad_token = 0
        self.bos_token = 1
        self.eos_token = 2
        self.nonote_token = 0
        self.num_rhythm_tokens = 89
        self.num_note_tokens = 2
        self.num_pitch_tokens = 71
        self.num_lift_tokens = 5
        self.lift_null = 0
        self.lift_sharp = 2
        self.lift_flat = 3
        self.encoder_structure = "hybrid"
        self.encoder_depth = 8
        self.backbone_layers = [2, 3, 7]
        self.encoder_dim = 256
        self.encoder_heads = 8
        self.decoder_dim = 256
        self.decoder_depth = 8
        self.decoder_heads = 8
        self.temperature = 0.01
        self.decoder_args = DecoderArgs()
        self.lift_vocab = json.load(open(self.filepaths.lifttokenizer))["model"]["vocab"]
        self.pitch_vocab = json.load(open(self.filepaths.pitchtokenizer))["model"]["vocab"]
        self.note_vocab = json.load(open(self.filepaths.notetokenizer))["model"]["vocab"]
        self.rhythm_vocab = json.load(open(self.filepaths.rhythmtokenizer))["model"]["vocab"]
        self.noteindexes = self._get_values_of_keys_starting_with("note-")
        self.restindexes = self._get_values_of_keys_starting_with(
            "rest-"
        ) + self._get_values_of_keys_starting_with("multirest-")
        self.chordindex = self.rhythm_vocab["|"]
        self.barlineindex = self.rhythm_vocab["barline"]

    def _get_values_of_keys_starting_with(self, prefix: str) -> list[int]:
        return [value for key, value in self.rhythm_vocab.items() if key.startswith(prefix)]

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
            "noteindexes": self.noteindexes,
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
