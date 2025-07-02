# type: ignore

import argparse
from pathlib import Path

import safetensors.torch as safetensors
import torch

from homr.simple_logging import eprint
from homr.transformer.configs import Config
from homr.transformer.tromr_arch import TrOMR


def load_checkpoint(path, model):
    if ".safetensors" in path:
        tensors = {}
        with safetensors.safe_open(path, framework="pt", device=0) as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        model.load_state_dict(tensors, strict=False)
    else:
        model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"), strict=False)
    return model


def average_checkpoints(checkpoint_dirs: list[str], config: Config):
    eprint("Loading checkpoints...")
    models = []

    for ckpt_dir in checkpoint_dirs:
        config.filepaths.checkpoint = str(Path(ckpt_dir) / "model.safetensors")  # Or another logic
        model = TrOMR(config)
        load_checkpoint(config.filepaths.checkpoint, model)
        models.append(model)

    # Get the state dict of the first model to initialize averaging
    target_state_dict = models[0].state_dict()
    for key in target_state_dict:
        if target_state_dict[key].dtype == torch.float32:
            for model in models[1:]:
                state_dict = model.state_dict()
                target_state_dict[key].data += state_dict[key].data
            target_state_dict[key].data /= len(models)

    # Save the averaged model
    output_path = "averages.pth"
    torch.save(target_state_dict, output_path)
    eprint(f"Averaged checkpoint saved to {output_path}")


def main():
    # Call example: average_weights.py checkpoint-264900/ checkpoint-262251/ checkpoint-259602/
    parser = argparse.ArgumentParser(description="Average model checkpoints")
    parser.add_argument("checkpoint_dirs", nargs="+", help="List of checkpoint folders")
    args = parser.parse_args()

    config = Config()

    average_checkpoints(args.checkpoint_dirs, config)


if __name__ == "__main__":
    main()
