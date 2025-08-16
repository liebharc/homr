import os
from time import perf_counter

import cv2
import numpy as np

from homr.simple_logging import eprint
from homr.transformer.configs import Config
from homr.transformer.decoder_inference import get_decoder
from homr.transformer.encoder_inference import Encoder
from homr.type_definitions import NDArray

class Staff2Score:
    """
    Inference class for Tromr. Use predict() for prediction
    """
    def __init__(self, use_gpu: bool = True) -> None:
        self.config = Config()
        self.encoder = Encoder(self.config.filepaths.encoder_path, use_gpu)
        self.decoder = get_decoder(self.config, self.config.filepaths.decoder_path, use_gpu)

        if not os.path.exists(self.config.filepaths.rhythmtokenizer):
            raise RuntimeError("Failed to find tokenizer config" + self.config.filepaths.rhythmtokenizer) # noqa: E501

    def predict(self, image: np.ndarray) -> list[str]:
        """
        Inference an image (np.ndarray) using Tromr.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x = _transform(image=image)

        t0 = perf_counter()

        # Create special tokens
        start_token = np.full((len(x), 1), self.config.bos_token, dtype=np.int64)
        nonote_token = np.full((len(x), 1), self.config.nonote_token, dtype=np.int64)

        # Generate context with encoder
        context = self.encoder.generate(x)

        # Make a prediction using decoder
        out = self.decoder.generate(start_token,
                                    nonote_token,
                                    seq_len=self.config.max_seq_len,
                                    eos_token=self.config.eos_token,
                                    context=context
                                    )

        eprint(f"Inference Time Tromr: {perf_counter()-t0}")

        return out

class ConvertToArray:
    def __init__(self):
        self.mean = np.array([0.7931]).reshape(1, 1, 1)
        self.std = np.array([0.1738]).reshape(1, 1, 1)

    def normalize(self, array: NDArray):
        return (array - self.mean) / self.std

    def __call__(self, image: NDArray):
        arr = np.array(image) / 255
        arr = arr[np.newaxis, np.newaxis, :, :]
        return self.normalize(arr).astype(np.float32)

_transform = ConvertToArray()
