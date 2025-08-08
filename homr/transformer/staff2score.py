import os
import cv2
from homr.transformer.encoder_inference import Encoder
from homr.transformer.decoder_inference import get_decoder
import numpy as np
from homr.transformer.configs import Config
from time import perf_counter


class Staff2Score():
    """
    Inference class for Tromr. Use predict() for prediction
    """
    def __init__(self, use_gpu: bool = True) -> None:
        self.config = Config()
        self.encoder = Encoder(self.config.filepaths.encoder_path, use_gpu)
        self.decoder = get_decoder(self.config, self.config.filepaths.decoder_path, use_gpu)

        if not os.path.exists(self.config.filepaths.rhythmtokenizer):
            raise RuntimeError("Failed to find tokenizer config" + self.config.filepaths.rhythmtokenizer)

    def predict(self, image: np.ndarray) -> list[str]:
        """
        Inference an image (np.ndarray) using Tromr.
        """
        data = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        if len(data.shape) == 3:
            data = data[:, :, 0]
            x = data[np.newaxis, np.newaxis, :, :].astype(np.float32)
    
        else:
            x = data[np.newaxis, np.newaxis, :, :].astype(np.float32)


        t0 = perf_counter()

        # Create special tokens
        start_token = np.full((len(x), 1), self.config.bos_token, dtype=np.int64)
        nonote_token = np.full((len(x), 1), self.config.nonote_token, dtype=np.int64)

        # Generate context with encoder
        context = self.encoder.generate(x)

        # Make a prediction using decoder
        out = self.decoder.generate(start_token, nonote_token, seq_len=self.config.max_seq_len, eos_token=self.config.eos_token, context=context)

        print(f"Inference Time: {perf_counter()-t0}")

        return out