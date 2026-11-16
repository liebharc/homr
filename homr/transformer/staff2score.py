import os
from time import perf_counter

import numpy as np
from PIL import Image

from homr.simple_logging import eprint
from homr.transformer.configs import Config
from homr.transformer.decoder_inference import get_decoder
from homr.transformer.encoder_inference import Encoder
from homr.transformer.vocabulary import EncodedSymbol
from homr.type_definitions import NDArray


class Staff2Score:
    """
    Inference class for Tromr. Use predict() for prediction
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.encoder = Encoder(self.config)
        self.decoder = get_decoder(self.config)

        if not os.path.exists(self.config.filepaths.rhythmtokenizer):
            raise RuntimeError(
                "Failed to find tokenizer config" + self.config.filepaths.rhythmtokenizer
            )  # noqa: E501

    def predict(self, image: NDArray, second_image: NDArray | None = None) -> list[EncodedSymbol]:
        """
        Inference an image (NDArray) using Tromr. second_image: another crop of the same
        staff, the decoder reads both together.
        """
        t0 = perf_counter()

        # The encoder and decoder may run in different precisions (e.g. the CoreML
        # encoder is fp16 while the decoder stays on the fp32 CPU model), so cast the
        # context to the dtype the decoder expects before handing it over.
        context_dtype = np.float16 if self.decoder.fp16 else np.float32
        images = [image] if second_image is None else [image, second_image]
        contexts = [
            self.encoder.generate(_transform(image=i)).astype(context_dtype, copy=False)
            for i in images
        ]
        out = self.decoder.generate(contexts)

        eprint(f"Inference Time Tromr: {perf_counter()-t0}")

        return out


class ConvertToArray:
    def __init__(self) -> None:
        self.mean = np.array([0.7931]).reshape(1, 1, 1)
        self.std = np.array([0.1738]).reshape(1, 1, 1)

    def normalize(self, array: NDArray) -> NDArray:
        return (array - self.mean) / self.std

    def __call__(self, image: NDArray) -> NDArray:
        arr = np.array(image) / 255
        arr = arr[np.newaxis, np.newaxis, :, :]
        return self.normalize(arr).astype(np.float32)


_transform = ConvertToArray()


def test_transformer_on_image(path_to_img: str) -> None:
    """
    Tests the transformer on an image and prints the results.
    Args:
        path_to_img(str): Path to the image to test
    """

    model = Staff2Score(Config())
    image = Image.open(path_to_img)
    out = model.predict(np.array(image))
    eprint(out)


if __name__ == "__main__":
    import sys

    test_transformer_on_image(sys.argv[1])
