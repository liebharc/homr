import numpy as np
import onnxruntime as ort

from homr.transformer.configs import Config
from homr.type_definitions import NDArray


class Encoder:
    def __init__(self, config: Config) -> None:
        if "CUDAExecutionProvider" in ort.get_available_providers():
            try:
                self.encoder = ort.InferenceSession(
                    config.filepaths.encoder_path_fp16, providers=["CUDAExecutionProvider"]
                )
                self.fp16 = True
            except Exception:
                self.encoder = ort.InferenceSession(config.filepaths.encoder_path_fp16)
                self.fp16 = True

        else:
            self.encoder = ort.InferenceSession(config.filepaths.encoder_path)
            self.fp16 = False

        self.input_name = self.encoder.get_inputs()[0].name
        self.output_name = self.encoder.get_outputs()[0].name

    def generate(self, x: NDArray) -> NDArray:
        if self.fp16:
            output = self.encoder.run([self.output_name], {self.input_name: x.astype(np.float16)})
        else:
            output = self.encoder.run([self.output_name], {self.input_name: x})
        return output[0]
