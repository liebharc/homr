import numpy as np
import onnxruntime as ort
from onnxruntime import OrtValue

from homr.simple_logging import eprint
from homr.transformer.configs import Config
from homr.type_definitions import NDArray


class Encoder:
    def __init__(self, config: Config) -> None:
        self.use_gpu = False
        if config.use_gpu_inference:
            try:
                self.encoder = ort.InferenceSession(
                    config.filepaths.encoder_path_fp16,
                    providers=[
                        (
                            "CUDAExecutionProvider",
                            {
                                "cudnn_conv_algo_search": "DEFAULT",
                            },
                        )
                    ],
                )
                self.fp16 = True
                self.use_gpu = True

            except Exception as ex:
                eprint(ex)
                eprint("Going on without GPU support")
                self.encoder = ort.InferenceSession(config.filepaths.encoder_path_fp16)
                self.fp16 = True

        else:
            self.encoder = ort.InferenceSession(config.filepaths.encoder_path)
            self.fp16 = False

        self.io_binding = self.encoder.io_binding()
        self.device_id = 0

        self.input_name = self.encoder.get_inputs()[0].name
        self.output_name = self.encoder.get_outputs()[0].name

    def generate(self, x: NDArray) -> list[OrtValue]:
        if self.fp16:
            self.io_binding.bind_cpu_input("input", x.astype(np.float16))
        else:
            self.io_binding.bind_cpu_input("input", x.astype(np.float32))

        self.io_binding.bind_output("output", "cuda" if self.use_gpu else "cpu", self.device_id)
        self.encoder.run_with_iobinding(self.io_binding)
        return self.io_binding.get_outputs()[0].numpy()
