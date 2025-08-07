from homr.inference.encoder_inference import Encoder
from homr.inference.decoder_inference import get_decoder
import numpy as np
from PIL import Image
from homr.inference.transformer.configs import Config
from time import perf_counter

class Tromr():
    def __init__(self):
        self.config = Config()
        self.encoder = Encoder("tromr_encoder.onnx")
        self.decoder = get_decoder(self.config, "tromr_decoder.onnx")
    
    def run(self, image_path):
        # Load image and convert to correct shape
        data = np.array(Image.open(image_path))
        if len(data.shape) == 3:
            data = data[:, :, 0]
            x = data[np.newaxis, np.newaxis, :, :].astype(np.float32)
        
        else:
            x = data[np.newaxis, np.newaxis, :, :].astype(np.float32)

        t0 = perf_counter()
        start_token = np.full((len(x), 1), self.config.bos_token, dtype=np.int64)
        nonote_token = np.full((len(x), 1), self.config.nonote_token, dtype=np.int64)

        context = self.encoder.generate(x)
        out = self.decoder.generate(start_token, nonote_token, seq_len=self.config.max_seq_len, eos_token=self.config.eos_token, context=context)

        print(perf_counter()-t0)
        return out

if __name__ == '__main__':
    a = Tromr()
    print(a.run('test_tromr.png'))