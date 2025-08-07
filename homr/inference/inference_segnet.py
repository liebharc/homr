import numpy as np
from PIL import Image
import onnxruntime as ort
from time import perf_counter
import os
from homr.type_definitions import NDArray
from pathlib import Path


SEGNET_PATH = os.path.join('homr', 'inference', 'segnet.onnx')

class Segnet():
    def __init__(self, model_path):
        self.model = ort.InferenceSession(model_path)
        self.input_name = self.model.get_inputs()[0].name # size: [1, 3, 320, 320]
        self.output_name = self.model.get_outputs()[0].name


    def run(self, input_data):
        out = self.model.run([self.output_name], {self.input_name: input_data})[0]
        out = np.argmax(out, axis=1)
        out = np.squeeze(out, axis=0)
        return out

class ExtractResult:
    def __init__(
        self,
        filename: Path,
        original: NDArray,
        staff: NDArray,
        symbols: NDArray,
        stems_rests: NDArray,
        notehead: NDArray,
        clefs_keys: NDArray,
    ):
        self.filename = filename
        self.original = original
        self.staff = staff
        self.symbols = symbols
        self.stems_rests = stems_rests
        self.notehead = notehead
        self.clefs_keys = clefs_keys


def merge_patches(patches, image_shape: list[int], win_size: int, step_size: int = -1):
    if step_size < 0:
        step_size = win_size // 2

    reconstructed = np.zeros(image_shape, dtype=np.float32)
    weight = np.zeros(image_shape, dtype=np.float32)

    idx = 0
    for iy in range(0, image_shape[0], step_size):
        if iy + win_size > image_shape[0]:
            y = image_shape[0] - win_size
        else:
            y = iy
        for ix in range(0, image_shape[1], step_size):
            if ix + win_size > image_shape[1]:
                x = image_shape[1] - win_size
            else:
                x = ix

            reconstructed[y : y + win_size, x : x + win_size] += patches[idx]
            weight[y : y + win_size, x : x + win_size] += 1
            idx += 1

    # Avoid division by zero
    weight[weight == 0] = 1
    reconstructed /= weight

    return reconstructed.astype(patches[0].dtype)


def inference(image_org: np.ndarray, image_path: str, win_size: int = 320, step_size = -1):
    print('Starting Inference.')
    if step_size < 0:
        step_size = win_size // 2
    
    model = Segnet(SEGNET_PATH)
    data = []
    image = np.transpose(image_org, (2, 0, 1)).astype(np.float32)
    image = np.expand_dims(image, axis=0)

    for y in range(0, image.shape[2], step_size):
        if y + win_size > image.shape[2]:
            y = image.shape[2] - win_size
        for x in range(0, image.shape[3], step_size):
            if x + win_size > image.shape[3]:
                x = image.shape[3] - win_size 
            hop = image[:, :, y : y + win_size, x : x + win_size]
            out = model.run(hop)
            data.append(out)


    data = merge_patches(data, (image_org.shape[0], image_org.shape[1]), win_size, step_size)

    stems_layer = 1
    stems_rests = np.where(data == stems_layer, 1, 0)
    notehead_layer = 2
    notehead = np.where(data == notehead_layer, 1, 0)
    clefs_keys_layer = 3
    clefs_keys = np.where(data == clefs_keys_layer, 1, 0)
    staff_layer = 4
    staff = np.where(data == staff_layer, 1, 0)
    print(staff.shape)
    symbol_layer = 5
    symbols = np.where(data == symbol_layer, 1, 0)

    return ExtractResult(Path(image_path), image_org, staff, symbols, stems_rests, notehead, clefs_keys)

if __name__ == '__main__':
    img = np.array(Image.open('test_segnet.png')).astype(np.float32)
    img_shape = img.shape
    t0 = perf_counter()
    out = inference(img, step_size=320)
    print(perf_counter()-t0)
    out = merge_patches(out, [1855, 3118], 320, 320)
    img = Image.fromarray(out*40)
    img.save('out.png')