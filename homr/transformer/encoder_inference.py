import onnxruntime as ort

class Encoder():
    def __init__(self, path, use_gpu):
        if use_gpu:
            try:
                self.encoder = ort.InferenceSession(path, providers=['CUDAExecutionProvider'])
            except:
                self.encoder = ort.InferenceSession(path)
    
        else:
            self.encoder = ort.InferenceSession(path)

        self.input_name = self.encoder.get_inputs()[0].name
        self.output_name = self.encoder.get_outputs()[0].name

    def generate(self, x):
        output = self.encoder.run([self.output_name], {self.input_name: x})
        return output[0]