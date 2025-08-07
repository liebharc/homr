import onnxruntime as ort

class Encoder():
    def __init__(self, path):
        self.encoder = ort.InferenceSession(path)
        self.input_name = self.encoder.get_inputs()[0].name
        self.output_name = self.encoder.get_outputs()[0].name

    def generate(self, x):
        output = self.encoder.run([self.output_name], {self.input_name: x})
        print(f"ouput: {output[0].shape}")
        return output[0]