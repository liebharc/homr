"""
Implementation of Staff2Score for testing
"""
import numpy as np
from PIL import Image
from training.architecture.transformer.staff2score import Staff2Score
from training.architecture.transformer.configs import Config

def test_transformer(path_to_img):
    model = Staff2Score(Config())
    image = Image.open(path_to_img).resize((1280, 128), Image.LANCZOS)
    out = model.predict(np.array(image))
    print(out)