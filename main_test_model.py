#from training.inference_torch.inference_transformer import test_transformer
from training.inference_torch.inference_segnet import test_segnet

# this prints the result from the transformer located in training/architecture/transformer
#test_transformer('test_transformer.png')

# you can use the result (in form of the ExtractResult class) or save it as a 
result = test_segnet('test_img.png', 6, 'out.png')
