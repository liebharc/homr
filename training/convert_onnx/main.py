# Main file to convert all the models of homr to onnx format.
import os

from training.convert_onnx.convert import convert_decoder, convert_encoder, convert_segnet
from training.convert_onnx.quantization import quantization_int8
from training.convert_onnx.split_weights import split_weights

def convert_all(transformer_path=None, segnet_path=None):
    if transformer_path is None and segnet_path is None:
        raise FileExistsError('You did not specify the path of your pytorch models')

    # Warnings might occur
    if segnet_path is not None:
       convert_segnet() # Make sure to the filepath of the segnet!
    
    if transformer_path is not None:
        split_weights(transformer_path) # Make sure to the filepath of the transformer!
        convert_encoder()
        path_to_decoder = convert_decoder()

        # Only the decoder gets quantized.
        # The segnet showed 80% worse performance on x86-64.
        # Only improved size by around 15MB without any speedups (maybe even slowing inference down).
        # FP16 slowed inference speed down (CPU).
        #quantization_int8(path_to_decoder)

    os.remove('decoder_weights.pt')
    os.remove('encoder_weights.pt')