# Main file to convert all the models of homr to onnx format.

from split_weights import split_weights
from convert import convert_segnet, convert_encoder, convert_decoder
from quantization import quantization_int8

if __name__ == "__main__":
    transformer_path = None # Make sure to the filepath of the transformer!
    segnet_path = None # Make sure to the filepath of the segnet!

    if transformer_path is None or segnet_path is None:
        raise FileExistsError('You did not specify the path of your pytorch models')

    split_weights(transformer_path) # Make sure to the filepath of the transformer!

    # Warnings might occur
    convert_segnet(segnet_path) # Make sure to the filepath of the segnet!
    convert_encoder()
    convert_decoder()

    # Only the decoder gets quantized. 
    # The segnet showed 80% worse performance on x86-64. 
    # Only improved size by around 15MB without any speedups (maybe even slowing inference down).
    # FP16 slowed inference speed down (CPU).
    quantization_int8('tromr_decoder.onnx', 'tromr_decoder.onnx')
    print('Finished')