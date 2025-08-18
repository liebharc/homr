import onnx
from onnxsim import simplify


def main(input_path: str, output_path: str | None = None) -> None:
    """
    Simplifies an onnx model to reduce it's size.
    Args:
        input_path(str): Path to the .onnx
        output_path(str): Path where the simplified model should be saved. If None uses input_path
    """
    if output_path is None:
        output_path = input_path

    # Load the ONNX model
    model = onnx.load(input_path)

    # Simplify
    model_simp, check = simplify(model)
    if not check:
        raise ValueError("Simplified ONNX model could not be validated")

    # Save the simplified model
    onnx.save(model_simp, output_path)
