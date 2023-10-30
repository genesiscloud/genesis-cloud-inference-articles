
import argparse
import logging
from onnxruntime.transformers import optimizer 
from onnxruntime.transformers.fusion_options import FusionOptions
from transformers import AutoConfig

def get_model_size(model_name):
    config = AutoConfig.from_pretrained(model_name)
    num_attention_heads = getattr(config, "num_attention_heads", 0)
    hidden_size = getattr(config, "hidden_size", 0)
    return num_attention_heads, hidden_size

def optimize_onnx(
        input_path,
        output_path,
        num_attention_heads=0,
        hidden_size=0):
    optimization_options = FusionOptions("bert")
    optimization_options.enable_gelu_approximation = False  # additional optimization
    # NOTE: For 'num_heads' and 'hidden_size' automatic detection with 0
    #     may not work with opset 13 or distilbert models
    optimized_model = optimizer.optimize_model(
        input=input_path,
        model_type="bert",
        use_gpu=True,
        opt_level=1,
        num_heads=num_attention_heads,
        hidden_size=hidden_size,
        optimization_options=optimization_options)
    logging.info(f"optimizations applied: {optimized_model.get_fused_operator_statistics()}")
    optimized_model.save_model_to_file(output_path) 

def run(model_name, input_path, output_path):
    num_attention_heads, hidden_size = get_model_size(model_name) 
    optimize_onnx(input_path, output_path, num_attention_heads, hidden_size)

def parse_args(commands=None):
    parser = argparse.ArgumentParser(
        description="optimize transformer models in ONNX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-m", 
        "--model", 
        required=True, 
        help="path to model or URL to Hugging Face hub")
    parser.add_argument(
        "-i", 
        "--input", 
        required=True, 
        help="path to input ONNX file")
    parser.add_argument(
        "-o", 
        "--output", 
        required=True, 
        help="path to output optimized ONNX file")
    args, _ = parser.parse_known_args(args=commands)
    return args

def main():
    args = parse_args()

    model_name = args.model
    input_path = args.input
    output_path = args.output

    run(model_name, input_path, output_path)

if __name__ == "__main__":
    main()

