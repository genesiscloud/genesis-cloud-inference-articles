import argparse
import torch
from torch.onnx import TrainingMode
from transformers import AutoTokenizer, AutoModel

def generate_input(batch_size, seq_len, include_token_ids):
    shape = (batch_size, seq_len)
    inputs = {}
    inputs["input_ids"] = torch.randint(high=100, size=shape, dtype=torch.long, device="cuda")
    if include_token_ids:
        inputs["token_type_ids"] = torch.ones(size=shape, dtype=torch.long, device="cuda")
    inputs["attention_mask"] = torch.ones(size=shape, dtype=torch.long, device="cuda")
    return inputs

def convert_to_onnx(model, output_path, inputs):
    dynamic_axis = {}
    for k in inputs.keys():
        dynamic_axis[k] = {0: "batch_size", 1: "sequence"}
    dynamic_axis["output"] = {0: "batch_size"}
    with torch.no_grad():
        torch.onnx.export(
            model,
            tuple(inputs.values()),
            output_path,
            verbose=False,               # default
            training=TrainingMode.EVAL,  # default
            input_names=list(inputs.keys()),
            output_names=["output"],
            opset_version=13,
            do_constant_folding=True,    # default
            dynamic_axes=dynamic_axis)


def run(model_name, output_path, batch_size, seq_len):
    assert torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_names = tokenizer.model_input_names
    include_token_ids = "token_type_ids" in input_names 
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.cuda() 
    inputs = generate_input(batch_size, seq_len, include_token_ids)
    convert_to_onnx(model, output_path, inputs)

def parse_args(commands=None):
    parser = argparse.ArgumentParser(
        description="convert transformer models to ONNX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-m", 
        "--model", 
        required=True, 
        help="path to model or URL to Hugging Face hub")
    parser.add_argument(
        "-o", 
        "--output", 
        required=True, 
        help="path to output ONNX file")
    parser.add_argument(
        "-b",
        "--batch-size",
        default=1,
        help="optimal batch size",
        type=int)
    parser.add_argument(
        "-s",
        "--seq-len",
        default=16,
        help="optimal sequence length",
        type=int)
    args, _ = parser.parse_known_args(args=commands)
    return args

def main():
    args = parse_args()

    model_name = args.model
    onnx_path = args.output
    batch_size = args.batch_size
    seq_len = args.seq_len

    run(model_name, onnx_path, batch_size, seq_len)

if __name__ == "__main__":
    main()

