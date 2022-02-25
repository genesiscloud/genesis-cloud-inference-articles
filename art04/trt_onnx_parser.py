import sys
import tensorrt as trt

def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python3 trt_onnx_parser.py <input_onnx_path> <output_plan_path>")

    onnx_path = sys.argv[1]
    plan_path = sys.argv[2]

    logger = trt.Logger()
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.max_workspace_size = 256 * 1024 * 1024
    config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)

    parser = trt.OnnxParser(network, logger)
    ok = parser.parse_from_file(onnx_path)
    if not ok:
        sys.exit("ONNX parse error")

    plan = builder.build_serialized_network(network, config)
    with open(plan_path, "wb") as fp:
        fp.write(plan)

    print("DONE")

main()

