import sys
import tensorrt as trt

MODELS = [
    'alexnet',

    'densenet121',
    'densenet161',
    'densenet169',
    'densenet201',

    'mnasnet0_5',
    'mnasnet1_0',

    'mobilenet_v2',
    'mobilenet_v3_large',
    'mobilenet_v3_small',

    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',

    'resnext50_32x4d',
    'resnext101_32x8d',

    'shufflenet_v2_x0_5',
    'shufflenet_v2_x1_0',

    'squeezenet1_0',
    'squeezenet1_1',

    'vgg11',
    'vgg11_bn',
    'vgg13',
    'vgg13_bn',
    'vgg16',
    'vgg16_bn',
    'vgg19',
    'vgg19_bn',

    'wide_resnet50_2',
    'wide_resnet101_2',
]


def setup_builder():
    logger = trt.Logger()
    builder = trt.Builder(logger)
    return (logger, builder)

def generate_plan(logger, builder, name):
    print('Generate TensorRT plan for ' + name)

    onnx_path = './onnx/' + name + '.onnx'
    plan_path = './plan/' + name + '.plan'

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.max_workspace_size = 256 * 1024 * 1024
    config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)

    parser = trt.OnnxParser(network, logger)
    ok = parser.parse_from_file(onnx_path)
    if not ok:
        sys.exit('ONNX parse error')

    plan = builder.build_serialized_network(network, config)
    with open(plan_path, "wb") as fp:
        fp.write(plan)

def main():
    logger, builder = setup_builder()
    for name in MODELS:
        generate_plan(logger, builder, name)
    print('DONE')

main()


