
import torch
import torchvision.models as models

MODELS = [
    ('alexnet', models.alexnet),

    ('densenet121', models.densenet121),
    ('densenet161', models.densenet161),
    ('densenet169', models.densenet169),
    ('densenet201', models.densenet201),

    ('googlenet', models.googlenet),

    ('mnasnet0_5', models.mnasnet0_5),
    ('mnasnet1_0', models.mnasnet1_0),

    ('mobilenet_v2', models.mobilenet_v2),
    ('mobilenet_v3_large', models.mobilenet_v3_large),
    ('mobilenet_v3_small', models.mobilenet_v3_small),

    ('resnet18', models.resnet18),
    ('resnet34', models.resnet34),
    ('resnet50', models.resnet50),
    ('resnet101', models.resnet101),
    ('resnet152', models.resnet152),

    ('resnext50_32x4d', models.resnext50_32x4d),
    ('resnext101_32x8d', models.resnext101_32x8d),

    ('shufflenet_v2_x0_5', models.shufflenet_v2_x0_5),
    ('shufflenet_v2_x1_0', models.shufflenet_v2_x1_0),

    ('squeezenet1_0', models.squeezenet1_0),
    ('squeezenet1_1', models.squeezenet1_1),

    ('vgg11', models.vgg11),
    ('vgg11_bn', models.vgg11_bn),
    ('vgg13', models.vgg13),
    ('vgg13_bn', models.vgg13_bn),
    ('vgg16', models.vgg16),
    ('vgg16_bn', models.vgg16_bn),
    ('vgg19', models.vgg19),
    ('vgg19_bn', models.vgg19_bn),

    ('wide_resnet50_2', models.wide_resnet50_2),
    ('wide_resnet101_2', models.wide_resnet101_2),
]

def generate_model(name, model):
    print('Generate', name)
    m = model(pretrained=True).cuda()
    m_scripted = torch.jit.script(m)
    m_scripted.save('./ts/' + name + '.ts')

for name, model in MODELS:
    generate_model(name, model)

