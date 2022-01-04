
import torch
import torchvision.models as models

def generate_model(name, model):
    print('Generate', name)
    m = model(pretrained=True).cuda()
    m_scripted = torch.jit.script(m)
    m_scripted.save('./ts/' + name + '.ts')

generate_model('resnet50', models.resnet50)


