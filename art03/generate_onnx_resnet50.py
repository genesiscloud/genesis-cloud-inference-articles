
import torch
import torchvision.models as models

input = torch.rand(1, 3, 224, 224)

model = models.resnet50(pretrained=True)
model.eval()
output = model(input)
torch.onnx.export(model, input, "resnet50.onnx", export_params=True)

