import torch
import torchvision.models as models

input = torch.rand(1, 3, 224, 224)

model = models.resnet50(pretrained=True)
model.eval()
output = model(input)
torch.onnx.export(
    model, 
    input, 
    "./onnx/resnet50.onnx",
    input_names=["input"],
    output_names=["output"],
    export_params=True)

