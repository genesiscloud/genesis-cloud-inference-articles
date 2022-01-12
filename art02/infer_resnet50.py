
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

IMG_PATH = "./data/husky01.jpg"

# load the pre-trained model
resnet50 = models.resnet50(pretrained=True)
resnet50.eval()

# specify image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

# import and transform image
img = Image.open(IMG_PATH)
img = transform(img)

# create a batch, run inference
input = torch.unsqueeze(img, 0)

# move the input and model to GPU
if torch.cuda.is_available():
    input = input.to("cuda")
    resnet50.to("cuda")

with torch.no_grad():
    output = resnet50(input)

# apply softmax and get Top-5 results
output = F.softmax(output, dim=1)
top5 = torch.topk(output[0], 5)

# read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# print results
for ind, val in zip(top5.indices, top5.values):
    print("{0} {1:.2f}%".format(categories[ind], val * 100))


