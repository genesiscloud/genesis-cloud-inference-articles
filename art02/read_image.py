
from torchvision import models, transforms
from PIL import Image

IMG_PATH = "./data/husky01.jpg"
DATA_PATH = "./data/husky01.dat"

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

# convert to numpy array and write to file
data = img.numpy()
data.tofile(DATA_PATH)


