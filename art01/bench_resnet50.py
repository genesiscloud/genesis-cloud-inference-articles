
from time import perf_counter
import torch
import torch.nn.functional as F
import torchvision.models as models

name = 'resnet50'
print('Start ' + name)

# create model

name = 'resnet50'
resnet50 = models.resnet50(pretrained=True).cuda()
resnet50.eval()

# create dummy input

input = torch.rand(1, 3, 224, 224).cuda()

# benchmark model

with torch.no_grad():
    for i in range(1, 10):
        resnet50(input)

start = perf_counter()
with torch.no_grad():
    for i in range(1, 100):
        resnet50(input)
end = perf_counter()

elapsed = ((end - start) / 100) * 1000
print('Model {0}: elapsed time {1:.2f} ms'.format(name, elapsed))

# print Top-5 results

output = resnet50(input)
top5 = F.softmax(output, dim=1).topk(5).indices
print('Top 5 results:\n {}'.format(top5))


