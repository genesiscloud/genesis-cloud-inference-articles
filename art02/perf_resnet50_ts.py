
from time import perf_counter
import torch
import torch.nn.functional as F
import torchvision.models as models

# create models

resnet50 = models.resnet50(pretrained=True).cuda()
resnet50_ts = torch.jit.script(resnet50)
input = torch.rand(1, 3, 224, 224).cuda()

resnet50.eval()
resnet50_ts.eval()

# benchmark original model

with torch.no_grad():
    for i in range(1, 10):
        resnet50(input)
    start = perf_counter()
    for i in range(1, 100):
        resnet50(input)
    end = perf_counter()

print('Perf original model {0:.2f} ms'.format(((end - start) / 100) * 1000))

# benchmark TorchScript model

with torch.no_grad():
    for i in range(1, 10):
        resnet50_ts(input)
    start = perf_counter()
    for i in range(1, 100):
        resnet50_ts(input)
    end = perf_counter()

print('Perf TorchScript model {0:.2f} ms'.format(((end - start) / 100) * 1000))

# compare Top-5 results

output = resnet50(input)
output_ts = resnet50_ts(input)

top5 = F.softmax(output, dim=1).topk(5).indices
top5_ts = F.softmax(output_ts, dim=1).topk(5).indices

print('Original model top 5 results:\n {}'.format(top5))
print('TorchScript model top 5 results:\n {}'.format(top5_ts))


