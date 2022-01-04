
from time import perf_counter
import torch
import torch.nn.functional as F
import torchvision.models as models

# create models

resnet50 = models.resnet50(pretrained=True).cuda()
resnet50_scripted = torch.jit.script(resnet50)
dummy_input = torch.rand(1, 3, 224, 224).cuda()

resnet50.eval()
resnet50_scripted.eval()

# benchmark original model

for i in range(1, 10):
    resnet50(dummy_input)
start = perf_counter()
for i in range(1, 100):
    resnet50(dummy_input)
end = perf_counter()
print('Perf original model {0:.2f} ms'.format(((end - start) / 100) * 1000))

# benchmark TorchScript model

for i in range(1, 10):
    resnet50_scripted(dummy_input)
start = perf_counter()
for i in range(1, 100):
    resnet50_scripted(dummy_input)
end = perf_counter()
print('Perf TorchScript model {0:.2f} ms'.format(((end - start) / 100) * 1000))

# compare Top-5 results

unscripted_output = resnet50(dummy_input)
scripted_output = resnet50_scripted(dummy_input)

unscripted_top5 = F.softmax(unscripted_output, dim=1).topk(5).indices
scripted_top5 = F.softmax(scripted_output, dim=1).topk(5).indices

print('Original model top 5 results:\n {}'.format(unscripted_top5))
print('TorchScript model top 5 results:\n {}'.format(scripted_top5))


