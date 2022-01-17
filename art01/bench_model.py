
import sys
from time import perf_counter
import torch
import torch.nn.functional as F
import torchvision.models as models

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 bench_model <model_name>")
 
    name = sys.argv[1]
    print('Start ' + name)

    # create model

    builder = getattr(models, name)
    model = builder(pretrained=True).cuda()
    model.eval()

    # create dummy input

    input = torch.rand(1, 3, 224, 224).cuda()

    # benchmark model

    with torch.no_grad():
        for i in range(1, 10):
            model(input)

    start = perf_counter()
    with torch.no_grad():
        for i in range(1, 100):
            model(input)
    end = perf_counter()

    elapsed = ((end - start) / 100) * 1000
    print('Model {0}: elapsed time {1:.2f} ms'.format(name, elapsed))
    # record for automated extraction
    print('#{0};{1:f}'.format(name, elapsed)) 

    # print Top-5 results

    output = model(input)
    top5 = F.softmax(output, dim=1).topk(5)
    top5p = top5.indices.detach().cpu().numpy()
    top5v = top5.values.detach().cpu().numpy()

    print("Top-5 results")
    for ind, val in zip(top5p[0], top5v[0]):
        print("  {0} {1:.2f}%".format(ind, val * 100)) 

main()


