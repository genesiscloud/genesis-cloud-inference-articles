
import sys

def get_model_name(s):
    pos = s.rfind("/")
    if pos >= 0:
        s = s[pos+1:]
    pos = s.find(".")
    if pos >= 0:
        s = s[:pos]
    return s

def main():
    if len(sys.argv) < 3:
        sys.exit("Usage: python3 merge_perf <path1> <path2> ...") 

    heads = []
    model_set = set()
    perf_all = {}
    for path in sys.argv[1:]:
        with open(path, "r") as fp:
            head = None
            perf = {}
            lines = fp.readlines()
            for line in lines:
                if not line.startswith("#"):
                    continue
                line = line[1:].strip()
                fields = line.split(";")
                if fields[0] == "head":
                    head = fields[1]
                else:
                    model = get_model_name(fields[0])
                    model_set.add(model)
                    perf[model] = float(fields[1])
            if head is None:
                raise ValueError("Missing head tag in " + path)
            heads.append(head)
            for key, value in perf.items():
                perf_all[head + "#" + key] = value

    line = "Model"
    for head in heads:
        line += ";" + head
    print(line)

    models = sorted(list(model_set))
    for model in models:
        line = model
        for head in heads:
            key = head + "#" + model
            value = "-"
            if key in perf_all:
                value = "{0:.2f}".format(perf_all[key])
            line += ";" + value
        print(line)

main()


