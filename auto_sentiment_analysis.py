
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def infer(model, input):
    with torch.no_grad():
        output = model(**input)
    return torch.softmax(output.logits, dim=1).numpy()

def get_score(result):
    return [round(v * 100, 2) for v in result]

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

input1 = "Border Collies are extremely energetic, acrobatic, and athletic."
input2 = "Border Collies are infamous for chewing holes in walls and furniture, and for destructive scraping and hole digging, due to boredom."

input = tokenizer(input1, return_tensors="pt")
result = infer(model, input)

score = get_score(result[0])
print(f"NEGATIVE: {score[0]}% POSITIVE: {score[1]}%")

input = tokenizer(input2, return_tensors="pt")
result = infer(model, input)

score = get_score(result[0])
print(f"NEGATIVE: {score[0]}% POSITIVE: {score[1]}%")

input = tokenizer([input1, input2], padding="max_length", max_length=32, return_tensors="pt")
result = infer(model, input)

for i, r in enumerate(result):
    score = get_score(r)
    print(f"[{i}] NEGATIVE: {score[0]}% POSITIVE: {score[1]}%")


