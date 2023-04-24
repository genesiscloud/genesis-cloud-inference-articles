
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

model_name = "distilroberta-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

text = f"The German Shepherd is a breed of working {tokenizer.mask_token} of medium to large size."

inputs = tokenizer(text, return_tensors="pt")
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

with torch.no_grad():
    result = model(**inputs)

token_logits = result.logits
mask_token_logits = token_logits[0, mask_token_index, :]
mask_token_logits = torch.softmax(mask_token_logits, dim=1)

top5 = torch.topk(mask_token_logits, 5, dim=1)
top5_scores = top5.values[0].tolist()
top5_tokens = top5.indices[0].tolist()

for score, token in zip(top5_scores, top5_tokens):
    token_str = tokenizer.decode([token])
    sequence = text.replace(tokenizer.mask_token, token_str)
    print(f"Score: {round(score, 4)} Token: [{token_str}] Sequence: {sequence}")


