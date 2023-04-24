
from transformers import pipeline

unmasker = pipeline("fill-mask")

text = f"The German Shepherd is a breed of working {unmasker.tokenizer.mask_token} of medium to large size."

result = unmasker(text)

for i, r in enumerate(result):
    score = round(r["score"], 4)
    token = r["token_str"]
    sequence = r["sequence"]
    print(f"[{i}] Score: {score} Token: [{token}] Sequence: {sequence}")


