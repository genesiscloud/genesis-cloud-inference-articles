
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

text = "Husky is a general term for a dog used in the polar regions."

input = tokenizer(text)
print(input)

text2 = tokenizer.decode(input["input_ids"])
print(text2)


