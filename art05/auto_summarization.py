
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "sshleifer/distilbart-cnn-12-6"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

article = r"""
The Alaskan Malamute is a large breed of dog that was originally bred for
its strength and endurance to haul heavy freight as a sled dog and hound.
The usual colors are various shades of gray and white, sable and white,
black and white, seal and white, red and white, or solid white.
The physical build of the Malamute is compact and strong with substance,
bone and snowshoe feet. Alaskan Malamutes are still in use as sled dogs
for personal travel, hauling freight, or helping move light objects.
However, most Malamutes today are kept as family pets or as show or performance dogs.
Malamutes are very fond of people, a trait that makes them particularly sought-after
family dogs, but unreliable watchdogs as they do not tend to bark.
"""

inputs = tokenizer(article, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_length=130, min_length=30)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))


