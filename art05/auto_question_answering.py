
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

def print_result(tokenizer, questions, input_ids, result):
    start_scores = result.start_logits
    end_scores = result.end_logits

    for question, token_ids, start_score, end_score in zip(questions, input_ids, start_scores, end_scores):
        # Get the most likely location of answer
        start = torch.argmax(start_score)
        end = torch.argmax(end_score) + 1

        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(token_ids[start:end])
        )

        print(f"Question: {question}")
        print(f"Answer: {answer}")

model_name = "distilbert-base-cased-distilled-squad"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

context = r"""
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

questions = [
    "How are Alaskan Malamutes used?",
    "What are the usual colors of Alaskan Malamutes?",
    "Are Alaskan Malamutes reliable watchdogs?"
]

for question in questions:
    inputs = tokenizer(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs)

    print_result(tokenizer, [question], input_ids, outputs)

batch_questions = [(question, context) for question in questions]
inputs = tokenizer(
    batch_questions, 
    add_special_tokens=True, 
    padding="max_length", 
    max_length=256, 
    return_tensors="pt")
input_ids = inputs["input_ids"]

with torch.no_grad():
    outputs = model(**inputs)

print_result(tokenizer, questions, input_ids, outputs)
    

