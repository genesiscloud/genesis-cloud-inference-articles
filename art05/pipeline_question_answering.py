
from transformers import pipeline

def print_result(result):
    answer = result["answer"]
    score = round(result["score"], 4)
    start = result["start"]
    end = result["end"]
    print(f"Answer: {answer}, score: {score}, start: {start}, end: {end}")

qa = pipeline("question-answering")

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

question1 = "How are Alaskan Malamutes used?"
result = qa(question=question1, context=context)
print_result(result)

question2 = "What are the usual colors of Alaskan Malamutes?"
result = qa(question=question2, context=context)
print_result(result)

question3 = "Are Alaskan Malamutes reliable watchdogs?"
result = qa(question=question3, context=context)
print_result(result)

result = qa(question=[question1, question2, question3], context=context)
for i, r in enumerate(result):
    print(f"[Result {i}]")
    print_result(r)

