
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

input1 = "Border Collies are extremely energetic, acrobatic, and athletic."
result = classifier(input1)[0]
print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

input2 = "Border Collies are infamous for chewing holes in walls and furniture, and for destructive scraping and hole digging, due to boredom."
result = classifier(input2)[0]
print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

output = classifier([input1, input2])
for i, result in enumerate(output):
    print(f"[{i}] label: {result['label']}, with score: {round(result['score'], 4)}")


