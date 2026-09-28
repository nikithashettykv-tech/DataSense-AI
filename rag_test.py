import json

# Load DataSense knowledge base
with open("dataset.json", "r", encoding="utf-8") as file:
    knowledge_base = json.load(file)

print("DataSense Knowledge Base Loaded!")
print("Number of entries:", len(knowledge_base))

while True:
    question = input("\nYou: ").strip()

    if question.lower() == "exit":
        print("Goodbye!")
        break

    if not question:
        continue

    # Simple keyword matching
    question_words = question.lower().split()

    best_match = None
    best_score = 0

    for item in knowledge_base:
        text = (item["question"] + " " + item["answer"]).lower()

        score = sum(
            1 for word in question_words
            if len(word) > 2 and word in text
        )

        if score > best_score:
            best_score = score
            best_match = item

    if best_match:
        print("\nDataSense Knowledge Base:")
        print(best_match["answer"])
    else:
        print("\nNo matching information found.")