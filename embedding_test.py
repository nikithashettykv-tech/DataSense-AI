import json
from sentence_transformers import SentenceTransformer, util

# Load dataset
with open("dataset.json", "r", encoding="utf-8") as file:
    knowledge_base = json.load(file)

print("Dataset loaded:", len(knowledge_base), "entries")

# Load embedding model
print("Loading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded!")

# Create text from questions and answers
texts = [
    item["question"] + " " + item["answer"]
    for item in knowledge_base
]

# Convert dataset into embeddings
print("Creating embeddings...")

knowledge_embeddings = embedding_model.encode(
    texts,
    convert_to_tensor=True
)

print("Embeddings created!")


# Search function
def search_knowledge(question):

    question_embedding = embedding_model.encode(
        question,
        convert_to_tensor=True
    )

    similarities = util.cos_sim(
        question_embedding,
        knowledge_embeddings
    )[0]

    best_index = similarities.argmax().item()
    best_score = similarities[best_index].item()

    return knowledge_base[best_index], best_score


# Test chatbot
while True:

    question = input("\nYou: ").strip()

    if question.lower() == "exit":
        print("Goodbye!")
        break

    result, score = search_knowledge(question)

    print("\nBest Match:")
    print(result["question"])

    print("\nSimilarity Score:")
    print(round(score, 3))

    print("\nAnswer:")
    print(result["answer"])