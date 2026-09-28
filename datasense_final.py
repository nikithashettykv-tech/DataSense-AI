import json
from sentence_transformers import SentenceTransformer, util

print("Loading DataSense knowledge base...")

with open("dataset.json", "r", encoding="utf-8") as file:
    knowledge_base = json.load(file)

print(f"Knowledge base loaded: {len(knowledge_base)} entries")

print("Loading semantic search model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Semantic search model loaded!")

texts = [
    item["question"] + " " + item["answer"]
    for item in knowledge_base
]

knowledge_embeddings = embedding_model.encode(
    texts,
    convert_to_tensor=True
)

print("Knowledge embeddings created!")


def retrieve_knowledge(question, top_k=3):

    question_embedding = embedding_model.encode(
        question,
        convert_to_tensor=True
    )

    similarities = util.cos_sim(
        question_embedding,
        knowledge_embeddings
    )[0]

    top_results = similarities.topk(top_k)

    results = []

    for i in range(top_k):

        index = top_results.indices[i].item()
        score = top_results.values[i].item()

        results.append(
            {
                "question": knowledge_base[index]["question"],
                "answer": knowledge_base[index]["answer"],
                "score": score
            }
        )

    return results


print("\n" + "=" * 60)
print("                 DATASENSE AI")
print("=" * 60)

print("Top-K RAG Data Science Learning Assistant")

print("\nAsk questions about:")
print("Python | SQL | Statistics | Machine Learning")
print("Deep Learning | NLP | Generative AI | Data Visualization")

print("\nType 'exit' to quit.")

print("=" * 60)


while True:

    question = input("\nYou: ").strip()

    if question.lower() == "exit":
        print("\nGoodbye! 👋")
        break

    if not question:
        continue

    results = retrieve_knowledge(question, top_k=3)

    print("\nDataSense AI:")

    if results[0]["score"] >= 0.60:

        relevant_results = [
            result
            for result in results
            if result["score"] >= 0.60
        ]

        for i, result in enumerate(relevant_results, start=1):

            print(f"\n{i}. {result['answer']}")

    else:

        print(
            "Sorry, I could not find enough relevant information "
            "in the current DataSense knowledge base."
        )

    print("\n" + "-" * 60)

    print("Top Retrieved Knowledge:")

    for i, result in enumerate(results, start=1):

        print(
            f"{i}. {result['question']} "
            f"(Score: {result['score']:.3f})"
        )

    print("-" * 60)