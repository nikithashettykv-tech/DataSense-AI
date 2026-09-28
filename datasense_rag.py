import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# Load DataSense Knowledge Base
# -----------------------------

with open("dataset.json", "r", encoding="utf-8") as file:
    knowledge_base = json.load(file)

print("Knowledge base loaded:", len(knowledge_base), "entries")


# -----------------------------
# Load Qwen Model
# -----------------------------

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading DataSense AI...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("\nDataSense AI is ready!")
print("Ask Data Science questions.")
print("Type 'exit' to quit.\n")


# -----------------------------
# Simple Knowledge Retrieval
# -----------------------------

def retrieve_knowledge(question):

    question_words = question.lower().split()

    best_match = None
    best_score = 0

    for item in knowledge_base:

        text = (
            item["question"] + " " +
            item["answer"]
        ).lower()

        score = sum(
            1
            for word in question_words
            if len(word) > 2 and word in text
        )

        if score > best_score:
            best_score = score
            best_match = item

    return best_match


# -----------------------------
# DataSense Chat
# -----------------------------

while True:

    question = input("You: ").strip()

    if question.lower() == "exit":
        print("Goodbye!")
        break

    if not question:
        continue

    # Retrieve relevant knowledge
    retrieved = retrieve_knowledge(question)

    if retrieved:

        context = retrieved["answer"]

    else:

        context = "No specific information was found in the DataSense knowledge base."


    # -----------------------------
    # Create Prompt for Qwen
    # -----------------------------

    messages = [
        {
            "role": "system",
            "content": (
                "You are DataSense, a Data Science learning assistant. "
                "Use the provided knowledge base information to answer "
                "the user's question accurately and clearly. "
                "Do not invent facts that contradict the knowledge base."
            )
        },
        {
            "role": "user",
            "content": (
                f"Knowledge Base Information:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Give a clear and simple answer suitable for a college student."
            )
        }
    ]


    # Convert to model input
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )


    # Generate answer
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.5
    )


    # Extract response
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )


    print("\nDataSense AI:")
    print(response)
    print("\n" + "-" * 60 + "\n")