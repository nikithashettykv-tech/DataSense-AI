from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading DataSense AI...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("\nDataSense AI is ready!")
print("Ask questions about Data Science.")
print("Type 'exit' to quit.\n")

while True:
    try:
        prompt = input("You: ").strip()

        if prompt.lower() == "exit":
            print("Goodbye!")
            break

        if not prompt:
            continue

        messages = [
            {
                "role": "system",
                "content": "You are DataSense, a helpful Data Science learning assistant. Explain concepts clearly and simply."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt")

        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7
        )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        print("\nDataSense AI:", response)
        print("\n" + "-" * 60 + "\n")

    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        break