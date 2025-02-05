from transformers import AutoTokenizer, AutoModelForCausalLM

# ✅ Load fine-tuned model
model_path = "fine_tuned_qwen2.5B"  # ✅ No "./"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")


# ✅ Improved text generation
def generate_response(prompt):
    inputs = tokenizer(
        prompt, return_tensors="pt", padding=True, truncation=True, max_length=256
    )
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")  # ✅ Set attention mask

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,  # ✅ Use attention mask
        max_new_tokens=100,
        do_sample=True,  # ✅ Enable sampling for diverse output
        top_k=50,  # ✅ Consider top 50 probable words
        top_p=0.9,  # ✅ Nucleus sampling
        temperature=0.7,  # ✅ Adjust randomness
        repetition_penalty=1.2,  # ✅ Reduce repetition
        pad_token_id=tokenizer.eos_token_id,  # ✅ Prevents padding issues
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)


# ✅ Test improved fine-tuned model
# print(generate_response("What is AI?"))
# print(generate_response("Who is Albert Einstein?"))
# print(generate_response("Explain quantum computing."))
print(generate_response("Who is Muhamad Ariq Azis Alhafits?"))
