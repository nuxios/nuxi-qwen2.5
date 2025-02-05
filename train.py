from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# Load dataset
dataset = load_dataset("json", data_files="dataset.json")
dataset = dataset["train"].train_test_split(test_size=0.1)
train_dataset = dataset["train"]
valid_dataset = dataset["test"]


# Tokenization function (with labels)
def tokenize_function(examples):
    tokenized_inputs = tokenizer(
        examples["input_text"], padding="max_length", truncation=True, max_length=512
    )
    tokenized_outputs = tokenizer(
        examples["output_text"], padding="max_length", truncation=True, max_length=512
    )

    tokenized_inputs["labels"] = tokenized_outputs["input_ids"]
    return tokenized_inputs


# Tokenize dataset
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_valid = valid_dataset.map(tokenize_function, batched=True)

# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
model.gradient_checkpointing_enable()

# Training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_qwen2.5B",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,  # ⬇ Lower batch size (from 2 → 1)
    per_device_eval_batch_size=1,  # ⬇ Lower evaluation batch size too
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",
    fp16=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    tokenizer=tokenizer,
)

trainer.train()

# ✅ Save the fine-tuned model
# trainer.save_model("fine_tuned_qwen2.5B")
# tokenizer.save_pretrained("fine_tuned_qwen2.5B")