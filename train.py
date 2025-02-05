from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from datasets import load_dataset
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.nn import CrossEntropyLoss

# ✅ Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# ✅ Load dataset
dataset = load_dataset("json", data_files="dataset.json", cache_dir=None)
train_dataset = dataset["train"]
valid_dataset = dataset["train"]


# ✅ Tokenization function
def tokenize_function(examples):
    inputs = tokenizer(
        examples["input_text"], padding="max_length", truncation=True, max_length=256
    )
    outputs = tokenizer(
        examples["output_text"], padding="max_length", truncation=True, max_length=256
    )

    inputs["labels"] = outputs["input_ids"]  # ✅ Pastikan labels benar
    return inputs


# ✅ Tokenize dataset
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_valid = valid_dataset.map(tokenize_function, batched=True)

# ✅ Gunakan 4-bit quantization dengan bitsandbytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_quant_type="nf4",  # ✅ Pakai Normal Float 4 untuk akurasi lebih baik
)

# ✅ Load model dengan 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B", quantization_config=bnb_config
)

# ✅ Pastikan model bisa di-train meskipun di-quantized (4-bit)
model = prepare_model_for_kbit_training(model)

# ✅ Tambahkan LoRA untuk fine-tuning yang lebih cepat & akurat
lora_config = LoraConfig(
    r=16,  # ✅ Tambah rank supaya model bisa belajar lebih banyak
    lora_alpha=64,  # ✅ Tambah alpha agar lebih stabil
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, lora_config)

# ✅ Optimized training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_qwen2.5B",
    eval_strategy="epoch",
    learning_rate=5e-6,  # ✅ Turunkan agar model lebih stabil
    warmup_steps=500,  # ✅ Supaya learning rate naik pelan-pelan
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,  # ✅ Lebih stabil
    num_train_epochs=5,  # ✅ Tambah epoch agar model bisa lebih banyak belajar
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",
    fp16=True,
)


# ✅ Custom loss function agar lebih stabil
def compute_loss(model, inputs):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.logits

    loss_fct = CrossEntropyLoss()
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    return loss


# ✅ Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    tokenizer=tokenizer,
    # compute_loss=compute_loss,  # ✅ Gunakan loss yang lebih stabil
)

# ✅ Jalankan Training!
trainer.train()

# ✅ Simpan model setelah training selesai
trainer.save_model("fine_tuned_qwen2.5B")
tokenizer.save_pretrained("fine_tuned_qwen2.5B")
