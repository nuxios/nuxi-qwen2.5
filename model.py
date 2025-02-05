from transformers import pipeline

# Input teks
input_text = "Can you explain what it means to be human in simple terms?"

# Inisialisasi pipeline
pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B",
    device=0,  # Gunakan GPU
)

# Generate teks
output = pipe(
    input_text,
    max_new_tokens=100,        # Tingkatkan panjang maksimum output
    temperature=0.7,           # Variasi teks
    top_k=50,                  # Sampling token teratas
    top_p=0.9,                 # Nucleus sampling
    repetition_penalty=1.2,    # Penalti untuk pengulangan
    do_sample=True,            # Aktifkan sampling
    truncation=True            # Potong input jika terlalu panjang
)

# Cetak hasil
print(output[0]['generated_text'])