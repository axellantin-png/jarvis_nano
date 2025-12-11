from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset

# 1. Charger ton tokenizer HF
tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer_hf")

# 2. Charger un mini modèle à fine-tuner
model = AutoModelForCausalLM.from_pretrained("gpt2")  # ou un modèle FR

# 3. Charger ton dataset JSONL
dataset = load_dataset("json", data_files="data/jarvis_instructions.jsonl")

# 4. Fonction pour transformer une ligne en texte
def format_example(example):
    user = example["instruction"]
    assistant = example["response"]
    text = f"<|user|>{user}\n<|assistant|>{assistant}"
    tokens = tokenizer(text, truncation=True, padding="max_length", max_length=256)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(format_example)

# 5. Paramètres d’entraînement
args = TrainingArguments(
    output_dir="models/jarvis_first",
    per_device_train_batch_size=2,
    learning_rate=5e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=200,
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],
)

trainer.train()

trainer.save_model("models/jarvis_first")
print("✔ Entraînement terminé !")
