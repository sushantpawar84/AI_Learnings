import json
from transformers import AutoTokenizer

# Load tokenizer (replace with your model's tokenizer)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Path to your JSONL dataset
input_file = "C:/Sushant/AI Learning/Datasets/java_dataset.jsonl"
output_file = "C:/Sushant/AI Learning/Datasets/java_dataset_tokenized.jsonl"

# Read dataset
dataset = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        dataset.append(json.loads(line))

# Tokenize all entries
tokenized_dataset = []
max_len = 0

for entry in dataset:
    tokenized = tokenizer(
        entry["input"],
        text_target=entry["output"],
        truncation=False,   # keep everything
        add_special_tokens=True
    )
    max_len = max(max_len, len(tokenized["input_ids"]))
    tokenized_dataset.append(tokenized)

# Pad all entries to max_len
for tokenized in tokenized_dataset:
    pad_len = max_len - len(tokenized["input_ids"])
    if pad_len > 0:
        tokenized["input_ids"] += [tokenizer.pad_token_id] * pad_len
        tokenized["attention_mask"] += [0] * pad_len
    # keep labels aligned too
    if "labels" in tokenized:
        pad_len_labels = max_len - len(tokenized["labels"])
        tokenized["labels"] += [-100] * pad_len_labels  # ignore index for padding

# Save back to JSONL
with open(output_file, "w", encoding="utf-8") as f:
    for tokenized in tokenized_dataset:
        serializable = dict(tokenized)
        f.write(json.dumps(serializable) + "\n")

print(f"Tokenized dataset saved to {output_file}, all sequences padded to length {max_len}")

