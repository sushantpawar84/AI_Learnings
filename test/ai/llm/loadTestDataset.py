import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Load your JSONL file
dataset = load_dataset(
    "json",
    data_files="C:/Sushant/AI Learning/Datasets/java_dataset_tokenized.jsonl",
    split="train"
)
print(dataset[0])

def clean_example(example):
    pad_id = tokenizer.pad_token_id
    example["input_ids"] = [
        pad_id if x is None else x for x in example["input_ids"]
    ]
    example["attention_mask"] = [
        0 if x is None else x for x in example["attention_mask"]
    ]
    example["labels"] = [
        -100 if x is None else x for x in example["labels"]
    ]
    return example

dataset = dataset.map(clean_example)
dataset.set_format("torch")

train_dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)


training_args = TrainingArguments(
    output_dir="./gpt2-java",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=50,
    save_steps=500,
    fp16=True if torch.cuda.is_available() else False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()

prompt = "What are the url patterns in HelloWorldSrv1Controller.java"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))