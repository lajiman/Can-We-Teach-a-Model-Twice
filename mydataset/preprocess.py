from transformers import AutoTokenizer

model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
max_input_length = 512
max_target_length = 50

def preprocess_function(examples):  # 根据huggingface提供的数据分布进行截断。
    model_inputs = tokenizer(
        examples["text"], max_length=max_input_length, padding = "max_length", truncation=True
    )
    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["summary"], max_length=max_target_length, truncation=True
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def tokenizer_datasets(dataset):
    return dataset.map(preprocess_function, batched=True)