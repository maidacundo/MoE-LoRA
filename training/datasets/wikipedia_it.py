from datasets import load_dataset
from functools import partial

def tokenize(element, tokenizer, context_length=128):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
        padding=True,
    )
    input_batch = []
    attention_mask_batch = []
    for input_ids, attention_mask in zip(outputs["input_ids"], outputs["attention_mask"]):
        input_batch.append(input_ids)
        attention_mask_batch.append(attention_mask)
    return {"input_ids": input_batch,
            "attention_mask": attention_mask_batch,}

def get_wikipedia_datasets(tokenizer, context_length=128, num_train_texts=5000, min_length=10):

    wikipedia_ds = load_dataset('maidacundo/wikipedia-it', split=f'train[:{num_train_texts}]')
    wikipedia_ds = wikipedia_ds.train_test_split(test_size=0.01)

    tokenize_fn = partial(tokenize, tokenizer=tokenizer, context_length=context_length)

    tokenized_datasets = wikipedia_ds.map(
        tokenize_fn, batched=True, remove_columns=wikipedia_ds['train'].column_names
    )

    tokenized_datasets = tokenized_datasets.filter(lambda x: sum(x['attention_mask']) > min_length)

    return tokenized_datasets