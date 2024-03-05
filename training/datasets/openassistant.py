from datasets import Dataset, DatasetDict, load_dataset
from functools import partial
import pandas as pd
import numpy as np

def get_chat_dataset(ds):
    # lets convert the train dataset to a pandas df
    df = ds.to_pandas()

    # lets grab the message trees to train on 
    message_tree_ids = np.unique(np.array(df["message_tree_id"]))
    messages = {}
    messages['message_tree_id'] = []
    messages['message_tree_text'] = []

    IM_START = "<|im_start|>\n"
    IM_END = "\n<|im_end|>\n"

    for message_tree_id in message_tree_ids:
        try:
            # look at all data for this message tree
            one_message_tree = df.query(f"message_tree_id == '{message_tree_id}'").sort_values("created_date")
            text = ""
            # root message
            text += IM_START
            text += "<human>: " + one_message_tree.iloc[0].text
            text += IM_END
            # find root message's children
            children = one_message_tree[one_message_tree.parent_id == one_message_tree.iloc[0].message_id]
            # find root message's top ranked child:
            child = children[children['rank'] == 0.0]
            text += IM_START
            text += "<bot>: " + child.iloc[0].text
            text += IM_END
            # proceed through rest of the above message tree until completion
            flag=True
            while flag:
                try:
                    # find next prompt
                    children = one_message_tree[one_message_tree.parent_id == child.message_id.iloc[0]]
                    children.index
                    one_message_tree.loc[children.index].iloc[0].role
                    text += IM_START
                    text += "<human>: " + one_message_tree.loc[children.index].iloc[0].text
                    text += IM_END
        
                    # find next children
                    children = one_message_tree[one_message_tree.parent_id == one_message_tree.loc[children.index].iloc[0].message_id]
                    children
                    # find top ranked child:
                    child = children[children['rank'] == 0.0]
                    text += IM_START
                    text += "<bot>: " + child.iloc[0].text
                    text += IM_END
                except:
                    flag=False
        
            messages['message_tree_id'].append(message_tree_id)
            messages['message_tree_text'].append(text)

        except IndexError:
            pass

    message_df = pd.DataFrame.from_dict(messages)

    # convert back to HF datasets format
    ds = Dataset.from_pandas(message_df)
    return ds

def tokenize(element, tokenizer, context_length=1024):
    outputs = tokenizer(
        element["message_tree_text"],
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

def get_openassistant_datasets(tokenizer, context_length=1024):
    # load dataset from huggingface datasets
    ds = load_dataset("OpenAssistant/oasst1")
    ds = ds.filter(lambda x: x['lang'] == 'en')

    train_dataset = get_chat_dataset(ds['train'])
    val_dataset = get_chat_dataset(ds['validation'])

    ds = DatasetDict({"train": train_dataset, 
                      "test": val_dataset})

    tokenize_fn = partial(tokenize, tokenizer=tokenizer, context_length=context_length)

    tokenized_datasets = ds.map(
        tokenize_fn, batched=True, remove_columns=ds['train'].column_names
    )
    
    return tokenized_datasets