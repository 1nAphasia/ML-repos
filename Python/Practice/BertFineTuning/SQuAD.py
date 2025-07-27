from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    BertPreTrainedModel,
    BertModel,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import DataLoader
import torch.nn as nn


def process_example(example):
    context = example["context"]
    question = example["question"]
    answers = example["answers"]

    is_impossible = len(answers["text"]) == 0

    if is_impossible:
        return {
            "context": context,
            "question": question,
            "answer_start": None,
            "answer_text": None,
        }
    else:
        answer_start = answers["answer_start"][0]
        answer_text = answers["text"][0]
        return {
            "context": context,
            "question": question,
            "answer_start": answer_start,
            "answer_text": answer_text,
        }


tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


def tokenize_func(target, tokenizer=tokenizer):
    tokenized_inputs = tokenizer(
        target["question"],
        target["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = tokenized_inputs.pop("offset_mapping")

    tokenized_inputs["start_positions"] = []
    tokenized_inputs["end_positions"] = []
    tokenized_inputs["is_impossible"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_inputs["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        if target["answer_start"] is None:
            tokenized_inputs["start_positions"].append(cls_index)
            tokenized_inputs["end_positions"].append(cls_index)
            tokenized_inputs["is_impossible"].append(True)

        else:
            start_char = target["answer_start"]
            end_char = start_char + len(target["answer_text"])
            token_start_index = 0
            while offsets[token_start_index][0] < start_char:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while offsets[token_end_index][1] > end_char:
                token_end_index -= 1

            if (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_inputs["start_positions"].append(token_start_index)
                tokenized_inputs["end_positions"].append(token_end_index)
                tokenized_inputs["is_impossible"].append(False)
            else:
                tokenized_inputs["start_positions"].append(cls_index)
                tokenized_inputs["end_positions"].append(cls_index)
                tokenized_inputs["is_impossible"].append(True)
    return tokenized_inputs


if __name__ == "__main__":

    dataset = load_dataset("squad_v2")

    train_data = dataset["train"].map(process_example, num_proc=6)
    test_data = dataset["validation"].map(process_example, num_proc=6)

    train_encoding = train_data.map(
        tokenize_func,
        batched=True,
        num_proc=6,
    )
    test_encoding = test_data.map(
        tokenize_func,
        batched=True,
        num_proc=6,
    )

    print(train_encoding)
