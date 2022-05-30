import argparse
import random

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from tqdm import tqdm

from rucola.augment import augment
from rucola.util import read_jsonl


class LabeledDataset(Dataset):
    def __init__(self, records, max_tokens, tokenizer):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.records = list()
        for r in tqdm(records):
            inputs = self.embed_record(r["sentence"])
            inputs["labels"] = torch.LongTensor([int(r["acceptable"])])
            self.records.append(inputs)

    def embed_record(self, text):
        inputs = self.tokenizer(
            text=text,
            add_special_tokens=True,
            max_length=self.max_tokens,
            padding="max_length",
            truncation="longest_first",
            return_tensors="pt"
        )
        return {key: value.squeeze(0) for key, value in inputs.items()}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def main(
    train_path,
    max_tokens,
    model_name,
    epochs,
    eval_steps,
    logging_steps,
    warmup_steps,
    lr,
    batch_size,
    grad_accum_steps,
    seed,
    out_dir,
    num_labels
):
    records = read_jsonl(train_path)

    random.seed(seed)
    random.shuffle(records)
    val_border = int(len(records) * 0.9)

    train_records = records[:val_border]
    random.shuffle(train_records)
    train_records = augment(train_records)
    val_records = records[val_border:]

    print("Train records: ", len(train_records))
    print("Val records: ", len(val_records))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = LabeledDataset(train_records, max_tokens, tokenizer)
    val_dataset = LabeledDataset(val_records, max_tokens, tokenizer)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model = model.to(device)

    training_args = TrainingArguments(
        output_dir="checkpoints",
        evaluation_strategy="steps",
        save_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=eval_steps,
        warmup_steps=warmup_steps,
        learning_rate=lr,
        num_train_epochs=epochs,
        gradient_accumulation_steps=grad_accum_steps,
        report_to="none",
        load_best_model_at_end=True,
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--eval-steps", type=int, default=16)
    parser.add_argument("--logging-steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-05)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-labels", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--model-name", type=str, default="sberbank-ai/ruRoberta-large")
    args = parser.parse_args()
    main(**vars(args))

