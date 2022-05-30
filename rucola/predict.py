import argparse
import json

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm


def read_jsonl(path):
    records = []
    with open(path) as r:
        for line in r:
            record = json.loads(line)
            records.append(record)
    return records


def pipe_predict(data, pipe):
    raw_preds = pipe(data, batch_size=64)
    label2id = pipe.model.config.label2id
    y_pred = np.array([label2id[sample["label"]] for sample in raw_preds])
    scores = np.array([sample["score"] for sample in raw_preds])
    return y_pred, scores


def main(
    model_name,
    test_path,
    output_path
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_num = 0 if device == "cuda" else -1

    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    test_records = list(read_jsonl(test_path))

    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, framework="pt", device=device_num)
    y_pred, scores = pipe_predict([r["sentence"] for r in test_records], pipe)
    if "acceptable" in test_records[0]:
        y_true = [int(r["acceptable"]) for r in test_records]

        print("Errors:")
        print("Label 1:")
        for record, label, pred, score in zip(test_records, y_true, y_pred, scores):
            if label != pred and label == 1:
                print("Id: {}, label: {}, prediction: {}, text: {}".format(record["id"], label, pred, record["sentence"]))

        print("Label 0:")
        for record, label, pred, score in zip(test_records, y_true, y_pred, scores):
            if label != pred and label == 0:
                print("Id: {}, label: {}, prediction: {}, text: {}".format(record["id"], label, pred, record["sentence"]))

        print(classification_report(y_true, y_pred, digits=3))
        print("ROC AUC:", roc_auc_score(y_true, y_pred))

    if output_path:
        with open(output_path, "w") as w:
            w.write("id,acceptable\n")
            for pred, record in zip(y_pred, test_records):
                w.write("{},{}\n".format(record["id"], int(pred > 0.5)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--output-path", type=str, default=None)
    args = parser.parse_args()
    main(**vars(args))
