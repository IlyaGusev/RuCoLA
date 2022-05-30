import argparse
import random
import sys
import copy

import razdel
import pymorphy2

from rucola.util import read_jsonl, write_jsonl

morph = pymorphy2.MorphAnalyzer()

LOW_PROB = 0.00
HIGH_PROB = 0.14

settings = {
    "NOUN": {
        "case": {
            "func": lambda x: x.tag.case,
            "prob": HIGH_PROB,
            "values": ("nomn", "gent", "datv", "ablt", "loct")
        },
        "number": {
            "func": lambda x: x.tag.number,
            "prob": HIGH_PROB,
            "values": ("sing", "plur")
        }
    },
    "VERB": {
        "number": {
            "func": lambda x: x.tag.number,
            "prob": LOW_PROB,
            "values": ("sing", "plur")
        }
    },
    "ADJF": {
        "case": {
            "func": lambda x: x.tag.case,
            "prob": LOW_PROB,
            "values": ("nomn", "gent", "datv", "ablt", "loct")
        },
        "number": {
            "func": lambda x: x.tag.number,
            "prob": LOW_PROB,
            "values": ("sing", "plur")
        },
        "gender": {
            "func": lambda x: x.tag.gender,
            "prob": LOW_PROB,
            "values": ("masc", "femn", "neut")
        }
    },
}


def detokenize(tokens, parses):
    new_text = ""
    prev_stop = 0
    for token, parse in zip(tokens, parses):
        start = token.start
        stop = token.stop
        delim = "" if start == prev_stop else " "
        is_upper = token.text[0].isupper()
        new_token = parse[0].word
        if is_upper:
            new_token = new_token[0].upper() + new_token[1:]
        new_text += delim + new_token
        prev_stop = stop
    return new_text


def augment_acceptable(record):
    assert int(record["acceptable"]) == 1
    text = record["sentence"]
    tokens = list(razdel.tokenize(text))
    parses = [morph.parse(t.text) for t in tokens]
    was_changed = False
    for i, parse in enumerate(parses):
        parse = parse[0]
        pos_settings = settings.get(parse.tag.POS)
        if not pos_settings:
            continue
        if i == 0:
            continue
        for key, gram_setings in pos_settings.items():
            current_value = gram_setings["func"](parse)
            prob = gram_setings["prob"]
            values = gram_setings["values"]
            new_value = random.choice(values)
            while new_value == current_value:
                new_value = random.choice(values)
            new_parse = parse.inflect({new_value})
            if not new_parse:
                continue
            is_changed = new_parse.word.lower() != parse.word.lower()
            if is_changed and random.random() < prob:
                parses[i] = [new_parse]
                was_changed = True
            if was_changed:
                break
        if was_changed:
            break

    if was_changed:
        new_text = detokenize(tokens, parses)
        #print("Old:", text)
        #print("New:", new_text)
        return {
            "sentence": new_text,
            "acceptable": "0",
            "id": record["id"] + "_corrupted"
        }


def build_random_pairs(records1, records2, prob):
    paired = []
    for idx1 in range(len(records1)):
        if random.random() > prob:
            continue
        idx2 = random.randrange(len(records2))
        new_text = records1[idx1]["sentence"] + " " + records2[idx2]["sentence"]
        #print("Paired:", new_text)
        paired.append({
            "sentence": new_text,
            "acceptable": records1[idx1]["acceptable"],
            "id": "concat_" + records1[idx1]["id"] + "_" + records2[idx2]["id"]
        })
    return paired


def augment_concat(records, prob):
    acceptable_records = [r for r in records if int(r["acceptable"]) == 1]
    not_acceptable_records = [r for r in records if int(r["acceptable"]) == 0]

    new_records = copy.copy(records)
    new_records.extend(build_random_pairs(acceptable_records, acceptable_records, prob))
    new_records.extend(build_random_pairs(not_acceptable_records, not_acceptable_records, prob))
    new_records.extend(build_random_pairs(not_acceptable_records, acceptable_records, prob))
    return new_records


def augment(records):
    print("Orig length:", len(records))

    records = augment_concat(records, 0.1)
    print("After concat:", len(records))

    new_records = []
    for record in records:
        new_records.append(record)
        if int(record["acceptable"]) == 1:
            new_record = augment_acceptable(record)
            if new_record:
                new_records.append(new_record)
    print("Final:", len(new_records))
    return new_records


def main(input_path, output_path):
    records = read_jsonl(input_path)
    new_records = augment(records)
    write_jsonl(output_path, new_records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()
    main(**vars(args))
