import json

def read_jsonl(path):
    records = []
    with open(path) as r:
        for line in r:
            record = json.loads(line)
            records.append(record)
    return records


def write_jsonl(path, records):
    with open(path, "w") as w:
        for record in records:
            w.write(json.dumps(record, ensure_ascii=False) + "\n")
