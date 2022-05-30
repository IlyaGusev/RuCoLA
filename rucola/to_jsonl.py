import sys
import csv
import json

input_path = sys.argv[1]
output_path = sys.argv[2]

with open(input_path) as r, open(output_path, "w") as w:
    reader = csv.reader(r)
    header = next(reader)
    for row in reader:
        record = dict(zip(header, row))
        w.write(json.dumps(record, ensure_ascii=False) + "\n")
