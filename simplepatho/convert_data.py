"""Converts raw translation files into jsonlines format for compatibility with Huggingface data loader."""
import argparse
import json
from pathlib import Path


def readlines(in_path):
    with open(in_path, encoding="utf-8") as fin:
        lines = [line.strip() for line in fin.readlines()]
    return lines


def write(src, tgt, out_path):
    with open(out_path, "w", encoding="utf-8") as fout:
        for x, y in zip(src, tgt):
            d = {"source": x, "target": y}
            json.dump(d, fout)
            fout.write("\n")


def main(args):
    raw_path = Path(args.raw_path)
    out_path = Path(args.out_path)
    out_path.mkdir(exist_ok=True, parents=True)

    src = readlines(raw_path / "train.source")
    tgt = readlines(raw_path / "train.target")
    write(src, tgt, out_path=out_path / "train.json")

    src = readlines(raw_path / "val.source")
    tgt = readlines(raw_path / "val.target")
    write(src, tgt, out_path=out_path / "val.json")

    src = readlines(raw_path / "test.source")
    tgt = readlines(raw_path / "test.target")
    write(src, tgt, out_path=out_path / "test.json")


def arg_parser():
    parser = argparse.ArgumentParser(description="Convert raw data into jsonlines format.")
    parser.add_argument("--raw_path", type=str, help="Path to the raw data.")
    parser.add_argument("--out_path", type=str, help="Path to the raw data.")
    return parser.parse_args()


if __name__ == "__main__":
    main(arg_parser())
