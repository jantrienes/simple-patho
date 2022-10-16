import argparse
from pathlib import Path

import pandas as pd
import textstat
from tqdm.contrib.concurrent import process_map

from simplepatho.preprocessing import ReportTokenizer, corpus_stats, novelty

INT_FORMAT = "{:,.0f}"
FLOAT_FORMAT = "{:.2f}"

FORMAT_MAPPING = {
    "Docs": INT_FORMAT,
    "Tokens": INT_FORMAT,
    "Sentences": INT_FORMAT,
    "Types (raw)": INT_FORMAT,
    "Types (lemmas)": INT_FORMAT,
    "Avg. text length (words)": INT_FORMAT,
    "Avg. text length (sents)": INT_FORMAT,
    "Avg. TTR": FLOAT_FORMAT,
    "Avg. Flesch-Reading Ease": FLOAT_FORMAT,
    "Avg. word length": FLOAT_FORMAT,
    "Avg. sent length": INT_FORMAT,
}


def format_summary_stats(df):
    rows = []
    index = df.index
    columns = df.columns
    for key in index:
        fmt = FORMAT_MAPPING[key]
        row = df.loc[key].apply(lambda x: fmt.format(x))
        rows.append(row)
    df = pd.DataFrame(rows, index=index, columns=columns)
    return df


def main(args):
    if args.language == "german":
        textstat.set_lang("de_DE")
    elif args.language == "english":
        textstat.set_lang("en_US")

    data_path = Path(args.data_path)
    df = pd.concat(
        [
            pd.read_json(data_path / "train.json", lines=True),
            pd.read_json(data_path / "val.json", lines=True),
            pd.read_json(data_path / "test.json", lines=True),
        ],
        ignore_index=True,
    )

    tokenizer = ReportTokenizer(language=args.language, spacy_model=args.spacy_model)
    docs_source = process_map(tokenizer, df["source"], chunksize=100)
    docs_target = process_map(tokenizer, df["target"], chunksize=100)

    df_stats = pd.concat(
        [pd.Series(corpus_stats(docs_source)), pd.Series(corpus_stats(docs_target))], axis=1, keys=["Source", "Target"]
    )

    novelty_ = []
    for src, tgt in zip(docs_source, docs_target):
        novelty_.append(
            (
                novelty(src.lemmas, tgt.lemmas, n=1) * 100,
                novelty(src.lemmas, tgt.lemmas, n=2) * 100,
                novelty(src.lemmas, tgt.lemmas, n=3) * 100,
            )
        )
    novelty_ = pd.DataFrame(novelty_, columns=["n=1", "n=2", "n=3"])

    compression = (tgt.n_tokens() / src.n_tokens() for src, tgt in zip(docs_source, docs_target))
    compression = pd.Series(compression)

    df_stats_formatted = format_summary_stats(df_stats)
    novelty_str = "/".join(novelty_.mean().round(0).astype(int).astype(str)) + "%"
    df_stats_formatted.loc["Novelty", "Target"] = novelty_str
    df_stats_formatted.loc["CMP", "Target"] = compression.mean().round(2).astype(str)
    df_stats_formatted = df_stats_formatted.fillna("")
    print(df_stats_formatted)


def arg_parser():
    parser = argparse.ArgumentParser(description="Calculate statistics for simplification corpus.")
    parser.add_argument(
        "--data_path", type=str, help="Path to processed data (should have jsonlines files {train/val/test}.json)."
    )
    parser.add_argument("--language", type=str, help="Language of documents.")
    parser.add_argument("--spacy_model", type=str, help="Spacy model to use.")
    return parser.parse_args()


if __name__ == "__main__":
    main(arg_parser())
