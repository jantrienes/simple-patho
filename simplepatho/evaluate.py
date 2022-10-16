import argparse
import re
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Iterable

import pandas as pd
from easse.bleu import corpus_bleu
from easse.sari import corpus_sari
from nltk import sent_tokenize, word_tokenize
from rouge_score import rouge_scorer
from tqdm.contrib.concurrent import process_map

from simplepatho.preprocessing import novelty

# fmt: off
ALL_METRICS = [
    "R-1", "R-1_p", "R-1_r",
    "R-2", "R-2_p", "R-2_r",
    "R-L", "R-L_p", "R-L_r",
    "BLEU", "SARI",
    "|x|_w", "|x|_s",
    "|y|_w", "|y|_s",
    "CMP_w", "CMP_s",
    "Nov. (n=1)", "Nov. (n=2)",
]
# fmt: on


def load_data(orig_path, refs_path, sys_path):
    with open(orig_path, encoding="utf-8") as fin:
        orig_sents = [l.strip() for l in fin.readlines()]
    with open(refs_path, encoding="utf-8") as fin:
        refs_sents_sents = [l.strip() for l in fin.readlines()]
    with open(sys_path, encoding="utf-8") as fin:
        sys_sents = [l.strip() for l in fin.readlines()]

    print(len(orig_sents))
    print(len(refs_sents_sents))
    print(len(sys_sents))
    assert len(orig_sents) == len(refs_sents_sents) and len(orig_sents) == len(sys_sents)
    return orig_sents, refs_sents_sents, sys_sents


class Aggregator:
    def __init__(self):
        self.stats = defaultdict(list)

    def add(self, scores):
        for metric, value in scores.items():
            self.stats[metric].append(value)


def calculate_statistics(orig_sents: Iterable[str], sys_sents: Iterable[str]):
    """Surface-level statistics. Uses nltk tokenizer/sentence splitter."""
    agg = Aggregator()

    for orig, sys in zip(orig_sents, sys_sents):
        orig_tokens = word_tokenize(orig)
        sys_tokens = word_tokenize(sys)

        n_words_orig = len(orig_tokens)
        n_words_sys = len(sys_tokens)
        cmp_w = n_words_sys / n_words_orig

        n_sents_orig = len(sent_tokenize(orig))
        n_sents_sys = len(sent_tokenize(sys))
        cmp_s = n_sents_sys / n_sents_orig

        novelty_uni = novelty(orig_tokens, sys_tokens, n=1) * 100
        novelty_bi = novelty(orig_tokens, sys_tokens, n=2) * 100

        agg.add(
            {
                "|x|_w": n_words_orig,
                "|x|_s": n_sents_orig,
                "|y|_w": n_words_sys,
                "|y|_s": n_sents_sys,
                "CMP_w": cmp_w,
                "CMP_s": cmp_s,
                "Nov. (n=1)": novelty_uni,
                "Nov. (n=2)": novelty_bi,
            }
        )

    return pd.DataFrame(agg.stats).mean()


def add_newline_to_end_of_each_sentence(x: str) -> str:
    """This was added to get rougeLsum scores matching published rougeL scores for BART and PEGASUS."""
    re.sub("<n>", "", x)  # remove pegasus newline char
    return "\n".join(sent_tokenize(x))


def calculate_rouge(sys_sents, refs_sents):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=False)
    scores = []
    for c, r in zip(sys_sents, refs_sents):
        c = add_newline_to_end_of_each_sentence(c)
        r = add_newline_to_end_of_each_sentence(r)
        s = scorer.score(r, c)
        scores.append(
            {
                "R-1": s["rouge1"].fmeasure,
                "R-1_p": s["rouge1"].precision,
                "R-1_r": s["rouge1"].recall,
                "R-2": s["rouge2"].fmeasure,
                "R-2_p": s["rouge2"].precision,
                "R-2_r": s["rouge2"].recall,
                "R-L": s["rougeLsum"].fmeasure,
                "R-L_p": s["rougeLsum"].precision,
                "R-L_r": s["rougeLsum"].recall,
            }
        )
    df = pd.json_normalize(scores).mean()
    df = df * 100
    return df.to_dict()


def calculate_bleu(sys_sents, refs_sents):
    return corpus_bleu(sys_sents=sys_sents, refs_sents=[[t for t in refs_sents]], lowercase=False)


def calculate_sari(orig_sents, sys_sents, refs_sents):
    return corpus_sari(orig_sents=orig_sents, sys_sents=sys_sents, refs_sents=[[t for t in refs_sents]])


def evaluate_run(orig_sents, refs_sents, sys_sents):
    rouge = calculate_rouge(sys_sents, refs_sents)
    bleu = calculate_bleu(sys_sents, refs_sents)
    sari = calculate_sari(orig_sents, sys_sents, refs_sents)
    surface_stats = calculate_statistics(orig_sents, sys_sents)

    metrics = rouge.copy()
    metrics["BLEU"] = bleu
    metrics["SARI"] = sari
    metrics = pd.Series(metrics)
    metrics = pd.concat([metrics, surface_stats])
    return metrics


def load_and_evaluate_run(orig_path, refs_path, sys_path):
    orig_sents, refs_sents, sys_sents = load_data(orig_path, refs_path, sys_path)
    run_name = Path(sys_path).parent.name
    metrics = evaluate_run(orig_sents, refs_sents, sys_sents)
    metrics.name = run_name
    return metrics


def main(args):
    _evaluate = partial(load_and_evaluate_run, args.orig_path, args.refs_path)
    runs = process_map(_evaluate, args.sys_paths, max_workers=args.n_jobs)
    df = pd.concat(runs, axis=1)
    df = df.loc[args.calculate_metrics]
    # Escape special markdown chars.
    df.rename(index=lambda s: s.replace("|", r"\|").replace("_", r"\_"), inplace=True)
    tab = df.T.to_markdown(floatfmt=f".{args.precision}f")
    if args.compact:
        tab = re.sub(" +", " ", tab)
        tab = re.sub("-+", "-", tab)
    print(tab)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orig_path",
        type=str,
        required=True,
        help="Path to source file. One document per line.",
    )
    parser.add_argument(
        "--refs_path",
        type=str,
        required=True,
        help="Path to target file. One document per line.",
    )
    parser.add_argument(
        "--sys_paths",
        nargs="+",
        type=str,
        required=True,
        help="Path(s) to system runs. One document per line.",
    )
    parser.add_argument("--precision", type=int, required=False, default=2)
    parser.add_argument(
        "--calculate_metrics", type=str, nargs="+", required=False, choices=ALL_METRICS, default=ALL_METRICS
    )
    parser.add_argument("--compact", action="store_true", default=False, required=False)
    parser.add_argument("--n_jobs", type=int, default=1, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    main(arg_parser())
