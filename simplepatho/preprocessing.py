import re
import string
from collections import namedtuple
from itertools import tee, zip_longest
from typing import List

import nltk
import numpy as np
import spacy
import textstat
from nltk import FreqDist, ngrams
from spacy.tokenizer import Tokenizer

EXTRA_ABBRV_DE = ["bspw", "ca", "sog", "ugs", "v.a", "s.o"]
REPLACE_SPECIAL = str.maketrans("", "", string.punctuation + string.digits)


def clean_token(token: str):
    """
    Clean vocabulary: replace tokens which consist of multiple punctuation characters,
    multiple digits or combinations thereof with a special token.
    """
    if len(token) == 1:
        return token
    if not token.translate(REPLACE_SPECIAL):
        return "<SPECIAL>"
    return token


def build_vocab(tokens: List[str], clean=True) -> FreqDist:
    if clean:
        tokens = (clean_token(token) for token in tokens)
    return FreqDist(tokens)


def novel_ngrams(a: List[str], b: List[str], n=2):
    """Count number of n-grams in b but not in a."""
    a = set(ngrams(a, n=n))
    b = set(ngrams(b, n=n))
    novel = len(b - a)
    total = len(b)
    return novel, total


def novelty(a: List[str], b: List[str], n, clean=True):
    """Fraction of n-grams in b but not in a. If there are no n-grams in b, novelty=0.

    Parameters
    ----------
    clean: bool
        When true (default), tokens are sanitized to avoid that numbers and special tokens skew this statistic.
    """
    if clean:
        a = (clean_token(token) for token in a)
        b = (clean_token(token) for token in b)
    novel, total = novel_ngrams(a, b, n=n)
    if total == 0:
        return 0
    return novel / total


class Document:
    def __init__(
        self,
        text: str,
        sents: List[str],
        sents_tokenized: List[List[str]],
        sents_lemmatized: List[List[str]],
    ):
        self.text = text
        self.sents = sents
        self.sents_tokenized = sents_tokenized
        self.sents_lemmatized = sents_lemmatized
        self.tokens = sum(sents_tokenized, [])
        self.lemmas = sum(sents_lemmatized, [])

    def n_tokens(self):
        return len(self.tokens)

    def n_sents(self):
        return len(self.sents)

    def avg_token_len(self):
        lens = [len(token) for token in self.tokens]
        return np.mean(lens)

    def avg_sent_len(self):
        lens = [len(sent) for sent in self.sents_tokenized]
        return np.mean(lens)

    def ttr(self):
        """
        Type-token-ration (TTR) as a measure of lexical diversity.

        TTR is defined the total number of unique words (types) divided by the
        total number of words (tokens) in a given segment of language.
        """
        vocab = build_vocab(self.lemmas)
        return len(vocab) / vocab.N()

    def fre(self):
        """
        We compute Flesch reading-ease scores (Flesch, 1948) with constants adjusted for German (Amstad, 1978).
        - Flesch, R. (1948). A new readability yardstick. Journal of applied psychology, 32(3):221.
        - Amstad, T. (1978). Wie verständlich sind unsere Zeitungen?
        """
        return textstat.flesch_reading_ease(self.text)

    def to_dict(self):
        return {
            "n_tokens": self.n_tokens(),
            "n_sents": self.n_sents(),
            "avg_token_len": self.avg_token_len(),
            "avg_sent_len": self.avg_sent_len(),
            "ttr": self.ttr(),
            "fre": self.fre(),
        }


class ReportTokenizer:
    def __init__(self, language="english", spacy_model="en_core_web_sm"):
        sent_splitter = nltk.data.load(f"tokenizers/punkt/{language}.pickle", cache=False)
        if language == "german":
            sent_splitter._params.abbrev_types.update(EXTRA_ABBRV_DE)

        self.sent_splitter = sent_splitter
        self.language = language
        tagger = spacy.load(spacy_model, disable=["ner"])
        tagger.tokenizer = Tokenizer(tagger.vocab, token_match=re.compile(r"\S+").match)
        self.tagger = tagger

    def escape_enum(self, text):
        return re.sub(r"(\d+)\.", r"<ENUM\1>", text)

    def unescape_enum(self, text):
        return re.sub(r"<ENUM(\d+)>", r"\1.", text)

    def sent_tokenize(self, text):
        text = self.escape_enum(text)
        paragraphs = text.split("\n\n")
        sentences = []
        for paragraph in paragraphs:
            sents = self.sent_splitter.tokenize(paragraph)
            sentences += sents
        sentences = [self.unescape_enum(sent) for sent in sentences]
        return sentences

    def word_tokenize(self, text):
        return nltk.word_tokenize(text, language=self.language)

    def lemmatize(self, sent):
        # pre-tokenize text with nltk
        sent = self.word_tokenize(sent)
        sent = " ".join(sent)
        # get lemmas with spaCy
        lemmas = []
        for token in self.tagger(sent):
            # The Lemmatizer returns `--` for punctuation. We keep original punctuation here.
            lemma = token.text if token.lemma_ == "--" else token.lemma_
            lemmas.append(lemma)
        return lemmas

    def __call__(self, text) -> Document:
        sents = self.sent_tokenize(text)
        sents_tokenized = []
        sents_lemmatized = []
        for sent in sents:
            sents_tokenized.append(self.word_tokenize(sent))
            sents_lemmatized.append(self.lemmatize(sent))
        return Document(text, sents, sents_tokenized, sents_lemmatized)


def corpus_stats(docs: List[Document]):
    """Calculate corpus-level statistics."""
    all_tokens = sum((doc.tokens for doc in docs), [])
    all_lemmas = sum((doc.lemmas for doc in docs), [])

    vocab_raw = build_vocab(all_tokens)
    vocab_lemmas = build_vocab(all_lemmas)

    avg_token_len = np.average([doc.avg_token_len() for doc in docs], weights=[doc.n_tokens() for doc in docs])
    avg_sent_len = np.average([doc.avg_sent_len() for doc in docs], weights=[doc.n_sents() for doc in docs])

    stats = {
        "Docs": len(docs),
        "Tokens": len(all_tokens),
        "Sentences": sum((doc.n_sents() for doc in docs)),
        "Types (raw)": len(vocab_raw),
        "Types (lemmas)": len(vocab_lemmas),
        "Avg. text length (words)": np.mean([doc.n_tokens() for doc in docs]),
        "Avg. text length (sents)": np.mean([doc.n_sents() for doc in docs]),
        "Avg. TTR": np.mean([doc.ttr() for doc in docs]),
        "Avg. Flesch-Reading Ease": np.mean([doc.fre() for doc in docs]),
        "Avg. word length": avg_token_len,
        "Avg. sent length": avg_sent_len,
    }
    return stats


Section = namedtuple(
    "Section",
    ["index", "span", "title", "title_normalized", "title_span", "text", "text_span"],
)


def pairwise_padded(iterable, fillvalue=None):
    "iterable -> (s0,s1), (s1,s2), (s2, s3), ..., (sN, fillvalue)"
    a, b = tee(iterable)
    next(b, None)
    return zip_longest(a, b, fillvalue=fillvalue)


def match_sections(text, pattern: re.Pattern, first_title="[UNNAMED]") -> List[Section]:
    """Assumes that whitespace has been stripped from the left of the string."""
    matches = list(pattern.finditer(text))
    sections = []

    if matches and matches[0].start() != 0:
        # In rare cases, the document does not start with a title that is matched by `RE_TITLE`.
        # In those cases, we track this text in a 'meta' section with title `first_title'.
        first_end = matches[0].start()
        first_section = Section(
            index=0,
            span=(0, first_end),
            title=first_title,
            title_normalized=first_title,
            title_span=None,
            text=text[0:first_end],
            text_span=(0, first_end),
        )
        sections.append(first_section)

    for i, (current, next_) in enumerate(pairwise_padded(matches), start=len(sections)):
        next_start = next_.start() if next_ is not None else len(text)
        section = Section(
            index=i,
            span=(current.start(), next_start),
            title=(current.group(0)),
            title_normalized=(current.group(1)),
            title_span=current.span(),
            text=text[current.end() : next_start],
            text_span=(current.end(), next_start),
        )
        sections.append(section)

    return sections
