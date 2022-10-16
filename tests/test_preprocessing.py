import re

import nltk

from simplepatho.preprocessing import (
    ReportTokenizer,
    Section,
    build_vocab,
    clean_token,
    match_sections,
    novelty,
)

tokenizer_en = ReportTokenizer(language="english", spacy_model="en_core_web_sm")
tokenizer_de = ReportTokenizer(language="german", spacy_model="de_core_news_sm")


def test_escape_enum():
    txt = "1. This is an item. 2. This is another item 3.4.5"
    expected = "<ENUM1> This is an item. <ENUM2> This is another item <ENUM3><ENUM4>5"
    assert tokenizer_en.escape_enum(txt) == expected
    assert tokenizer_en.unescape_enum(tokenizer_en.escape_enum(txt)) == txt


def test_nltk_sent_tokenize():
    # Default NLTK sentence splitter does not handle enumerations correctly.
    txt = "1. This is an item. 2. This is another item. 3. This is the third item."
    assert nltk.sent_tokenize(txt) == [
        "1.",
        "This is an item.",
        "2.",
        "This is another item.",
        "3.",
        "This is the third item.",
    ]


def test_sent_tokenize_with_enums():
    txt = "1. This is an item. 2. This is another item. 3. This is the third item 1.2.3"
    assert tokenizer_en.sent_tokenize(txt) == [
        "1. This is an item.",
        "2. This is another item.",
        "3. This is the third item 1.2.3",
    ]


def test_sent_tokenize_with_paragraphs():
    # In default NLTK, a sentence can span across paragraphs
    actual = nltk.sent_tokenize("this is the first pargraphs\n\nand this is the second paragraph")
    expected = ["this is the first pargraphs\n\nand this is the second paragraph"]
    assert actual == expected

    # We want to start new sentences upon a new paragraph
    actual = tokenizer_en.sent_tokenize("this is the first pargraphs\n\nand this is the second paragraph")
    expected = ["this is the first pargraphs", "and this is the second paragraph"]
    assert actual == expected


def test_sent_tokenize_de_abbrevs():
    assert tokenizer_de.sent_tokenize("Es enthält einen ca. 1 cm großen Rückstand. Zweiter Satz.") == [
        "Es enthält einen ca. 1 cm großen Rückstand.",
        "Zweiter Satz.",
    ]

    assert tokenizer_de.sent_tokenize("Im Zentrum gibt es sog. Florett-Zellen. Zweiter Satz.") == [
        "Im Zentrum gibt es sog. Florett-Zellen.",
        "Zweiter Satz.",
    ]

    assert tokenizer_de.sent_tokenize(
        "Es konnte ein Riesenzelltumor diagnostiziert werden (Erklärung s.o.). Zweiter Satz."
    ) == [
        "Es konnte ein Riesenzelltumor diagnostiziert werden (Erklärung s.o.).",
        "Zweiter Satz.",
    ]

    assert tokenizer_de.sent_tokenize("Der Tumor (ugs. Krebs), geht vom Weichgewebe aus. Zweiter Satz.") == [
        "Der Tumor (ugs. Krebs), geht vom Weichgewebe aus.",
        "Zweiter Satz.",
    ]


def test_clean_token():
    assert clean_token("31,5") == "<SPECIAL>"
    assert clean_token("753") == "<SPECIAL>"
    assert clean_token("1/2") == "<SPECIAL>"
    assert clean_token("12-16") == "<SPECIAL>"
    assert clean_token("......") == "<SPECIAL>"
    assert clean_token("+++") == "<SPECIAL>"
    assert clean_token("25.02.2021") == "<SPECIAL>"
    assert clean_token(".") == "."
    assert clean_token("do") == "do"
    assert clean_token("nothing") == "nothing"
    assert clean_token("CDK4") == "CDK4"
    assert clean_token("1") == "1"


def test_document():
    d = tokenizer_de("Dies sind zwei Sätze. Dies ist der zweite Satz.")
    assert d.text == "Dies sind zwei Sätze. Dies ist der zweite Satz."
    assert d.sents == ["Dies sind zwei Sätze.", "Dies ist der zweite Satz."]
    assert d.tokens == ["Dies", "sind", "zwei", "Sätze", ".", "Dies", "ist", "der", "zweite", "Satz", "."]
    assert d.lemmas == ["dieser", "sein", "zwei", "Satz", ".", "dieser", "sein", "der", "zweiter", "Satz", "."]
    assert d.sents_tokenized == [["Dies", "sind", "zwei", "Sätze", "."], ["Dies", "ist", "der", "zweite", "Satz", "."]]
    assert d.sents_lemmatized == [
        ["dieser", "sein", "zwei", "Satz", "."],
        ["dieser", "sein", "der", "zweiter", "Satz", "."],
    ]

    assert d.n_sents() == 2
    assert d.n_tokens() == 11
    assert d.avg_token_len() == sum(len(token) for token in d.tokens) / len(d.tokens)
    assert d.avg_sent_len() == (5 + 6) / 2
    assert d.ttr() == len(set(d.lemmas)) / len(d.tokens)
    assert d.fre() > 0


def test_build_vocab():
    tokens = ["12.4", "test", "test", "word", "22", "..."]
    vocab = build_vocab(tokens)
    assert vocab == {
        "<SPECIAL>": 3,
        "test": 2,
        "word": 1,
    }
    vocab = build_vocab(tokens, clean=False)
    assert vocab == {"12.4": 1, "test": 2, "word": 1, "22": 1, "...": 1.0}


def test_novelty():
    a = "this is a test".split()
    b = "this is another test".split()
    assert novelty(a, b, n=1) == 1 / 4
    assert novelty(a, b, n=2) == 2 / 3
    assert novelty(a, b, n=5) == 0, "When there are no ngrams of length n in b, novelty should be 0 per definition."


def test_match_sections():
    title_pattern = (
        # A section title starts either at the beginning of the document,
        # or at two preceding line breaks.
        r"(?:^|\n{2,})"
        # Start capture group for section title
        r"("
        # A title needs to consist of 1-3 words (min. 3 characters per word) and it should end with a ':' or '.'.
        r"(?:\w{3,}\s){0,2}\w{3,}[:.]"
        # End of capture group
        r")"
        # Match up to one line break after the title
        r"\n?"
    )
    title_pattern = re.compile(title_pattern)

    s = (
        "Klinische gegevens:\n"
        "Follow up mammacarcinoom L wv Ziekenhuis X\n\n"
        "Verslag:\n"
        "Mammogram beiderzijds in twee richtingen.\n\n"
        "Lorem ipsum dolor sit amet.\n\n"
        "Conclusie.\n"
        "BIRADS-II"
    )

    assert match_sections(s, pattern=title_pattern) == [
        Section(
            index=0,
            span=(0, 62),
            title="Klinische gegevens:\n",
            title_normalized="Klinische gegevens:",
            title_span=(0, 20),
            text="Follow up mammacarcinoom L wv Ziekenhuis X",
            text_span=(20, 62),
        ),
        Section(
            index=1,
            span=(62, 143),
            title="\n\nVerslag:\n",
            title_normalized="Verslag:",
            title_span=(62, 73),
            text="Mammogram beiderzijds in twee richtingen.\n\nLorem ipsum dolor sit amet.",
            text_span=(73, 143),
        ),
        Section(
            index=2,
            span=(143, len(s)),
            title="\n\nConclusie.\n",
            title_normalized="Conclusie.",
            title_span=(143, 156),
            text="BIRADS-II",
            text_span=(156, len(s)),
        ),
    ]


def test_match_sections_first_empty():
    pattern = r"\n\n(Conclusie.)\n"
    pattern = re.compile(pattern)
    s = "Some text before the first section.\n\nConclusie.\nBIRAD"

    assert match_sections(s, pattern=pattern) == [
        Section(
            index=0,
            span=(0, 35),
            title="[UNNAMED]",
            title_normalized="[UNNAMED]",
            title_span=None,
            text="Some text before the first section.",
            text_span=(0, 35),
        ),
        Section(
            index=1,
            span=(35, len(s)),
            title="\n\nConclusie.\n",
            title_normalized="Conclusie.",
            title_span=(35, 48),
            text="BIRAD",
            text_span=(48, len(s)),
        ),
    ]


def test_match_sections_no_match():
    pattern = r"\n\n(Conclusie.)\n"
    pattern = re.compile(pattern)
    s = "This text does not have any sections that match the pattern"
    assert match_sections(s, pattern=pattern) == []


def test_match_sections_one():
    pattern = r"(Conclusie.)\n"
    pattern = re.compile(pattern)
    s = "Conclusie.\nThis test only has one section that matches the pattern."
    assert match_sections(s, pattern=pattern) == [
        Section(
            index=0,
            span=(0, len(s)),
            title="Conclusie.\n",
            title_normalized="Conclusie.",
            title_span=(0, 11),
            text="This test only has one section that matches the pattern.",
            text_span=(11, len(s)),
        )
    ]
